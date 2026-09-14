"""Per-layer sensitivity ranking WITHOUT the BF16 source: which MoE layers deserve K=4?

sc_measure.py needs fp16 weights (it perturbs them in place), so it cannot run here. The same
logic works relatively in ACTIVATION space on the quantized pack: perturb ONE layer's MoE output
with noise at a fixed relative Frobenius norm, teacher-force fixed token sequences, and measure KL
of the final logits against the clean run. Layers where the same perturbation costs more KL are
where cutting quantization error (K=3 -> K=4) should buy most.

CAVEATS, both established empirically:
  - activation-space != weight-space; the map depends on each layer's input covariance.
  - the ORDERING is not scale-invariant (eps 0.01 vs 0.02 reorder the top of the list), and the
    KL response is sub-quadratic in eps, so trust the SET of early layers, not fine ordering.
Usage: layer_sens.py [eps] [rows] [len]
"""
import sys, torch
sys.path.insert(0, "/root")
MODEL = "/root/workspace/GLM-5.3-Flash-exl3-3.05bpw"
from sens_text import TEXT
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
NROW = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 512

def kl(p_logits, q_logits):
    p = torch.log_softmax(p_logits.float(), dim=-1)
    q = torch.log_softmax(q_logits.float(), dim=-1)
    return (p.exp() * (p - q)).sum(-1).mean().item()

def layer_of(m):
    p = m.key.split(".")
    for i, s in enumerate(p):
        if s == "layers":
            return int(p[i + 1])
    return -1

def main():
    rows = TEXT[:NROW]
    assert len(rows) == NROW, f"only {len(rows)} texts available, asked for {NROW}"
    config = Config.from_directory(MODEL)
    tok = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_batch_size=1, max_history=0)
    model.load(use_per_device=[90.0, 56.0], progressbar=False)
    print("loaded; eps=%.3f rows=%d len=%d" % (EPS, len(rows), LEN), flush=True)

    found = []
    def walk(m):
        if isinstance(m, BlockSparseMLP):
            found.append(m)
        for sm in m.modules:
            walk(sm)
    for m in model.modules:
        walk(m)
    found = sorted(found, key=layer_of)
    print("BlockSparseMLP modules: %d (layers %d..%d)"
          % (len(found), layer_of(found[0]), layer_of(found[-1])), flush=True)

    ids = [tok.encode(t, add_bos=True)[:, :LEN] for t in rows]
    def run_all():
        return [model.forward(x, params={"last_tokens_only": min(64, x.shape[1])})[0].clone()
                for x in ids]

    with torch.inference_mode():
        clean = run_all()
        print("clean reference captured\n", flush=True)
        print("  %-7s %-14s" % ("layer", "KL(noise@layer)"), flush=True)
        results = []
        for m in found:
            L = layer_of(m)
            orig = m.forward
            def noisy(x, params, _o=orig):
                y = _o(x, params)
                n = torch.randn_like(y.float())
                n = n / n.norm() * y.float().norm() * EPS
                return (y.float() + n).to(y.dtype)
            m.forward = noisy
            torch.manual_seed(1234)
            try:
                pert = run_all()
                d = sum(kl(c, p) for c, p in zip(clean, pert)) / len(clean)
            finally:
                m.forward = orig
            results.append((d, L))
            print("  %-7d %.6e" % (L, d), flush=True)

    results.sort(reverse=True)
    print("\n=== RANKED most-sensitive first ===", flush=True)
    print("  " + " ".join(str(L) for _, L in results))
    ends = sorted(range(3, 45), key=lambda L: (min(L - 3, 44 - L), L))
    for n in (18, 20):
        meas = sorted(L for _, L in results[:n])
        pos = sorted(ends[:n])
        print("\n  top %d measured:   %s" % (n, meas))
        print("  top %d positional: %s" % (n, pos))
        print("      overlap: %d / %d" % (len(set(meas) & set(pos)), n))

if __name__ == "__main__":
    main()
