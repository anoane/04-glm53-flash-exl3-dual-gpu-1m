"""Residual-normalised variant: the criticism that could flip the early>late conclusion.

The main sweep scaled noise to each layer's OWN MoE output norm. If later layers contribute less
into the residual stream (mHC hyper-connections, hc_mult=4), that scaling understates them and
would manufacture the observed monotonic decay. Here every layer gets the SAME ABSOLUTE
perturbation, taken from one global reference norm captured at the first MoE layer.

If early>late survives BOTH normalisations it is a property of the model.
If it flattens or flips here, the decay was an artefact and exl3's positional rule stands.
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
    config = Config.from_directory(MODEL)
    tok = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_batch_size=1, max_history=0)
    model.load(use_per_device=[90.0, 56.0], progressbar=False)

    found = []
    def walk(m):
        if isinstance(m, BlockSparseMLP):
            found.append(m)
        for sm in m.modules:
            walk(sm)
    for m in model.modules:
        walk(m)
    found = sorted(found, key=layer_of)

    ids = [tok.encode(t, add_bos=True)[:, :LEN] for t in rows]
    def run_all():
        return [model.forward(x, params={"last_tokens_only": min(64, x.shape[1])})[0].clone()
                for x in ids]

    # one global reference norm, from the first MoE layer's output
    ref = {}
    m0 = found[0]
    orig0 = m0.forward
    def probe(x, params, _o=orig0):
        y = _o(x, params)
        ref.setdefault("n", y.float().norm().item())
        return y
    m0.forward = probe
    with torch.inference_mode():
        clean = run_all()
    m0.forward = orig0
    REF = ref["n"]
    print("loaded; eps=%.3f rows=%d  GLOBAL ref norm (layer %d out) = %.4f\n"
          % (EPS, len(rows), layer_of(m0), REF), flush=True)

    with torch.inference_mode():
        print("  %-7s %-14s %-14s" % ("layer", "KL(abs-noise)", "own-norm"), flush=True)
        results = []
        for m in found:
            L = layer_of(m)
            orig = m.forward
            own = {}
            def noisy(x, params, _o=orig, _own=own):
                y = _o(x, params)
                yn = y.float().norm()
                _own.setdefault("n", yn.item())
                n = torch.randn_like(y.float())
                n = n / n.norm() * REF * EPS      # SAME absolute magnitude for every layer
                return (y.float() + n).to(y.dtype)
            m.forward = noisy
            torch.manual_seed(1234)
            try:
                pert = run_all()
                d = sum(kl(c, p) for c, p in zip(clean, pert)) / len(clean)
            finally:
                m.forward = orig
            results.append((d, L))
            print("  %-7d %.6e   %.4f" % (L, d, own.get("n", float("nan"))), flush=True)

    results.sort(reverse=True)
    print("\n=== RANKED (residual-normalised) most-sensitive first ===", flush=True)
    print("  " + " ".join(str(L) for _, L in results))
    ends = sorted(range(3, 45), key=lambda L: (min(L - 3, 44 - L), L))
    for n in (18, 20):
        meas = sorted(L for _, L in results[:n])
        pos = sorted(ends[:n])
        print("\n  top %d residual-normalised: %s" % (n, meas))
        print("  top %d positional:          %s" % (n, pos))
        print("      overlap: %d / %d" % (len(set(meas) & set(pos)), n))
    print("\n  (per-layer-normalised top20 was [3..22]; if this run agrees, early>late is real)")

if __name__ == "__main__":
    main()
