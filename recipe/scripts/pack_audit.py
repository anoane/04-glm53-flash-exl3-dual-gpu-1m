"""Independent audit of a built pack: prove it is what FRANKENSTEIN.json claims.

Reads the expected K=4 layer set from the pack's own FRANKENSTEIN.json instead of a hardcoded
range -- the previous version asserted "K=4 for 3..22" and so reported the post-revert pack
(now [3..18]) as corrupt when it was correct.

Checks every shard's header parses and ends exactly at file size, then reports the per-layer
K histogram from tensor geometry alone (K = trellis.shape[-1] // 16).
Usage: pack_audit.py <pack_dir>
"""
import json, os, struct, sys, collections

D = sys.argv[1]

try:
    meta = json.load(open(os.path.join(D, "FRANKENSTEIN.json")))
    EXPECT4 = set(meta.get("upgraded_layers") or [])
    src = "FRANKENSTEIN.json"
except Exception:
    EXPECT4 = set()
    src = "none found -> expecting all K=3"
print("expected K=4 layers (%s): %s" % (src, sorted(EXPECT4)))

# The MTP head is quantized at mtp_bits, NOT the base bits -- exempt it from the K=3 rule.
MTP_LAYER, MTP_K = None, None
try:
    _cfg = json.load(open(os.path.join(D, "config.json")))
    MTP_LAYER = (_cfg.get("text_config") or _cfg).get("num_hidden_layers")
    MTP_K = json.load(open(os.path.join(D, "quantization_config.json"))).get("mtp_bits")
except Exception:
    pass
if MTP_LAYER is not None:
    print("MTP head layer %s expected at K=%s (mtp_bits)" % (MTP_LAYER, MTP_K))

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > (256 << 20):
            raise ValueError("bad header length %d" % n)
        return json.loads(f.read(n)), 8 + n

def layer_of(k):
    p = k.split(".")
    try:
        return int(p[p.index("layers") + 1])
    except (ValueError, IndexError):
        return None

shards = sorted(f for f in os.listdir(D) if f.endswith(".safetensors"))
print("shards: %d" % len(shards))
bad = []
perlayer = collections.defaultdict(collections.Counter)
total = 0
for sh in shards:
    p = os.path.join(D, sh)
    sz = os.path.getsize(p)
    total += sz
    try:
        hdr, dstart = read_header(p)
    except Exception as e:
        bad.append((sh, "header: %s" % e)); print("  %-34s HEADER FAIL" % sh); continue
    keys = [k for k in hdr if k != "__metadata__"]
    end = max(hdr[k]["data_offsets"][1] for k in keys)
    exact = (dstart + end == sz)
    if not exact:
        bad.append((sh, "size mismatch header_end=%d file=%d" % (dstart + end, sz)))
    print("  %-34s %6.2f GiB  %6d tensors  exact_end=%s" % (sh, sz / 2**30, len(keys), exact))
    for k in keys:
        if k.endswith(".trellis") and ".mlp.experts." in k:
            L = layer_of(k)
            if L is not None:
                perlayer[L][hdr[k]["shape"][-1] // 16] += 1

wrong = []
k4, k3 = [], []
for L in sorted(perlayer):
    h = dict(perlayer[L])
    if MTP_LAYER is not None and L == MTP_LAYER and MTP_K:
        want = MTP_K          # MTP head follows mtp_bits
    else:
        want = 4 if L in EXPECT4 else 3
    ok = (list(h.keys()) == [want]) and sum(h.values()) == 864
    (k4 if want == 4 else k3).append(L)
    if not ok:
        wrong.append((L, h, want))

print("\nper-layer K: %d layers at K=4 %s" % (len(k4), sorted(k4)))
print("             %d layers at K=3 %s" % (len(k3), (sorted(k3)[:6] + ["..."]) if len(k3) > 6 else sorted(k3)))
print("pack total: %.2f GiB" % (total / 2**30))
if bad:
    print("\n!! SHARD PROBLEMS:")
    for s, w in bad: print("   ", s, w)
if wrong:
    print("\n!! LAYERS WITH UNEXPECTED K:")
    for L, h, want in wrong: print("   layer %d: got %s want K=%d" % (L, h, want))
if not bad and not wrong:
    print("\nAUDIT PASSED: K assignment matches FRANKENSTEIN.json, all shards intact")
