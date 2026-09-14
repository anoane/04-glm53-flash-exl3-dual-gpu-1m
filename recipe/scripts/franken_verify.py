"""Verify fetched 4.05 tensors BEFORE building: a silent mismatch would poison everything.

Checks, per fetched key:
  - it exists in the 3.05 pack (same key name)
  - dtype matches the 3.05 counterpart
  - trellis: K == 4 (shape[-1]//16) vs 3.05's K == 3, and all other dims IDENTICAL
  - suh/svh/mul1: shape and dtype IDENTICAL to 3.05 (these do not scale with K)
  - logical weight count matches: numel*16/K must be equal in both packs
"""
import json, os, sys
from safetensors import safe_open

SRC = "/root/workspace/GLM-5.3-Flash-exl3-3.05bpw"
OVR = sys.argv[1] if len(sys.argv) > 1 else "/root/workspace/franken_probe_L9_L18.safetensors"

wm = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]

src_meta = {}
need = {}
with safe_open(OVR, framework="pt") as f:
    ovr_keys = list(f.keys())
    for k in ovr_keys:
        sl = f.get_slice(k)
        need[k] = (tuple(sl.get_shape()), sl.get_dtype())

shards = {}
for k in need:
    sh = wm.get(k)
    if sh is None:
        continue
    shards.setdefault(sh, []).append(k)
for sh, ks in shards.items():
    with safe_open(os.path.join(SRC, sh), framework="pt") as f:
        for k in ks:
            sl = f.get_slice(k)
            src_meta[k] = (tuple(sl.get_shape()), sl.get_dtype())

print("fetched tensors: %d" % len(need))
print("resolved in 3.05 pack: %d" % len(src_meta))
missing = [k for k in need if k not in src_meta]
if missing:
    print("!! %d fetched keys NOT in the 3.05 pack, e.g. %s" % (len(missing), missing[:3]))

bad = []
kdist = {}
logical_ok = 0
for k, (sh_o, dt_o) in need.items():
    if k not in src_meta:
        continue
    sh_s, dt_s = src_meta[k]
    if dt_o != dt_s:
        bad.append((k, "dtype %s vs %s" % (dt_o, dt_s))); continue
    if k.endswith(".trellis"):
        Ko, Ks = sh_o[-1] // 16, sh_s[-1] // 16
        kdist[(Ks, Ko)] = kdist.get((Ks, Ko), 0) + 1
        if Ko != 4:
            bad.append((k, "fetched K=%d, expected 4" % Ko)); continue
        if Ks != 3:
            bad.append((k, "source K=%d, expected 3" % Ks)); continue
        if sh_o[:-1] != sh_s[:-1]:
            bad.append((k, "non-K dims differ %s vs %s" % (sh_o, sh_s))); continue
        no = 1
        for x in sh_o: no *= x
        ns = 1
        for x in sh_s: ns *= x
        if no * 16 // Ko != ns * 16 // Ks:
            bad.append((k, "logical weights differ")); continue
        logical_ok += 1
    else:
        if sh_o != sh_s:
            bad.append((k, "shape %s vs %s" % (sh_o, sh_s)))

print("\nK transition histogram (src_K -> fetched_K): %s" % kdist)
print("trellis tensors with matching logical weight count: %d" % logical_ok)
if bad:
    print("\n!! %d PROBLEMS (first 10):" % len(bad))
    for k, why in bad[:10]:
        print("   %s : %s" % (k, why))
    sys.exit(1)
print("\nALL CHECKS PASSED - safe to build")
