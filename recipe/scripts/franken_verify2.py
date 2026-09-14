"""Verify 4.05 override tensors against their 3.05 counterparts, reading from downloaded shards.

A silent shape/dtype mismatch would poison the whole experiment, so check before building:
  - key exists in both packs
  - dtype identical
  - trellis: source K==3, override K==4, all non-K dims identical, logical weight count equal
  - suh/svh/mul1: shape and dtype identical (they do not scale with K)
Usage: franken_verify2.py <layer> [layer ...]
"""
import json, os, sys, struct

SRC = "/root/workspace/GLM-5.3-Flash-exl3-3.05bpw"
NEW = "/root/workspace/franken_405_shards"

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def layer_of(key):
    p = key.split(".")
    try:
        return int(p[p.index("layers") + 1])
    except (ValueError, IndexError):
        return None

def numel(shape):
    n = 1
    for x in shape:
        n *= x
    return n

def main():
    layers = {int(x) for x in sys.argv[1:]} or {9, 18}
    wm_new = json.load(open(os.path.join(NEW, "model.safetensors.index.json")))["weight_map"]
    wm_src = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]

    keys = [k for k, sh in wm_new.items()
            if ".mlp.experts." in k and layer_of(k) in layers]
    print("checking %d override keys for layers %s" % (len(keys), sorted(layers)))

    hdr_new, hdr_src = {}, {}
    for sh in sorted({wm_new[k] for k in keys}):
        p = os.path.join(NEW, sh)
        if not os.path.isfile(p):
            print("!! missing downloaded shard %s" % sh); sys.exit(1)
        hdr_new[sh] = read_header(p)
    for sh in sorted({wm_src[k] for k in keys if k in wm_src}):
        hdr_src[sh] = read_header(os.path.join(SRC, sh))

    bad, kd, ok = [], {}, 0
    for k in keys:
        if k not in wm_src:
            bad.append((k, "not in 3.05 pack")); continue
        hn = hdr_new[wm_new[k]].get(k)
        hs = hdr_src[wm_src[k]].get(k)
        if hn is None or hs is None:
            bad.append((k, "header entry missing")); continue
        if hn["dtype"] != hs["dtype"]:
            bad.append((k, "dtype %s vs %s" % (hn["dtype"], hs["dtype"]))); continue
        sn, ss = hn["shape"], hs["shape"]
        if k.endswith(".trellis"):
            Kn, Ks = sn[-1] // 16, ss[-1] // 16
            kd[(Ks, Kn)] = kd.get((Ks, Kn), 0) + 1
            if Kn != 4 or Ks != 3:
                bad.append((k, "K %d -> %d (want 3 -> 4)" % (Ks, Kn))); continue
            if sn[:-1] != ss[:-1]:
                bad.append((k, "non-K dims %s vs %s" % (sn, ss))); continue
            if numel(sn) * 16 // Kn != numel(ss) * 16 // Ks:
                bad.append((k, "logical weights differ")); continue
            ok += 1
        else:
            if sn != ss:
                bad.append((k, "shape %s vs %s" % (sn, ss)))

    print("K transition histogram (src -> ovr): %s" % kd)
    print("trellis with matching logical weights: %d" % ok)
    if bad:
        print("\n!! %d PROBLEMS (first 10):" % len(bad))
        for k, why in bad[:10]:
            print("   %s : %s" % (k, why))
        sys.exit(1)
    print("\nALL CHECKS PASSED - safe to build")

if __name__ == "__main__":
    main()
