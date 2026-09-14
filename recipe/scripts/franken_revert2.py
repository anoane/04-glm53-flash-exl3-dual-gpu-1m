"""In-place revert of upgraded layers K=4 -> K=3, for the UNCENSORED pack.

The original franken_revert.py hardcoded the CENSORED donor
(/root/workspace/GLM-5.3-Flash-exl3-3.05bpw, now emptied) and the old DST, so it cannot be
used. This is a fresh, self-contained implementation.

In-place because a clean rebuild into a new directory needs ~128 GiB and only ~121 GiB is free.
Only the shards containing the target layers' expert tensors are rewritten; each is written to
a temp file in the same directory and then os.replace()d, so a crash never leaves a torn shard.
Writing a temp + replace also cannot damage a hardlinked (links=2) shard's other holder --
the base pack's inode is left untouched.

Per-tensor donor lookup: the 3.05 base is itself mixed-K (experts K=3, attention K=5, etc.),
so every tensor is taken by NAME from the base, never assumed uniform.

Usage: franken_revert2.py <layer> [layer ...]
Env:   FRANKEN_DST (pack to edit), FRANKEN_BASE (K=3 donor)
"""
import json, os, struct, sys, time

DST  = os.environ.get("FRANKEN_DST",  "/root/workspace/GLM-5.3-Flash-Uncensored-franken-13L")
BASE = os.environ.get("FRANKEN_BASE", "/root/workspace/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3")
CHUNK = 64 << 20

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def layer_of(key):
    p = key.split(".")
    try:
        return int(p[p.index("layers") + 1])
    except (ValueError, IndexError):
        return None

def main():
    layers = {int(x) for x in sys.argv[1:]}
    assert layers, "give at least one layer to revert"

    n_base = len([f for f in os.listdir(BASE) if f.endswith(".safetensors")])
    if n_base == 0:
        sys.exit("!! donor %s has no safetensors" % BASE)
    meta_p = os.path.join(DST, "FRANKENSTEIN.json")
    meta = json.load(open(meta_p))
    up = set(meta.get("upgraded_layers") or [])
    if not layers <= up:
        sys.exit("!! not currently upgraded: %s (upgraded=%s)" % (sorted(layers - up), sorted(up)))

    wm = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"]
    ovr = {k: sh for k, sh in wm.items()
           if ".mlp.experts." in k and layer_of(k) in layers}
    affected = sorted({ovr[k] for k in ovr})
    print("reverting layers %s : %d tensors across shards %s"
          % (sorted(layers), len(ovr), affected), flush=True)

    bh = {}
    for sh in affected:
        # donor tensors may live in DIFFERENT shard names in the base pack
        pass
    base_wm = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))["weight_map"]
    need_base = sorted({base_wm[k] for k in ovr})
    for sh in need_base:
        bh[sh] = read_header(os.path.join(BASE, sh))
        print("  donor header: %s" % sh, flush=True)

    for sh in affected:
        t0 = time.time()
        sp = os.path.join(DST, sh)
        tmp = sp + ".tmp"
        hdr, dstart = read_header(sp)
        order = sorted((k for k in hdr if k != "__metadata__"),
                       key=lambda k: hdr[k]["data_offsets"][0])
        subs = [k for k in order if k in ovr]
        before = os.path.getsize(sp)

        new_hdr, off = {}, 0
        for k in order:
            h = bh[base_wm[k]][0][k] if k in ovr else hdr[k]
            a, b = h["data_offsets"]
            new_hdr[k] = {"dtype": h["dtype"], "shape": h["shape"],
                          "data_offsets": [off, off + (b - a)]}
            off += b - a
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        hj = json.dumps(new_hdr, separators=(",", ":")).encode()
        hj += b" " * ((-len(hj)) % 8)

        fhs = {s2: open(os.path.join(BASE, s2), "rb") for s2 in need_base}
        try:
            with open(tmp, "wb") as out, open(sp, "rb") as fsrc:
                out.write(struct.pack("<Q", len(hj)))
                out.write(hj)
                for k in order:
                    if k in ovr:
                        s2 = base_wm[k]
                        h, base_off, fh = bh[s2][0][k], bh[s2][1], fhs[s2]
                    else:
                        h, base_off, fh = hdr[k], dstart, fsrc
                    a, b = h["data_offsets"]
                    fh.seek(base_off + a)
                    left = b - a
                    while left > 0:
                        buf = fh.read(min(CHUNK, left))
                        if not buf:
                            raise IOError("short read on %s" % k)
                        out.write(buf)
                        left -= len(buf)
        finally:
            for fh in fhs.values():
                fh.close()
        os.replace(tmp, sp)
        print("  %s: %d reverted, %.2f -> %.2f GiB (%.0fs)"
              % (sh, len(subs), before / 2**30, os.path.getsize(sp) / 2**30, time.time() - t0),
              flush=True)

    meta["upgraded_layers"] = sorted(up - layers)
    meta.setdefault("reverted_to_K3", [])
    meta["reverted_to_K3"] = sorted(set(meta["reverted_to_K3"]) | layers)
    json.dump(meta, open(meta_p, "w"), indent=2)
    print("\nnow upgraded: %s" % meta["upgraded_layers"])
    print("reverted    : %s" % meta["reverted_to_K3"])

if __name__ == "__main__":
    main()
