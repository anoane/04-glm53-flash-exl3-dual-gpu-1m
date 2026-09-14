"""Revert chosen layers in the 20L pack from K=4 back to K=3, taking tensors from the 3.05 pack.

No download needed: the K=3 expert tensors are already in GLM-5.3-Flash-exl3-3.05bpw.
Each reverted layer frees 0.8438 GiB of weights, which reduces the spill onto cuda:1 and is
what lets 1M + MTP fit (the last attempt was one module short at 49/50).

HARDLINK SAFETY: some 20L shards share inodes with the 3.05 pack. Every rebuilt shard is written
to a temp file and os.replace()d, so the 3.05 baseline is never written through.

Usage: franken_revert.py <layer> [layer ...]
"""
import json, os, struct, sys, time

SRC3 = "/root/workspace/GLM-5.3-Flash-exl3-3.05bpw"      # K=3 donor
DST  = "/root/workspace/GLM-5.3-Flash-franken-20L"        # pack being edited
CHUNK = 64 << 20

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def layer_of(k):
    p = k.split(".")
    try:
        return int(p[p.index("layers") + 1])
    except (ValueError, IndexError):
        return None

def main():
    layers = {int(x) for x in sys.argv[1:]}
    assert layers, "give at least one layer"
    wm3 = json.load(open(os.path.join(SRC3, "model.safetensors.index.json")))["weight_map"]
    wmd = json.load(open(os.path.join(DST,  "model.safetensors.index.json")))["weight_map"]

    revert = {k for k in wmd if ".mlp.experts." in k and layer_of(k) in layers}
    assert revert, "no expert tensors matched"
    affected = sorted({wmd[k] for k in revert})
    print("reverting layers %s : %d tensors across shards %s"
          % (sorted(layers), len(revert), affected), flush=True)

    # donor headers
    dh = {}
    for sh in sorted({wm3[k] for k in revert if k in wm3}):
        dh[sh] = read_header(os.path.join(SRC3, sh))

    for sh in affected:
        t0 = time.time()
        dp = os.path.join(DST, sh)
        hdr, dstart = read_header(dp)
        order = sorted((k for k in hdr if k != "__metadata__"),
                       key=lambda k: hdr[k]["data_offsets"][0])
        subs = [k for k in order if k in revert]
        pre = os.path.getsize(dp)

        new_hdr, off = {}, 0
        for k in order:
            h = dh[wm3[k]][0][k] if k in revert else hdr[k]
            a, b = h["data_offsets"]
            new_hdr[k] = {"dtype": h["dtype"], "shape": h["shape"],
                          "data_offsets": [off, off + (b - a)]}
            off += b - a
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        hj = json.dumps(new_hdr, separators=(",", ":")).encode()
        hj += b" " * ((-len(hj)) % 8)

        tmp = dp + ".rebuild"
        fh3 = {s: open(os.path.join(SRC3, s), "rb") for s in dh}
        try:
            with open(tmp, "wb") as out, open(dp, "rb") as fdst:
                out.write(struct.pack("<Q", len(hj)))
                out.write(hj)
                for k in order:
                    if k in revert:
                        s3 = wm3[k]
                        h, base, fh = dh[s3][0][k], dh[s3][1], fh3[s3]
                    else:
                        h, base, fh = hdr[k], dstart, fdst
                    a, b = h["data_offsets"]
                    fh.seek(base + a)
                    left = b - a
                    while left > 0:
                        buf = fh.read(min(CHUNK, left))
                        if not buf:
                            raise IOError("short read on " + k)
                        out.write(buf)
                        left -= len(buf)
        finally:
            for f in fh3.values():
                f.close()
        os.replace(tmp, dp)          # breaks any hardlink; 3.05 untouched
        post = os.path.getsize(dp)
        print("  %s: %d reverted, %.2f -> %.2f GiB (%.0fs)"
              % (sh, len(subs), pre / 2**30, post / 2**30, time.time() - t0), flush=True)

    note = os.path.join(DST, "FRANKENSTEIN.json")
    j = json.load(open(note))
    j["upgraded_layers"] = [L for L in j.get("upgraded_layers", []) if L not in layers]
    j["reverted_to_K3"] = sorted(set(j.get("reverted_to_K3", [])) | layers)
    j["note"] = ("layers in upgraded_layers are K=4 from the 4.05bpw branch; reverted_to_K3 were "
                 "rolled back to the 3.05 donor to make 1M + MTP fit")
    json.dump(j, open(note, "w"), indent=2)
    print("\nnow upgraded: %s" % j["upgraded_layers"])
    print("reverted    : %s" % j["reverted_to_K3"])

if __name__ == "__main__":
    main()
