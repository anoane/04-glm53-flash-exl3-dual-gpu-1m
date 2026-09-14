"""Forward-restore layers to K=4 from the 4.05 shards, editing the pack IN PLACE.

DO NOT use franken_build2.py for this. Its lines 73-80 do os.remove()+os.link() from the 3.05
source for every file NOT being rewritten -- which, against an existing franken pack, would wipe
the already-upgraded layers 3..14 and relink them to 3.05 while reporting success.

This mirrors franken_revert.py's proven shape instead:
  * only shards containing the target layers are touched
  * per-tensor donor selection (4.05 for target-layer experts, current pack for everything else)
  * temp file + os.replace, so hardlinks are broken safely and no other pack is written through

Usage: franken_restore.py <layer> [layer ...]
"""
import json, os, struct, sys, time

NEW  = "/root/workspace/franken_405_shards"          # K=4 donor
DST  = "/root/workspace/GLM-5.3-Flash-franken-20L"    # pack edited in place
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

    wm_new = json.load(open(os.path.join(NEW, "model.safetensors.index.json")))["weight_map"]
    wm_dst = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"]

    ovr = {k: wm_new[k] for k in wm_dst
           if ".mlp.experts." in k and layer_of(k) in layers and k in wm_new}
    assert ovr, "no expert tensors matched"

    missing = sorted({s for s in ovr.values() if not os.path.isfile(os.path.join(NEW, s))})
    if missing:
        print("!! donor shards absent: %s" % missing); sys.exit(1)

    affected = sorted({wm_dst[k] for k in ovr})
    print("restoring layers %s to K=4" % sorted(layers))
    print("  %d tensors, donor shards %s" % (len(ovr), sorted(set(ovr.values()))))
    print("  BLAST RADIUS -- pack shards rewritten: %s" % affected, flush=True)

    nh = {s: read_header(os.path.join(NEW, s)) for s in sorted(set(ovr.values()))}

    for sh in affected:
        t0 = time.time()
        dp = os.path.join(DST, sh)
        hdr, dstart = read_header(dp)
        order = sorted((k for k in hdr if k != "__metadata__"),
                       key=lambda k: hdr[k]["data_offsets"][0])
        subs = [k for k in order if k in ovr]
        pre = os.path.getsize(dp)

        new_hdr, off = {}, 0
        for k in order:
            h = nh[ovr[k]][0][k] if k in ovr else hdr[k]
            a, b = h["data_offsets"]
            new_hdr[k] = {"dtype": h["dtype"], "shape": h["shape"],
                          "data_offsets": [off, off + (b - a)]}
            off += b - a
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        hj = json.dumps(new_hdr, separators=(",", ":")).encode()
        hj += b" " * ((-len(hj)) % 8)

        tmp = dp + ".rebuild"
        fhs = {s: open(os.path.join(NEW, s), "rb") for s in nh}
        try:
            with open(tmp, "wb") as out, open(dp, "rb") as fdst:
                out.write(struct.pack("<Q", len(hj)))
                out.write(hj)
                for k in order:
                    if k in ovr:
                        s2 = ovr[k]
                        h, base, fh = nh[s2][0][k], nh[s2][1], fhs[s2]
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
            for f in fhs.values():
                f.close()
        os.replace(tmp, dp)
        print("  %s: %d restored, %.2f -> %.2f GiB (%.0fs)"
              % (sh, len(subs), pre / 2**30, os.path.getsize(dp) / 2**30, time.time() - t0), flush=True)

    note = os.path.join(DST, "FRANKENSTEIN.json")
    j = json.load(open(note))
    j["upgraded_layers"] = sorted(set(j.get("upgraded_layers", [])) | layers)
    j["reverted_to_K3"] = [L for L in j.get("reverted_to_K3", []) if L not in layers]
    up = len(j["upgraded_layers"])
    j["effective_expert_bpw"] = round((up * 4 + (42 - up) * 3) / 42, 3)
    json.dump(j, open(note, "w"), indent=2)
    print("\n  upgraded now: %s" % j["upgraded_layers"])
    print("  reverted    : %s" % j["reverted_to_K3"])
    print("  effective   : %.3f bpw" % j["effective_expert_bpw"])

if __name__ == "__main__":
    main()
