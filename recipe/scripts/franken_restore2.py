"""In-place RESTORE of layers K=3 -> K=4, pulling experts from the 4.05 donor.

Mirror of franken_revert2.py (same temp-file + os.replace safety, same per-tensor
donor-by-name lookup). Needed because franken_restore.py hardcodes the CENSORED packs,
which were emptied on 2026-09-13.

WHY THIS EXISTS: the 13-layer pack was reverted to 12 on a WRONG diagnosis -- all 11 of its
load attempts died in the DRAFT phase (draft_gpu_split [0,5] too small for the 3.533 GiB
in-index MTP head), never reaching the trunk, so its fit was never actually tested.

Usage: franken_restore2.py <layer> [layer ...]
Env:   FRANKEN_DST (pack to edit), FRANKEN_DONOR (K=4 source)
"""
import json, os, struct, sys, time

DST   = os.environ.get("FRANKEN_DST",   "/root/workspace/GLM-5.3-Flash-Uncensored-franken-13L")
DONOR = os.environ.get("FRANKEN_DONOR", "/root/workspace/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3")
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
    assert layers, "give at least one layer to restore"

    if not any(f.endswith(".safetensors") for f in os.listdir(DONOR)):
        sys.exit("!! donor %s has no safetensors" % DONOR)

    meta_p = os.path.join(DST, "FRANKENSTEIN.json")
    meta = json.load(open(meta_p))
    up = set(meta.get("upgraded_layers") or [])
    if layers & up:
        sys.exit("!! already upgraded: %s" % sorted(layers & up))

    wm = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"]
    donor_wm = json.load(open(os.path.join(DONOR, "model.safetensors.index.json")))["weight_map"]
    if set(wm) != set(donor_wm):
        sys.exit("!! key sets differ between pack and donor")

    ovr = {k: sh for k, sh in wm.items()
           if ".mlp.experts." in k and layer_of(k) in layers}
    affected = sorted({ovr[k] for k in ovr})
    print("restoring layers %s : %d tensors across shards %s"
          % (sorted(layers), len(ovr), affected), flush=True)

    dh = {}
    for sh in sorted({donor_wm[k] for k in ovr}):
        dh[sh] = read_header(os.path.join(DONOR, sh))
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
            h = dh[donor_wm[k]][0][k] if k in ovr else hdr[k]
            a, b = h["data_offsets"]
            new_hdr[k] = {"dtype": h["dtype"], "shape": h["shape"],
                          "data_offsets": [off, off + (b - a)]}
            off += b - a
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        hj = json.dumps(new_hdr, separators=(",", ":")).encode()
        hj += b" " * ((-len(hj)) % 8)

        fhs = {s2: open(os.path.join(DONOR, s2), "rb") for s2 in dh}
        try:
            with open(tmp, "wb") as out, open(sp, "rb") as fsrc:
                out.write(struct.pack("<Q", len(hj)))
                out.write(hj)
                for k in order:
                    if k in ovr:
                        s2 = donor_wm[k]
                        h, base_off, fh = dh[s2][0][k], dh[s2][1], fhs[s2]
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
        print("  %s: %d restored, %.2f -> %.2f GiB (%.0fs)"
              % (sh, len(subs), before / 2**30, os.path.getsize(sp) / 2**30, time.time() - t0),
              flush=True)

    meta["upgraded_layers"] = sorted(up | layers)
    meta["reverted_to_K3"] = sorted(set(meta.get("reverted_to_K3") or []) - layers)
    json.dump(meta, open(meta_p, "w"), indent=2)
    print("\nnow upgraded: %s" % meta["upgraded_layers"])
    print("reverted    : %s" % meta["reverted_to_K3"])

if __name__ == "__main__":
    main()
