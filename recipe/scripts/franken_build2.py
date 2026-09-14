"""Build the frankenstein pack, reading override tensors DIRECTLY from downloaded 4.05 shards.

Avoids materialising a 6.75 GiB intermediate override file: we already have the 4 whole shards
that contain the probe layers, so map key -> (shard, offsets) from the 4.05 index and pull bytes
straight out of them.

Correctness points carried over:
  - a rewritten 3.05 shard holds SEVERAL layers (00003 = 8,9,10,11; 00006 = 17,18,19,20); only the
    override layers' expert tensors are substituted, everything else is copied byte-for-byte.
  - shard FILENAMES are preserved, so model.safetensors.index.json stays valid and we never depend
    on the unsorted-glob sibling-override path.
  - streamed with seek+read, so an 8 GiB shard never lands in RAM.

Usage: franken_build2.py <dest_dir> <layer> [layer ...]
"""
import json, os, shutil, struct, sys, time

SRC = "/root/workspace/GLM-5.3-Flash-exl3-3.05bpw"
NEW = "/root/workspace/franken_405_shards"
CHUNK = 64 << 20

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def layer_of(key):
    p = key.split(".")
    try:
        return int(p[p.index("layers") + 1])
    except (ValueError, IndexError):
        return None

def main():
    dst = sys.argv[1]
    layers = {int(x) for x in sys.argv[2:]}
    assert layers, "give at least one layer"
    os.makedirs(dst, exist_ok=True)

    wm_new = json.load(open(os.path.join(NEW, "model.safetensors.index.json")))["weight_map"]
    wm_src = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]

    # which keys are we overriding, and where do they live in the downloaded shards?
    ovr = {}
    for k, sh in wm_new.items():
        if ".mlp.experts." not in k:
            continue
        if layer_of(k) in layers:
            ovr[k] = sh
    have = {sh for sh in set(ovr.values())
            if os.path.isfile(os.path.join(NEW, sh))}
    missing = sorted(set(ovr.values()) - have)
    if missing:
        print("!! required 4.05 shards not downloaded: %s" % missing)
        sys.exit(1)
    print("override keys: %d across shards %s" % (len(ovr), sorted(have)), flush=True)

    # cache headers of the downloaded shards
    nh = {}
    for sh in sorted(have):
        nh[sh] = read_header(os.path.join(NEW, sh))
        print("  header read: %s" % sh, flush=True)

    affected = sorted({wm_src[k] for k in ovr if k in wm_src})
    print("3.05 shards to rewrite: %s" % affected, flush=True)

    linked = 0
    for f in sorted(os.listdir(SRC)):
        if f.endswith(".kate-swp"):
            continue
        s, d = os.path.join(SRC, f), os.path.join(dst, f)
        if not os.path.isfile(s) or f in affected:
            continue
        if os.path.exists(d):
            os.remove(d)
        try:
            os.link(s, d); linked += 1
        except OSError:
            shutil.copy2(s, d); linked += 1
    print("hardlinked %d unchanged files (incl. mtp.safetensors, index, tokenizer)" % linked, flush=True)

    for sh in affected:
        t0 = time.time()
        sp, dp = os.path.join(SRC, sh), os.path.join(dst, sh)
        hdr, dstart = read_header(sp)
        order = sorted((k for k in hdr if k != "__metadata__"),
                       key=lambda k: hdr[k]["data_offsets"][0])
        subs = [k for k in order if k in ovr]
        print("  %s: %d tensors, substituting %d" % (sh, len(order), len(subs)), flush=True)

        new_hdr, off = {}, 0
        for k in order:
            if k in ovr:
                h = nh[ovr[k]][0][k]
            else:
                h = hdr[k]
            a, b = h["data_offsets"]
            new_hdr[k] = {"dtype": h["dtype"], "shape": h["shape"],
                          "data_offsets": [off, off + (b - a)]}
            off += b - a
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        hj = json.dumps(new_hdr, separators=(",", ":")).encode()
        hj += b" " * ((-len(hj)) % 8)

        fhs = {s2: open(os.path.join(NEW, s2), "rb") for s2 in sorted(have)}
        try:
            with open(dp, "wb") as out, open(sp, "rb") as fsrc:
                out.write(struct.pack("<Q", len(hj)))
                out.write(hj)
                for k in order:
                    if k in ovr:
                        s2 = ovr[k]
                        h, base, fh = nh[s2][0][k], nh[s2][1], fhs[s2]
                    else:
                        h, base, fh = hdr[k], dstart, fsrc
                    a, b = h["data_offsets"]
                    fh.seek(base + a)
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
        print("    wrote %.2f GiB in %.0fs" % (os.path.getsize(dp) / 2**30, time.time() - t0), flush=True)

    with open(os.path.join(dst, "FRANKENSTEIN.json"), "w") as f:
        json.dump({"frankenstein": True, "base_pack": SRC, "upgraded_layers": sorted(layers),
                   "override_tensors": len(ovr), "rewritten_shards": affected,
                   "source": "turboderp/GLM-5.3-Flash-exl3@4.05bpw expert tensors (K=4)",
                   "note": "quantization_config.json still reports bits=3.05 and stale "
                           "tensor_storage; both are write-only at load time"}, f, indent=2)
    print("\nbuilt %s" % dst, flush=True)

if __name__ == "__main__":
    main()
