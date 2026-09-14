"""Build the frankenstein pack: hardlink untouched shards, stream-rewrite the affected ones.

Key correctness points:
  - a rewritten shard contains SEVERAL layers (e.g. 00003 holds 8,9,10,11). Only the override
    layer's tensors are substituted; every other tensor is copied through byte-for-byte.
  - shard FILENAMES are preserved, so model.safetensors.index.json stays valid untouched and we
    never rely on the unsorted-glob sibling-override path (which is order-dependent).
  - streamed with seek+read in chunks, so an 8 GiB shard never lands in RAM.

Usage: franken_build.py <override.safetensors> <dest_dir>
"""
import json, os, shutil, struct, sys, time

SRC = "/root/workspace/GLM-5.3-Flash-exl3-3.05bpw"
CHUNK = 64 << 20

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def main():
    ovr_path, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    ovr_hdr, ovr_data = read_header(ovr_path)
    ovr_keys = {k for k in ovr_hdr if k not in ("__metadata__",)}
    print("override tensors: %d" % len(ovr_keys), flush=True)

    wm = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    affected = sorted({wm[k] for k in ovr_keys if k in wm})
    print("shards to rewrite: %s" % affected, flush=True)

    # 1) hardlink everything not being rewritten
    linked = 0
    for f in sorted(os.listdir(SRC)):
        if f.startswith(".") and f.endswith(".kate-swp"):
            continue                                  # stray editor swap file
        s, d = os.path.join(SRC, f), os.path.join(dst, f)
        if not os.path.isfile(s) or f in affected:
            continue
        if os.path.exists(d):
            os.remove(d)
        try:
            os.link(s, d)
            linked += 1
        except OSError:
            shutil.copy2(s, d)
            linked += 1
    print("hardlinked/copied %d files (shards + aux, incl. mtp.safetensors)" % linked, flush=True)

    # 2) rewrite the affected shards, substituting only override keys
    for sh in affected:
        t0 = time.time()
        sp = os.path.join(SRC, sh)
        dp = os.path.join(dst, sh)
        hdr, data_start = read_header(sp)
        order = [k for k in hdr if k not in ("__metadata__",)]
        order.sort(key=lambda k: hdr[k]["data_offsets"][0])
        sub = [k for k in order if k in ovr_keys]
        print("  %s: %d tensors, substituting %d" % (sh, len(order), len(sub)), flush=True)

        new_hdr, off = {}, 0
        for k in order:
            src_h = ovr_hdr[k] if k in ovr_keys else hdr[k]
            a, b = src_h["data_offsets"]
            n = b - a
            new_hdr[k] = {"dtype": src_h["dtype"], "shape": src_h["shape"],
                          "data_offsets": [off, off + n]}
            off += n
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        hj = json.dumps(new_hdr, separators=(",", ":")).encode()
        hj += b" " * ((-len(hj)) % 8)

        with open(dp, "wb") as out, open(sp, "rb") as fsrc, open(ovr_path, "rb") as fovr:
            out.write(struct.pack("<Q", len(hj)))
            out.write(hj)
            for k in order:
                if k in ovr_keys:
                    fh, base, h = fovr, ovr_data, ovr_hdr[k]
                else:
                    fh, base, h = fsrc, data_start, hdr[k]
                a, b = h["data_offsets"]
                fh.seek(base + a)
                left = b - a
                while left > 0:
                    buf = fh.read(min(CHUNK, left))
                    if not buf:
                        raise IOError("short read on %s" % k)
                    out.write(buf)
                    left -= len(buf)
        print("    wrote %.2f GiB in %.0fs" % (os.path.getsize(dp) / 2**30, time.time() - t0), flush=True)

    # 3) annotate provenance so the pack is not mistaken for a clean 3.05
    note = {
        "frankenstein": True,
        "base_pack": SRC,
        "override_file": ovr_path,
        "override_tensor_count": len(ovr_keys),
        "rewritten_shards": affected,
        "note": "expert tensors for the override layers are K=4 from the 4.05bpw branch; "
                "quantization_config.json still reports bits=3.05 and stale tensor_storage "
                "(both are write-only at load time)",
    }
    with open(os.path.join(dst, "FRANKENSTEIN.json"), "w") as f:
        json.dump(note, f, indent=2)
    print("\nbuilt %s" % dst, flush=True)
    tot = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)
              if os.path.isfile(os.path.join(dst, f)))
    print("  apparent size %.2f GiB (hardlinks share blocks with the original)" % (tot / 2**30))

if __name__ == "__main__":
    main()
