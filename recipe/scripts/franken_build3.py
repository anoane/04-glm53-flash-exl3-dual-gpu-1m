"""Build the UNCENSORED frankenstein: K=4 expert layers from the uncensored 4.05 pack
swapped into the uncensored 3.05 base.

Same proven engine as franken_build2.py (index-driven, seek+read streaming, shard FILENAMES
preserved so model.safetensors.index.json stays valid). Changes for the 2026-09-13 rebuild:
  - SRC/DONOR repointed at the MikeRoz uncensored packs (the old censored dirs were emptied).
  - Layout-agnostic: uncensored 3.05 = 15 shards, uncensored 4.05 = 23 (old censored 4.05 was 19).
    Nothing keys off shard counts; everything comes from each pack's own weight_map.
  - Provenance string corrected to the uncensored source.
  - GUARDS added: key-set equality, no in-place clobber, and a check that no stray
    mtp.safetensors is present (these packs carry the MTP head IN-INDEX as layer 45, so a
    sibling file would load a SECOND, mismatched head).

Usage: franken_build3.py <dest_dir> <layer> [layer ...]
  e.g. franken_build3.py /root/workspace/GLM-5.3-Flash-Uncensored-franken-13L 3 4 5 6 7 8 9 10 11 12 13 14 15
"""
import json, os, shutil, struct, sys, time

SRC   = os.environ.get("FRANKEN_SRC",   "/root/workspace/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3")
DONOR = os.environ.get("FRANKEN_DONOR", "/root/workspace/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3")
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

    # --- guards -----------------------------------------------------------
    assert os.path.realpath(dst) not in (os.path.realpath(SRC), os.path.realpath(DONOR)), \
        "refusing to build in place over SRC/DONOR"
    if os.path.isdir(dst) and any(f.endswith(".safetensors") for f in os.listdir(dst)):
        sys.exit("!! %s already holds safetensors; refusing to clobber (this script relinks "
                 "every non-rewritten file from SRC and would wipe prior upgrades)" % dst)
    for pack in (SRC, DONOR):
        stray = [f for f in os.listdir(pack) if f.endswith(".safetensors") and "model-" not in f]
        if stray:
            sys.exit("!! stray non-shard safetensors in %s: %s -- the loader globs *.safetensors "
                     "and would load them as extra weights" % (pack, stray))

    wm_new = json.load(open(os.path.join(DONOR, "model.safetensors.index.json")))["weight_map"]
    wm_src = json.load(open(os.path.join(SRC,   "model.safetensors.index.json")))["weight_map"]
    if set(wm_new) != set(wm_src):
        sys.exit("!! key sets differ between SRC and DONOR -- not the same checkpoint")
    print("key sets identical: %d keys" % len(wm_src), flush=True)

    os.makedirs(dst, exist_ok=True)

    ovr = {k: sh for k, sh in wm_new.items()
           if ".mlp.experts." in k and layer_of(k) in layers}
    have = {sh for sh in set(ovr.values()) if os.path.isfile(os.path.join(DONOR, sh))}
    missing = sorted(set(ovr.values()) - have)
    if missing:
        sys.exit("!! required donor shards not present: %s" % missing)
    print("override keys: %d across donor shards %s" % (len(ovr), sorted(have)), flush=True)

    nh = {}
    for sh in sorted(have):
        nh[sh] = read_header(os.path.join(DONOR, sh))
        print("  header read: %s" % sh, flush=True)

    affected = sorted({wm_src[k] for k in ovr if k in wm_src})
    print("base shards to rewrite: %s" % affected, flush=True)

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
    print("hardlinked %d unchanged files (index, tokenizer, config, untouched shards)" % linked, flush=True)

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
            h = nh[ovr[k]][0][k] if k in ovr else hdr[k]
            a, b = h["data_offsets"]
            new_hdr[k] = {"dtype": h["dtype"], "shape": h["shape"],
                          "data_offsets": [off, off + (b - a)]}
            off += b - a
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        hj = json.dumps(new_hdr, separators=(",", ":")).encode()
        hj += b" " * ((-len(hj)) % 8)

        fhs = {s2: open(os.path.join(DONOR, s2), "rb") for s2 in sorted(have)}
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
        json.dump({"frankenstein": True,
                   "base_pack": SRC,
                   "donor_pack": DONOR,
                   "upgraded_layers": sorted(layers),
                   "override_tensors": len(ovr),
                   "rewritten_shards": affected,
                   "source": "MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3 expert tensors (K=4)",
                   "base": "MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3 (K=3)",
                   "note": "UNCENSORED rebuild 2026-09-13. MTP head is IN-INDEX at layer 45 (no "
                           "sibling mtp.safetensors). quantization_config.json still reports "
                           "bits=3.05 and stale tensor_storage; both are write-only at load time."},
                  f, indent=2)
    print("\nbuilt %s" % dst, flush=True)

if __name__ == "__main__":
    main()
