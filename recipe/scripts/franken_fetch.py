"""Byte-range fetch of selected expert tensors from the 4.05bpw branch.

Avoids pulling all 153.7 GiB: safetensors stores an 8-byte header length, then a JSON header with
per-tensor {dtype, shape, data_offsets:[start,end]} relative to the end of the header. So we can
compute absolute byte ranges for exactly the tensors we want and issue ranged GETs (HF serves
accept-ranges: bytes), then write a fresh valid safetensors file containing just those tensors.

Usage: franken_fetch.py <out.safetensors> <layer> [layer ...]
"""
import json, os, sys, struct, urllib.request, collections, time

TOK = open("/root/.hf_token").read().strip()
REPO = "turboderp/GLM-5.3-Flash-exl3"
REV = "4.05bpw"
CHUNK = 32 << 20

def req(url, rng=None):
    h = {"Authorization": f"Bearer {TOK}"}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    return urllib.request.Request(url, headers=h)

def resolve(path):
    return f"https://huggingface.co/{REPO}/resolve/{REV}/{path}"

def get_json(path):
    return json.load(urllib.request.urlopen(req(resolve(path)), timeout=90))

def read_header(shard):
    url = resolve(shard)
    n = struct.unpack("<Q", urllib.request.urlopen(req(url, (0, 7)), timeout=90).read(8))[0]
    raw = urllib.request.urlopen(req(url, (8, 8 + n - 1)), timeout=180).read()
    return json.loads(raw), 8 + n

def main():
    out = sys.argv[1]
    layers = [int(x) for x in sys.argv[2:]]
    assert layers, "give at least one layer index"
    print(f"fetching expert tensors for layers {layers} from {REPO}@{REV}", flush=True)

    wm = get_json("model.safetensors.index.json")["weight_map"]
    want = collections.defaultdict(list)
    for k, shard in wm.items():
        if ".mlp.experts." not in k:
            continue
        parts = k.split(".")
        try:
            L = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        if L in layers:
            want[shard].append(k)
    print(f"  {sum(len(v) for v in want.values())} tensors across {len(want)} shard(s)", flush=True)

    tensors = {}          # key -> (dtype, shape, bytes)
    total = 0
    t0 = time.time()
    for shard in sorted(want):
        hdr, data_start = read_header(shard)
        keys = want[shard]
        # merge into contiguous spans to minimise requests
        spans = sorted((hdr[k]["data_offsets"][0], hdr[k]["data_offsets"][1], k) for k in keys)
        merged = []
        for s, e, k in spans:
            if merged and s - merged[-1][1] <= CHUNK:
                merged[-1][1] = max(merged[-1][1], e)
                merged[-1][2].append(k)
            else:
                merged.append([s, e, [k]])
        print(f"  {shard}: {len(keys)} tensors in {len(merged)} span(s)", flush=True)
        url = resolve(shard)
        for s, e, ks in merged:
            buf = bytearray()
            a, b = data_start + s, data_start + e - 1
            pos = a
            while pos <= b:
                hi = min(pos + CHUNK - 1, b)
                buf += urllib.request.urlopen(req(url, (pos, hi)), timeout=300).read()
                pos = hi + 1
            for k in ks:
                ks_s, ks_e = hdr[k]["data_offsets"]
                tensors[k] = (hdr[k]["dtype"], hdr[k]["shape"],
                              bytes(buf[ks_s - s: ks_e - s]))
                total += ks_e - ks_s
            print(f"    span {e-s:>12,} B  running total {total/2**30:6.2f} GiB"
                  f"  {total/2**20/max(1e-9, time.time()-t0):7.1f} MiB/s", flush=True)

    # write a valid safetensors file
    header, off = {}, 0
    for k in sorted(tensors):
        dt, shape, blob = tensors[k]
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    hj = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(hj)) % 8
    hj += b" " * pad
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for k in sorted(tensors):
            f.write(tensors[k][2])
    print(f"\nwrote {out}: {len(tensors)} tensors, {os.path.getsize(out)/2**30:.2f} GiB "
          f"in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
