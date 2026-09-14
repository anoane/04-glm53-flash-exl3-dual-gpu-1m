"""Plan the cheap 2-layer probe: exact byte ranges to fetch for layers 3 and 44 from 4.05bpw.

Rationale: advance_state_parallel confirms exl3 calibrates each layer on activations that already
passed through the QUANTIZED upstream layers, so a 4.05 layer's error compensation assumes a 4.05
upstream. In a frankenstein pack its upstream is 3.05. That mismatch is second-order but unproven,
so measure it on 2 layers (~6.8 GiB) before committing ~67 GiB for 20 layers.
"""
import json, urllib.request, collections, re
TOK = open("/root/.hf_token").read().strip()
REPO = "turboderp/GLM-5.3-Flash-exl3"

def hf(path, rev):
    url = f"https://huggingface.co/{REPO}/resolve/{rev}/{path}"
    r = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOK}"})
    return json.load(urllib.request.urlopen(r, timeout=90))

wm = hf("model.safetensors.index.json", "4.05bpw")["weight_map"]
LAYERS = [3, 44]
sel = collections.defaultdict(list)
for k, v in wm.items():
    if ".mlp.experts." not in k:
        continue
    m = re.search(r"layers\.(\d+)\.", k)
    if m and int(m.group(1)) in LAYERS:
        sel[v].append(k)

print("=== probe fetch plan: layers %s from 4.05bpw ===" % LAYERS)
tot = 0
for shard in sorted(sel):
    print("  %s : %d tensors" % (shard, len(sel[shard])))
print("  total tensors: %d" % sum(len(v) for v in sel.values()))
EXPERT_W = 288 * 3 * 4096 * 2048
print("  expected bytes at K=4: %.2f GiB for %d layers" % (len(LAYERS) * EXPERT_W * 4 / 8 / 2**30, len(LAYERS)))
print()
print("  Contiguity matters for range fetching: safetensors lays tensors out in header order,")
print("  so one layer's 3456 expert tensors are usually one near-contiguous span per shard.")
print("  Fetch = read header (first 8 bytes = header len), find min/max offset for the selected")
print("  keys, then one ranged GET per shard span.")
print()
print("=== what the probe then measures ===")
print("  1. loads: K auto-detected per tensor (exl3.py:66 self.K = trellis.shape[-1]//16), no patch")
print("  2. top-1 agreement vs pure 3.05 on a coding corpus (the metric that tracks the bitrate")
print("     cliff; perplexity does NOT -- see bitrate-cliff-glm53)")
print("  3. rfn of per-step logits vs pure 3.05, as a magnitude check")
print("  A frankenstein that HELPS should move top-1 agreement toward the 4.05 reference;")
print("  if it moves AWAY from both packs, the hessian mismatch dominates and the idea is dead.")
