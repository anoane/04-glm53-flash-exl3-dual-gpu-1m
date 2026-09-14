#!/usr/bin/env python3
"""Minimal patch: seed MoE expert placement from a precomputed profile.

Upstream supports EXL3_MOE_CPU_SPLIT_STATS but only as a *replacement* for dynamic placement
(it is refused unless EXL3_MOE_CPU_SWAP=0) and only reads a JSON of per-layer counts.

This adds EXL3_MOE_PROFILE, which:
  - accepts .npz censuses / .safetensors / .json, several at once with weights
  - loads and merges ONCE per model, not once per layer
  - supports mode=seed: start from the measured hot set AND keep adapting, which upstream
    cannot express today (profile => static only)
"""
import sys
p = sys.argv[1]
s = open(p).read()
if "EXL3_MOE_PROFILE" in s:
    print("already patched"); sys.exit(0)

old = '''        stats_path = os.environ.get("EXL3_MOE_CPU_SPLIT_STATS")
        if stats_path and self._split_dynamic:
            # Static placement from a stats file only applies with dynamic swapping disabled
            print(f" !! {self.key}: EXL3_MOE_CPU_SPLIT_STATS ignored, set EXL3_MOE_CPU_SWAP=0 to use it")
            stats_path = None
'''
new = '''        # EXL3_MOE_PROFILE: precomputed placement from one or more measured profiles.
        # mode "seed" keeps dynamic swapping enabled (start hot, keep adapting); mode
        # "static" freezes the profile order. The merged ranking is cached on the config so
        # the profile files are read once per model, not once per layer.
        prof_spec = os.environ.get("EXL3_MOE_PROFILE")
        prof_rank = None
        if prof_spec and not self.tid2eid_key:
            mode = os.environ.get("EXL3_MOE_PROFILE_MODE", "seed").lower()
            cache = getattr(self.config, "_moe_profile_cache", None)
            if cache is None:
                from ..model.moe_profile import build_ranking, ranking_lookup
                md = getattr(self.config, "model_dir", None) or getattr(self.config, "directory", None)
                ranking, keys, info = build_ranking(prof_spec, md, self.num_experts)
                cache = {"ranking": ranking, "by_key": ranking_lookup(ranking, keys),
                         "info": info, "ordinal": 0, "assigned": {}}
                self.config._moe_profile_cache = cache
                print(f" -- MoE profile: {len(info['sources'])} source(s), "
                      f"{info['layers']} layers x {info['experts']} experts, mode={mode}")
                for srcinfo in info["sources"]:
                    print(f"      {srcinfo['name']} (w={srcinfo['weight']}) <- {srcinfo['path']}")
            row = cache["by_key"].get(self.key)
            if row is None:
                # Positional profiles (.npz) carry no layer keys: assign rows to MoE
                # layers in registration order, which is the order they were captured in.
                idx = cache["assigned"].get(self.key)
                if idx is None:
                    idx = cache["ordinal"]
                    cache["assigned"][self.key] = idx
                    cache["ordinal"] += 1
                if idx < cache["ranking"].shape[0]:
                    row = cache["ranking"][idx]
            if row is not None:
                prof_rank = [int(e) for e in row]
            else:
                print(f" !! {self.key}: no profile row for this layer, placement unpermuted")
            if mode == "static":
                self._split_dynamic = False

        stats_path = os.environ.get("EXL3_MOE_CPU_SPLIT_STATS")
        if stats_path and self._split_dynamic:
            # Static placement from a stats file only applies with dynamic swapping disabled
            print(f" !! {self.key}: EXL3_MOE_CPU_SPLIT_STATS ignored, set EXL3_MOE_CPU_SWAP=0 to use it")
            stats_path = None
'''
assert old in s, "anchor A (stats_path guard) not found"
s = s.replace(old, new, 1)

# Apply the profile permutation on the same code path the JSON stats use.
old2 = '''        if stats_path:
            import json
            counts = json.load(open(stats_path)).get(self.key)
            if counts is not None and len(counts) == self.num_experts:
                perm = sorted(range(self.num_experts), key = lambda e: -counts[e])
                self._split_perm = perm
'''
new2 = '''        if prof_rank is not None and len(prof_rank) == self.num_experts:
            # Precomputed hot->cold order: no counts to sort at load, just apply it.
            self._split_perm = prof_rank
            if self.gated:
                self.gates = [self.gates[e] for e in prof_rank]
            self.ups = [self.ups[e] for e in prof_rank]
            self.downs = [self.downs[e] for e in prof_rank]
            stats_path = None
        if stats_path:
            import json
            counts = json.load(open(stats_path)).get(self.key)
            if counts is not None and len(counts) == self.num_experts:
                perm = sorted(range(self.num_experts), key = lambda e: -counts[e])
                self._split_perm = perm
'''
assert old2 in s, "anchor B (stats_path apply) not found"
s = s.replace(old2, new2, 1)

open(p, "w").write(s)
print("patched block_sparse_mlp_cpu.py")
