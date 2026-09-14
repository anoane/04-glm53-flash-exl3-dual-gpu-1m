# GLM-5.3-Flash exl3 recipes (kept across the uncensored rebuild, 2026-09-13)

## What is here
- `franken_pack_meta/FRANKENSTEIN.json` — the PROVEN recipe: K=4 on layers **[3..15]** (13 layers),
  reverted [16..22]. 14 layers ([3..16]) FAILS to load at 46/50 modules. 13 is the ceiling.
- `franken_pack_meta/gen_config_franken.json` — generation_config with eos **154828** added
  (`<|assistant|>`); upstream omits it and replies run on into a second assistant turn.
- `franken_pack_meta/old_405_index.json` — key->shard map of the OLD censored 4.05 branch.
- `scripts/` — franken_build2 / franken_restore / franken_revert / pack_audit / layer_sens / fill_1m2.
- `aux_weights/` — see the WARNING below.

## WARNING: do NOT copy aux_weights/*.safetensors into any model directory
The loader globs `directory/*.safetensors`, so ANY safetensors file in a model dir gets loaded.
These two files come from the OLD CENSORED turboderp pack:
- `mtp.safetensors` (2.67 G) — the censored MTP draft head
- `kpool_aux.safetensors` (12 M)
The MikeRoz UNCENSORED packs do NOT need either: their MTP head is folded INTO the sharded index as
**layer 45** (`eh_proj`/`enorm`/`hnorm`/`shared_head`, 3508 keys, `mtp_bits: 4`), and the 24 kpool
keys are in-index too. Dropping the old files into a new pack would load a censored-base head
alongside the real one. They are kept ONLY as a fallback for the old packs.

## Verified facts about the uncensored pair
- Key sets are IDENTICAL between 3.05 and 4.05: **151,554 keys each** -> whole-layer K swap is valid.
- Both: `quant_method exl3`, `version 1.4.6`, `head_bits 6`, `mtp_bits 4`, `codebook mul1`,
  `out_scales always`; differ only in `bits` (3.05 vs 4.05).
- 43 MoE layers, indices 3..45. No target layer straddles a shard. 3456 expert tensors per layer.
- Shard layout CHANGED: 3.05 = 15 shards, 4.05 = **23** (the old censored 4.05 had 19).
- Building [3..15] rewrites **5 of 15** shards of the 3.05 base.
