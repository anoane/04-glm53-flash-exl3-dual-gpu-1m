# MoE expert-placement profiles for exllamav3

CPU MoE offload (`-mcs`) splits each layer's experts between VRAM and system RAM. Routing is
skewed, so *which* experts stay resident decides how much traffic crosses the memory bus.
Upstream discovers that ordering at runtime and converges slowly: a sweep moves at most
`EXL3_MOE_CPU_SWAP_MAX` (64) experts across **all** layers every
`EXL3_MOE_CPU_SWAP_INTERVAL` (128) decode steps, so a large model needs thousands of tokens
to settle, and it re-reads promoted experts from the checkpoint on disk.

This patch lets placement start correct at token zero from a **profile** — a table of
per-expert selection counts measured offline and shipped as a file. Nothing is probed or
computed at load beyond one lookup per layer.

Additive and inert unless `--moe_cpu_profile` is passed. `+107/-10` against exllamav3 dev.

## Measured

exllamav3 dev `0531096`, GLM-5.3-Flash-exl3 4.05bpw (45 layers, 42 MoE, 288 experts, topk 8)
on one RTX PRO 6000 Blackwell 96 GB + Ryzen 9 9950X3D / 128 GB DDR5. Driven through the
Generator on held-out text, `-cs 270336`, warm, first generation discarded.

    BEFORE  -mcs 180                                 (stock: dynamic placement, no profile)
    AFTER   -mcs 128 -mcp <census> -mcpm static      (census matched to domain and context)

| workload | context | before | after | |
|---|---:|---:|---:|---:|
| prose | 32,768  | 17.01 tok/s | **32.91** | 1.93x |
| prose | 262,144 | 16.82 / 16.95 | **40.24 / 39.71** | **2.37x** |
| code  | 32,768  | 16.82 | **32.78** | 1.95x |
| code  | 262,144 | 17.11 / 17.42 | **33.66 / 34.01** | 1.96x |

Prefill improves as well, since resident experts cut CPU work during prefill too:
573 -> 771 tok/s (prose 32k), 696 -> 889 (prose 256k), 691 -> 889 (code 256k).

Two contributions multiply, and only one of them needs this patch:

| lever | needs code | contribution |
|---|---|---:|
| `-mcs 180` -> `-mcs 128` (residency) | no, a flag | ~1.35x |
| census matched to domain + context | yes | ~1.64x |

Residency is free: 160 resident of 288 is simply the most that fits at `-cs 270336` on this
card (164 fails to load). The profile is what this patch is for. Prose reaches higher than
code because code routing is more diffuse -- its held-out capture is 70.1% against 74.0%
for prose at the same residency, so its floor on cold rate is higher.

## Why this exists in this form

The first version of this tool fitted a profile to **one** prompt and then reported how well
that prompt's own hot set covered that same prompt's routing. The hot set is chosen to cover
it, so the number was ~93% no matter how badly the ranking generalized — unfalsifiable by
construction. Measured live, such a profile delivered 42-45% capture against 37.5% for
placing experts at random, and was **slower than using no profile at all**.

The builder now samples many independent windows and scores capture on a disjoint held-out
split, beside two reference points that make the number readable:

| | meaning |
|---|---|
| `uniform` | what random placement scores (`R/E`). A profile at this level is worthless. |
| `oracle`  | fit on the held-out set itself: the ceiling for *any* static profile. |

It warns when held-out is within 3 points of random, and when in-sample exceeds held-out by
more than 25 points. With fewer than 4 windows it refuses to report capture at all rather
than presenting an in-sample number as if it generalized.

## Usage

```bash
# 1. build a census. -corpus takes a bundled name, a text file, or a JSON list of prompts.
#    The bundled .utf8 corpora are not installed as package data, so on an installed
#    exllamav3 pass a path (or set EXL3_CAL_DATA_DIR); the builder lists what it searched.
python util/moe_profile_build.py -m /models/GLM -cs 98304 -mcs 144 \
    -corpus /data/wikitext-2-raw/wiki.train.raw \
    -o wiki.npz -nprompts 12 -plen 65536 -gen 192 -resident 144

# 2. install where the loader looks
cp wiki.npz wiki.meta.json /models/GLM/moe_profiles/

# 3. serve
--moe_cpu_profile wiki --moe_cpu_profile_mode static -mcs 128

# audit any existing profile offline, no model load
python util/moe_profile_build.py -score wiki.npz -resident 160
```

## Flags

| flag | meaning |
|---|---|
| `-mcp` / `--moe_cpu_profile` | comma-separated profiles, optional `:weight` each |
| `-mcpm` / `--moe_cpu_profile_mode` | `static` (freeze the order) or `seed` (start hot, keep adapting) |
| `-mcpd` / `--moe_cpu_profile_dir` | extra search path |
| `-mcpq` / `--moe_cpu_profile_any_quant` | allow a checkpoint-fingerprint mismatch, with a warning |

Profiles resolve from `<model_dir>/moe_profiles`, `$EXL3_MOE_PROFILE_DIR`, or a path.
`seed` mode is new capability: upstream refuses a profile unless swapping is off.

## Two properties a profile must match

Both were measured, and getting either wrong costs the entire benefit:

**Context length.** A census fitted on 1k windows and one fitted on 64k windows produce
different rankings, and each wins in its own regime. Cold rate under a short-context census
rises from 28.7% at 32k to 55.6% at 262k; a long-context census holds it. This is a genuine
length effect, not a content artifact: with the trailing 4096 tokens held byte-identical and
only the preceding context varied, cold rate still rose 20.8% -> 28.6%.

**Domain.** A wikitext census applied to code is worth ~1.01x — nothing. A code census on
code is worth ~1.49x. Build the census from traffic that resembles the traffic you serve.

`capture` in the sidecar is only comparable when the test windows match the deployment
context length; the scorer does not enforce this, so a census can score well and still be
the wrong one to deploy.

## Safety

Profiles are per `(model, checkpoint)`. Model identity (architecture, layers, experts,
`moe_intermediate_size`, `hidden_size`) is **fatal on mismatch, even with the override**.
Checkpoint identity (`checkpoint_sha` + `quant_method`/`bits`/`head_bits`/`codebook`) is
fatal unless `--moe_cpu_profile_any_quant`. `checkpoint_sha` hashes safetensors *headers*
only: 0.03 s and 19.7 MB of reads on a 165 GB checkpoint.

## Format

The on-disk census is the usage census layout (`counts_decode` / `counts_prefill`, `int64
[prompts, layers, experts]`), so censuses from external tooling load directly. Ship the
`.meta.json` sidecar where possible: it carries `layer_keys` and the fingerprint, and keyed
matching avoids a positional off-by-one on architectures that interleave dense and sparse
MLPs. `.safetensors` / `.exl3moe` and the `EXL3_MOE_CPU_SPLIT_STATS` JSON also load.

## Measurement notes

Each of these was learned by getting it wrong first.

* **`eval/perf.py` is not valid for placement or long-context claims.** `measure_generate`
  builds state via `cache.get_test_state()`, which passes `clear=True`, so 34 of 45
  recurrent layers decode from zero state and its long-context routing is really
  short-context routing. Drive the Generator instead.
* **Report `cpu-assign/row` next to every tok/s** (`EXL3_MOE_HANDOFF_PROF=1`). It is the
  discriminator; throughput alone cannot separate a placement effect from a warm-up artifact.
* **Use in-distribution text whenever a profile is in the loop.** Synthetic filler inverts
  the sign of the result.
* **Warm up above 2048 tokens.** Crossing `index_topk` JITs and captures a new graph slot in
  every MLA module mid-generation: +5.45 ms/token over a 127-token measurement.
* **Discard the first generation after load**, and the first timed point after a fresh full
  prefill (a reproducible +4.4 to +5.4 ms).
* **Single replicates carry 8-13% noise on this hardware.** Anything under ~15% needs repeats.

## Tests

`test_moe_profile.py`, 32 cases. The load-bearing one is
`test_capture_exposes_per_prompt_overfit`: it constructs the exact pathology that shipped —
each prompt hot on its own experts — and asserts held-out capture collapses to chance while
in-sample still looks strong. That test fails against the original builder's methodology by
construction.

```bash
pytest tests/test_moe_profile.py -q
```
