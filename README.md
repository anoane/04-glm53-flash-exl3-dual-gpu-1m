# GLM-5.3-Flash EXL3 across two GPUs at 1M context

GLM-5.3-Flash served by **TabbyAPI / exllamav3** split across an **RTX PRO 6000** (96 GB,
`sm_120`) and a **CMP 170HX** (64 GB, `sm_80`), with a **1,048,576-token fp16 KV cache** and an
MTP draft head on the second card.

**No host-RAM offload.** Everything is resident in VRAM across the two cards. That is what makes
this different from the DeepSeek-V4.1 setup, which streams experts from DDR5.

---

### Hardware this was built and measured on

| | |
|---|---|
| Host | Proxmox VE 9.2.x, kernel 7.0.x-pve, Secure Boot **disabled** |
| CPU | AMD Ryzen 9 9950X3D (16C/32T) — 24 vCPU passed to the guest |
| RAM | 160 GiB DDR5 allocated to the guest (157 GiB usable) — **no swap, deliberately** |
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell Workstation — 97,887 MiB, `sm_120`, `10de:2bb1`, PCIe Gen5 x16 (~42 GB/s H2D measured), 400 W default limit / 600 W max |
| GPU 1 | NVIDIA CMP 170HX — 65,536 MiB **after unlock** (8 GB stock), `sm_80` (GA100), `10de:20c2`, PCIe **Gen2 x1 (~0.38 GB/s)**, 200 W default limit / 250 W max |
| Guest | Ubuntu 24.04.4 LTS, kernel 6.8.0-139-generic, NVIDIA driver 610.43.02 |
| Storage | 5.8 TB NVMe (~93 GB free with all packs resident) |

**This recipe uses both GPUs and no host offload at all** — every weight is resident in VRAM
across the two cards (`gpu_split: [91, 57]` GiB). That is the key difference from repo 05.

> **This is a point-in-time recipe and may be slightly outdated or incomplete.**
> It was transcribed from a working system rather than written as a clean-room guide: driver,
> engine and image versions move quickly, some steps that were obvious in the moment are
> under-documented, and a few numbers were measured once rather than averaged. Every measured
> figure below is specific to the hardware in the table above — on a different PCIe topology,
> a different RAM size, or a card without the CMP's x1 bottleneck, the tuning will differ.
> Read it as a worked example with its reasoning shown, not as a turnkey script.

## Result

| | |
|---|---|
| Context | **1,048,576** (fp16 KV), verified by a real 1,040,000-token fill |
| Fill | 953 s at ~1,092 tok/s |
| Headroom at full fill | 2.97 GiB free on cuda:0, 4.45 GiB on cuda:1 |
| Peak temperature | 75 °C / 73 °C — thermal guard never fired |
| Weights | franken pack, whole-pack **3.3732 bpw** |

---

## The frankenstein pack

The shipped quantizations are 3.05 bpw and 4.05 bpw. Neither is ideal: 3.05 is lossy in the
layers that matter, 4.05 does not leave room for a 1M KV cache. So the pack is **spliced**.

`recipe/franken_pack_meta/FRANKENSTEIN.json` is the proven recipe:

```
K=4 expert tensors on layers [3..15]   (13 layers, taken from the 4.05 branch)
everything else from the 3.05 base
-> whole-pack 3.3732 bpw
```

Facts that make the splice legal, verified before building:

- Key sets are **identical** between 3.05 and 4.05 — **151,554 keys each** — so a whole-layer
  K swap is well defined.
- Both packs: `quant_method exl3`, `version 1.4.6`, `head_bits 6`, `mtp_bits 4`,
  `codebook mul1`, `out_scales always`. They differ **only** in `bits`.
- 43 MoE layers at indices 3..45; no target layer straddles a shard boundary; 3,456 expert
  tensors per layer.
- Building `[3..15]` rewrites **5 of the 15** shards of the 3.05 base.

### 13 layers is the ceiling

`[3..16]` (14 layers) **fails to load at 46/50 modules**. Layer choice came from a measured
per-layer sensitivity sweep (`recipe/scripts/layer_sens.py`), not from a guess and not from an
`.exl3moe` profile.

### Two traps

**Do not copy `aux_weights/*.safetensors` into a model directory.** The loader globs
`directory/*.safetensors`, so *any* safetensors file in a model dir gets loaded. Those two files
(`mtp.safetensors` 2.7 G, `kpool_aux.safetensors` 12 M) come from an older pack whose MTP head is
a *separate sibling file*. The packs used here fold the MTP head **into the sharded index as
layer 45** (`eh_proj`/`enorm`/`hnorm`/`shared_head`, 3,508 keys) and carry the 24 kpool keys
in-index too. Dropping the old files in loads a second, wrong draft head alongside the real one.
They are kept only as a fallback for the older packs.

**Generation runs on without an eos fix.** `recipe/franken_pack_meta/gen_config_franken.json`
adds eos **154828** (`<|assistant|>`); upstream omits it and replies continue into a second
assistant turn.

---

## Serving

`serve/tabbyapi-config.yml` — the parts that matter:

```yaml
model:
  max_seq_len: 1048576
  cache_size:  1048576
  max_batch_size: 1        # measured constraint, not a preference
  chunk_size: 2048         # every prefill number here was taken at this chunking
  gpu_split_auto: false
  gpu_split: [91, 57]      # GiB: Blackwell, CMP
  tool_format: glm4_5      # without this, tool calls come back as raw XML
draft_model:
  draft_mode: mtp
  draft_gpu_split: [0, 6]  # the MTP head lives entirely on the CMP
  draft_num_tokens: 1
memory:
  sysmem_recurrent_cache: 4096
```

**`gpu_split: [91, 57]`** is the whole dual-GPU story: an explicit GiB budget per card with
`gpu_split_auto: false`. Auto-split does not understand that these two cards have different
speeds and different jobs.

**`draft_gpu_split: [0, 6]`** pins the MTP draft head to the second card. The in-index MTP head
is **3.533 GiB** — budget 6, not 5, or loading dies at `Loading draft modules 33% 1/3`.

**`draft_num_tokens: 1`** — measured. Depth 1 is *faster* than deeper drafting here.

### systemd

`systemd/glm53-sec.service` caps power before start and restores after stop:

```
ExecStartPre=nvidia-smi -i 0 -pl 300      ExecStopPost=nvidia-smi -i 0 -pl 400
ExecStartPre=nvidia-smi -i 1 -pl 150      ExecStopPost=nvidia-smi -i 1 -pl 200
TimeoutStartSec=900
```

The service binds **localhost only** (`disable_auth: true` in the config) — TLS and bearer auth
belong at the reverse proxy, never on the model server.

---

## Client

`client/models.json` and `client/settings.json` are pi.dev templates with credentials replaced
by `<API_TOKEN>`. Set `contextWindow: 1048576` and point `thinkingField` at
`reasoning_content` for this model.

> A stale `defaultModel` in `settings.json` fails **silently** by falling through to another
> provider. If answers look like a different model, check that first.

---

## Layout

```
serve/tabbyapi-config.yml     the full serving config
systemd/glm53-sec.service     unit with power caps
recipe/franken_pack_meta/     FRANKENSTEIN.json + generation config
recipe/scripts/               build / verify / revert / restore / sensitivity sweep
recipe/exl3-moe-profile-patch/ exllamav3 MoE expert-placement profile patch
client/                       pi.dev templates
```

Sanitized: domains are `example.invalid`, tokens are `<API_TOKEN>`. Hardware, bit rates, layer
indices and measured numbers are real.
