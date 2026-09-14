#!/usr/bin/env python3
"""Expose MoE expert-placement profiles and swap policy on the CLI."""
import sys
p = sys.argv[1]
s = open(p).read()
if "--moe_cpu_profile" in s:
    print("already patched"); sys.exit(0)

# model_init.py does not import os; the wiring below needs it.
if "\nimport os" not in s and not s.startswith("import os"):
    s = s.replace("from types import SimpleNamespace",
                  "import os\nfrom types import SimpleNamespace", 1)

anchor = [l for l in s.split("\n") if '"-mct", "--moe_cpu_threads"' in l]
assert anchor, "could not find -mct argument line"
anchor = anchor[0]
args = anchor + '''
    parser.add_argument("-mcp", "--moe_cpu_profile", type = str, help = "Precomputed MoE expert placement. Comma-separated profiles with optional weights, e.g. \'code:3,wiki:1\'. Each is a filename or a name resolved under <model_dir>/moe_profiles, $EXL3_MOE_PROFILE_DIR, or ~/.cache/exllamav3/moe_profiles. Accepts usage censuses (.npz), .safetensors and .json. Placement starts on the measured hot set instead of converging there over thousands of decode steps", default = None)
    parser.add_argument("-mcpm", "--moe_cpu_profile_mode", type = str, choices = ["seed", "static"], help = "seed (default): start from the profile and keep adapting. static: freeze the profile order, no swapping (most stable latency, never perturbs a generation)", default = None)
    parser.add_argument("-mcpd", "--moe_cpu_profile_dir", type = str, help = "Extra directory to search for profiles (os.pathsep-separated)", default = None)
    parser.add_argument("-mcpq", "--moe_cpu_profile_any_quant", action = "store_true", help = "Accept a profile built on a different checkpoint/quantization of the same model. Routing depends on quantization, so placement may be worse than dynamic; measure before trusting it. Model-identity mismatches stay fatal")
    parser.add_argument("-mcw", "--moe_cpu_swap", type = str, choices = ["on", "off"], help = "Dynamic expert placement (default: on)", default = None)
    parser.add_argument("-mcwi", "--moe_cpu_swap_interval", type = int, help = "Decode steps between placement sweeps (default: 128)", default = None)
    parser.add_argument("-mcwm", "--moe_cpu_swap_max", type = int, help = "Maximum expert swaps per sweep, across all layers (default: 64)", default = None)
    parser.add_argument("-mcwh", "--moe_cpu_swap_hysteresis", type = float, help = "Hit-count ratio a CPU-resident expert must beat to be promoted (default: 2.0)", default = None)'''
s = s.replace(anchor, args, 1)

wire_anchor = '    if getattr(args, "moe_cpu_offload", 0):'
assert wire_anchor in s, "could not find moe_cpu_offload wiring anchor"
wiring = '''    # MoE expert placement. block_sparse_mlp_cpu reads these from the environment while the
    # model loads, so they must be set before load, not after.
    _g = lambda n: vars(args).get(n)
    if _g("moe_cpu_profile_dir"): os.environ["EXL3_MOE_PROFILE_DIR"] = _g("moe_cpu_profile_dir")
    if _g("moe_cpu_profile"):     os.environ["EXL3_MOE_PROFILE"] = _g("moe_cpu_profile")
    if _g("moe_cpu_profile_mode"):os.environ["EXL3_MOE_PROFILE_MODE"] = _g("moe_cpu_profile_mode")
    if _g("moe_cpu_profile_any_quant"): os.environ["EXL3_MOE_PROFILE_ALLOW_QUANT_MISMATCH"] = "1"
    if _g("moe_cpu_swap_interval") is not None:   os.environ["EXL3_MOE_CPU_SWAP_INTERVAL"] = str(_g("moe_cpu_swap_interval"))
    if _g("moe_cpu_swap_max") is not None:        os.environ["EXL3_MOE_CPU_SWAP_MAX"] = str(_g("moe_cpu_swap_max"))
    if _g("moe_cpu_swap_hysteresis") is not None: os.environ["EXL3_MOE_CPU_SWAP_HYST"] = str(_g("moe_cpu_swap_hysteresis"))
    if _g("moe_cpu_swap") is not None:            os.environ["EXL3_MOE_CPU_SWAP"] = "1" if _g("moe_cpu_swap") == "on" else "0"
    if _g("moe_cpu_profile") and _g("moe_cpu_profile_mode") == "static":
        os.environ["EXL3_MOE_CPU_SWAP"] = "0"

''' + wire_anchor
s = s.replace(wire_anchor, wiring, 1)
open(p, "w").write(s)
print("patched model_init.py")
