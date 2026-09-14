"""Point the serving stack at the newly built UNCENSORED frankenstein.

Does NOT start the service: glm53-sec.service has Restart=on-failure, and an over-budget
pack turns that into a restart loop (8 attempts burned on the [3..16] build). Start it by
hand afterwards and watch the load.

Steps: repoint the /root/serve/models symlink farm, then update model_name in the config
by exact literal replacement (assert unique, backup, diff, re-parse).
Run: cutover_uncensored.py <pack_dir>
"""
import json, os, shutil, subprocess, sys, yaml

pack = sys.argv[1] if len(sys.argv) > 1 else "/root/workspace/GLM-5.3-Flash-Uncensored-franken-13L"
name = os.path.basename(pack)
FARM = "/root/serve/models"
CFG  = "/root/serve/glm53-sec.yml"

if not os.path.isdir(pack):
    sys.exit("!! pack not found: %s" % pack)
n_shards = len([f for f in os.listdir(pack) if f.endswith(".safetensors")])
if n_shards == 0:
    sys.exit("!! %s holds no safetensors" % pack)
print("pack: %s (%d safetensors)" % (pack, n_shards))
if os.path.exists(os.path.join(pack, "FRANKENSTEIN.json")):
    fj = json.load(open(os.path.join(pack, "FRANKENSTEIN.json")))
    print("  upgraded_layers: %s" % fj.get("upgraded_layers"))
    print("  base : %s" % fj.get("base"))
    print("  donor: %s" % fj.get("source"))

# --- 1. symlink farm: exactly ONE entry, so /v1/models lists one model -----
print("\n=== symlink farm ===")
for f in os.listdir(FARM):
    p = os.path.join(FARM, f)
    print("  removing old entry: %s -> %s" % (f, os.path.realpath(p)))
    os.remove(p)
os.symlink(pack, os.path.join(FARM, name))
print("  now: %s -> %s" % (name, os.path.realpath(os.path.join(FARM, name))))

# --- 2. model_name in the serving config ----------------------------------
print("\n=== config ===")
src = open(CFG).read()
cur = yaml.safe_load(src)["model"]["model_name"]
old_line = "  model_name: %s\n" % cur
if cur == name:
    print("  model_name already %s" % name)
else:
    assert src.count(old_line) == 1, "model_name line not unique; edit by hand"
    shutil.copy2(CFG, CFG + ".bak.uncensored")
    open(CFG, "w").write(src.replace(old_line, "  model_name: %s\n" % name))
    print(subprocess.run(["diff", CFG + ".bak.uncensored", CFG],
                         capture_output=True, text=True).stdout)

cfg = yaml.safe_load(open(CFG))
m, d = cfg["model"], cfg["draft_model"]
print("  model_name   : %s" % m["model_name"])
print("  model_dir    : %s" % m["model_dir"])
print("  max_seq_len  : %s   cache_size: %s" % (m["max_seq_len"], m["cache_size"]))
print("  gpu_split    : %s   draft_gpu_split: %s" % (m["gpu_split"], d["draft_gpu_split"]))
print("  draft_mode   : %s   draft_num_tokens: %s" % (d["draft_mode"], d["draft_num_tokens"]))
print("  tool_format  : %s" % m.get("tool_format"))
print("\nNOT started. Next: systemctl start glm53-sec.service   (watch the load)")
print("Then pi.dev needs model id + settings.defaultModel set to: %s" % name)
