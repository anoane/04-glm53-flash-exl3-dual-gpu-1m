"""1M-context fill through the SERVING path, with the prompt sized by real tokenization.

First attempt used CHARS_PER_TOK=3.6 and overshot: 3,600,114 chars -> 1,140,868 tokens, rejected
with context_length_exceeded (the validator working correctly, not a failure). Real ratio for
this text is 3.155 chars/token. Rather than trust a ratio again, this encodes locally, slices to
an exact token count, decodes, and re-encodes to VERIFY before sending.

Budget: cache 1,048,576 - max_tokens 120 - chat-template overhead. Target 1,040,000 to leave slack.
"""
import json, subprocess, threading, time, urllib.request

PACK = "/root/workspace/GLM-5.3-Flash-franken-20L"
TARGET = 1040000
MAXTOK = 120
LINE = "static int handle_%d(char *req){ char tmp[64]; sprintf(tmp, req); return 0; }\n"

peak = [0, 0]
stop = [False]

def sampler():
    while not stop[0]:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.split()
        try:
            for i in (0, 1):
                v = int(o[i])
                if v > peak[i]:
                    peak[i] = v
        except Exception:
            pass
        time.sleep(1.0)

def main():
    from exllamav3 import Config, Tokenizer
    tok = Tokenizer.from_config(Config.from_directory(PACK))

    raw = "".join(LINE % i for i in range(int(TARGET * 3.4 / 74) + 2000))
    ids = tok.encode(raw, add_bos=False)[0]
    print("raw text %d chars -> %d tokens (%.3f chars/tok)"
          % (len(raw), ids.shape[0], len(raw) / ids.shape[0]), flush=True)

    ids = ids[:TARGET]
    prompt = tok.decode(ids.unsqueeze(0))[0]
    recheck = tok.encode(prompt, add_bos=False)[0].shape[0]
    print("trimmed to %d tokens; re-encoded = %d tokens; %d chars"
          % (TARGET, recheck, len(prompt)), flush=True)
    if recheck + MAXTOK > 1048576:
        print("  still too long, trimming harder"); raise SystemExit(1)

    threading.Thread(target=sampler, daemon=True).start()
    body = {"model": "GLM-5.3-Flash-franken-20L",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAXTOK, "temperature": 0}
    req = urllib.request.Request("http://127.0.0.1:1919/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    print("POSTing %d tokens, expect ~15-17 min prefill at 300/150W..." % recheck, flush=True)
    t0 = time.time()
    ok = False
    try:
        d = json.load(urllib.request.urlopen(req, timeout=5400))
        ch = (d.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        dt = time.time() - t0
        print("  RESULT: OK in %.0fs (%.1f min)  finish=%s  -> %.0f tok/s prefill"
              % (dt, dt / 60, ch.get("finish_reason"), recheck / dt), flush=True)
        print("  answer: %s" % (m.get("content") or "")[:200].replace("\n", " "), flush=True)
        ok = True
    except urllib.error.HTTPError as e:
        print("  RESULT: HTTP %s after %.0fs  %s"
              % (e.code, time.time() - t0, e.read().decode()[:250]), flush=True)
    except Exception as e:
        print("  RESULT: %s after %.0fs: %s"
              % (type(e).__name__, time.time() - t0, str(e)[:250]), flush=True)
    stop[0] = True
    time.sleep(2)
    print("\n=== PEAK with ~1M context FILLED via the serving path ===", flush=True)
    print("  cuda:0 peak %d / 97887 MiB -> %.2f GiB free" % (peak[0], (97887 - peak[0]) / 1024))
    print("  cuda:1 peak %d / 65536 MiB -> %.2f GiB free" % (peak[1], (65536 - peak[1]) / 1024))
    print("  VERDICT: %s" % ("HOLDS AT 1M" if ok else "FAILED"))

if __name__ == "__main__":
    main()
