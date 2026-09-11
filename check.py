# /// script
# dependencies = ["rich>=13.7"]
# ///
"""Re-check every finding the reviewer raised, against the fixed code."""
import importlib.util, sys, json, time, pathlib, re
spec = importlib.util.spec_from_file_location("tk", "/Users/dkraemer/projects/burn/burn.py")
tk = importlib.util.module_from_spec(spec); sys.modules["tk"] = tk; spec.loader.exec_module(tk)

# 1. CARRIED must never exceed a session's actual weighted spend.
led = tk.Ledger(); tk.harvest(led, 10080)
by = {}
for c in led.ordered(): by.setdefault(c.session, []).append(c)
worst, bad, wc, nc = (0, None), 0, [], []
for sess, items in by.items():
    if len(items) < 8: continue
    items.sort(key=lambda c: c.at)
    carried = sum(v[2] for v in tk.attribution(items, led.tools).values())
    actual = sum(c.weight for c in items)
    comp = len(tk.compactions(items))
    r = carried / max(actual, 1)
    (wc if comp else nc).append(r)
    if r > 1.0: bad += 1
    if r > worst[0]: worst = (r, sess)
print(f"1 CARRIED invariant: {bad} violations; worst ratio {worst[0]:.2f} ({worst[1]})")
print(f"  mean with compaction {sum(wc)/max(len(wc),1):.2f} (n={len(wc)}), "
      f"without {sum(nc)/max(len(nc),1):.2f} (n={len(nc)})")

# 2. Corrupt rates cache must not crash.
tk._RATES = None
tk.RATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
tk.RATES_CACHE.write_text("{not json at all")
try:
    r = tk.calibrated_rates(); print(f"2 corrupt cache: survived, rebuilt {len(r)} models")
except Exception as e: print(f"2 corrupt cache: STILL CRASHES {type(e).__name__}: {e}")
tk._RATES = None
tk.RATES_CACHE.write_text(json.dumps({"m": "wrong-shape"}))
try:
    r = tk.calibrated_rates(); print(f"2b bad shape: survived, rebuilt {len(r)} models")
except Exception as e: print(f"2b bad shape: STILL CRASHES {type(e).__name__}: {e}")

# 3. Codex gauge labels come from window_minutes, not the key name.
led2 = tk.Ledger()
led2.limits = {"primary": {"used_percent": 12.0, "window_minutes": 10080, "resets_at": None}}
from rich.console import Console
con = Console(width=100)
with con.capture() as cap:
    con.print(tk.meters(led2, tk.View(), [], time.time()))
plain = re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
label = [l for l in plain.splitlines() if "Codex" in l][0].split("[")[0].strip()
print(f"3 lone 7d gauge under key 'primary' labelled: {label!r}")

# 4. tool_use registered even when its record carries no usage.
led3 = tk.Ledger(); p = pathlib.Path("/tmp/fake.jsonl")
lines = [
 json.dumps({"type":"assistant","timestamp":"2026-09-11T12:00:00Z","sessionId":"s1","cwd":"/x",
   "requestId":"r1","message":{"id":"m1","model":"claude-opus-5","usage":{"input_tokens":1,
   "cache_creation_input_tokens":0,"cache_read_input_tokens":10,"output_tokens":5},
   "content":[{"type":"thinking","thinking":"..."}]}}),
 json.dumps({"type":"assistant","timestamp":"2026-09-11T12:00:01Z","sessionId":"s1","cwd":"/x",
   "requestId":"r1","message":{"id":"m1","content":[{"type":"tool_use","id":"t1","name":"Bash"}]}}),
 json.dumps({"type":"user","timestamp":"2026-09-11T12:00:02Z","sessionId":"s1",
   "message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"x"*400}]}}),
]
tk.read_claude(lines, led3, p)
print(f"4 tool_use w/o usage on its record: {len(led3.tools)} Tooling event(s) "
      f"{'OK' if len(led3.tools)==1 else 'DROPPED'}; calls={len(led3.calls)}")

# 5. Tied timestamps at the window boundary must all get credit.
led4 = tk.Ledger()
base = 1000000.0
mk = lambda at, cr, out: tk.Call(at=at, source="cc", session="s", project="p",
                                 model="claude-opus-5", counts=[0,0,cr,out])
calls = [mk(base, 1000, 10), mk(base+10, 4000, 10), mk(base+20, 5000, 10)]
led4.tools = [tk.Tooling(base+10, "s", n, 100) for n in ("Read", "Grep", "Glob")]
got = tk.attribution(calls, led4.tools)
print(f"5 three tools tied exactly at boundary: credited {sorted(k for k in got if not k.startswith('('))}")

# 6. resample guard
print(f"6 resample(width=0): {tk.resample([1.0,2.0], 0)}")
