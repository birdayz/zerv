import json, time, urllib.request, sys
port=sys.argv[1]
body=json.load(open("third_party/kv-split/needle.json"))
t0=time.time(); first=None; text=""; usage=None
r=urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=1800)
for line in r:
    line=line.decode().strip()
    if not line.startswith("data:") or line=="data: [DONE]": continue
    d=json.loads(line[5:])
    if d.get("usage"): usage=d["usage"]
    for ch in d.get("choices",[]):
        c=ch["delta"].get("content") or ""
        if c and first is None: first=time.time()
        text+=c
end=time.time()
print(json.dumps(dict(usage=usage, ttft_s=first-t0, total_s=end-t0, decode_tok_s=(usage['completion_tokens']-1)/(end-first), text=text)))
