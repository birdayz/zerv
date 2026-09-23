#!/usr/bin/env python3
"""Pass-through HTTP proxy for an OpenAI-compatible server that records every
`POST /v1/chat/completions` request body, in order, as JSON lines. Responses (including
SSE streams) are forwarded unchanged. Used to capture real client sessions (e.g. bruh)
for replay benchmarks. Usage: record_proxy.py --listen 18201 --upstream 18080 --output FILE"""
import argparse
import http.client
import http.server
import json
from pathlib import Path
import threading


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--upstream", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.open("x")
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def forward(self, body):
            c = http.client.HTTPConnection("127.0.0.1", a.upstream, timeout=3600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
            c.request(self.command, self.path, body, headers)
            r = c.getresponse()
            self.send_response(r.status)
            chunked = r.getheader("transfer-encoding", "").lower() == "chunked"
            for k, v in r.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
            if chunked:
                self.send_header("transfer-encoding", "chunked"); self.end_headers()
                while True:
                    chunk = r.read1(65536)
                    if not chunk: break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            else:
                data = r.read()
                self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
            c.close()

        def do_GET(self):
            self.forward(None)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            if self.path.endswith("/chat/completions"):
                with lock:
                    out.write(json.dumps(json.loads(body)) + "\n"); out.flush()
            self.forward(body)

        def log_message(self, *args):
            pass

    http.server.ThreadingHTTPServer(("127.0.0.1", a.listen), Handler).serve_forever()


if __name__ == "__main__":
    main()
