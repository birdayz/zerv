"""Transparent loopback SSE observer; upstream client/workload remains unchanged."""
import hashlib
import http.client
import http.server
import json
import threading
import time


def observe(record, line, now):
    if not line.startswith(b'data: '): return
    data = line[6:].strip()
    if data == b'[DONE]':
        record['done'] = True
        return
    event = json.loads(data)
    if event.get('error'): record['error'] = event['error']
    if event.get('usage'): record['usage'] = event['usage']
    for choice in event.get('choices', []):
        if choice.get('finish_reason') is not None:
            record['finishes'].append(choice['finish_reason'])
        delta = choice.get('delta', {})
        if delta.get('role'): record.setdefault('role_s', now)
        text = (delta.get('reasoning_content') or delta.get('reasoning') or '') + (delta.get('content') or '')
        if text:
            record['text_times'].append(now)
            record['text'].append(text)


def validate(record, count):
    return (record.get('status') == 200 and not record.get('error') and record.get('done')
            and record.get('finishes') == ['length']
            and record.get('usage', {}).get('completion_tokens') == count
            and bool(record.get('text_times')))


class Observer:
    def __init__(self, target, port=0):
        self.records = []
        self.lock = threading.Lock()
        self.target = target
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass

            def do_POST(self):
                if self.path != '/v1/chat/completions':
                    self.send_error(404)
                    return
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 4 * 1024**2:
                    self.send_error(413)
                    return
                raw = self.rfile.read(length)
                body = json.loads(raw)
                record = dict(body=body, request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
                              send=time.perf_counter(), text_times=[], text=[], finishes=[], done=False)
                with owner.lock:
                    record['index'] = len(owner.records)
                    owner.records.append(record)
                conn = http.client.HTTPConnection('127.0.0.1', owner.target, timeout=600)
                try:
                    conn.request('POST', self.path, raw, {'Content-Type': 'application/json'})
                    response = conn.getresponse()
                    record['status'] = response.status
                    self.send_response(response.status)
                    self.send_header('Content-Type', response.getheader('Content-Type', 'text/event-stream'))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    for line in response:
                        observe(record, line, time.perf_counter())
                        self.wfile.write(line)
                        self.wfile.flush()
                    record['end'] = time.perf_counter()
                except Exception as error:
                    record['error'] = repr(error)
                finally:
                    conn.close()
                    record['output_sha256'] = hashlib.sha256(''.join(record.pop('text')).encode()).hexdigest()
                    self.close_connection = True

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)

    @property
    def port(self): return self.server.server_port

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
