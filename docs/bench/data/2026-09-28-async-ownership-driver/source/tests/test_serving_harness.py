"""The serving benchmark client fails a stalled request instead of waiting (bench/run_serving.py).
Regression for 2026-09-24: llama-server stalled mid-stream and the run waited about an hour."""
import http.server
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]/"bench"))
import run_serving  # noqa: E402


class Handler(http.server.BaseHTTPRequestHandler):
    mode = "stall-body"

    def do_POST(self):
        try:
            self.respond()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up, as intended

    def respond(self):
        self.rfile.read(int(self.headers["content-length"]))
        if Handler.mode == "stall-headers":
            time.sleep(3)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
        self.wfile.flush()
        if Handler.mode == "stall-body":
            time.sleep(3)
            return
        if Handler.mode == "missing-done":
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
            return
        if Handler.mode == "stream-error":
            self.wfile.write(b'data: {"error":{"message":"generation failed"}}\n\ndata: [DONE]\n\n')
            return
        if Handler.mode == "timings":
            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"draft_n":12,"draft_n_accepted":9}}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return
        # "slow": one event every 0.2 s for 2 s (never silent for long, but too long in total)
        for _ in range(10):
            time.sleep(0.2)
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n')
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class StallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def request(self, mode, stall_s, limit_s):
        Handler.mode = mode
        start = time.perf_counter()
        with self.assertRaises(run_serving.RequestStalled):
            run_serving.stream_request(self.server.server_address[1], {"messages": []}, stall_s, limit_s)
        return time.perf_counter() - start

    def test_stall_mid_stream(self):
        self.assertLess(self.request("stall-body", 0.5, 60), 2.5)

    def test_stall_before_headers(self):
        self.assertLess(self.request("stall-headers", 0.5, 60), 2.5)

    def test_total_limit(self):
        self.assertLess(self.request("slow", 5, 0.5), 2.5)


class SpecCounterTests(unittest.TestCase):
    """Draft acceptance: llama-server's per-request timings and zerv's /metrics counters."""

    def test_llama_timings(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            Handler.mode = "timings"
            r = run_serving.stream_request(server.server_address[1], {"messages": []}, 5, 5)
        finally:
            server.shutdown()
        self.assertEqual(r["spec"], dict(drafted=12, verified=12, accepted=9))
        self.assertEqual(r["finish"], "stop")

    def test_multiturn_rejects_in_stream_errors(self):
        import run_multiturn
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            Handler.mode = "stream-error"
            record = {}
            self.assertEqual("", run_multiturn.turn(server.server_address[1], {}, record))
            self.assertIn("generation failed", record["error"])
            self.assertNotIn("output_sha256", record)
        finally:
            server.shutdown()

    def test_multiturn_rejects_incomplete_stream(self):
        import run_multiturn
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            Handler.mode = "missing-done"
            record = {}
            self.assertEqual("", run_multiturn.turn(server.server_address[1], {}, record))
            self.assertIn("without [DONE]", record["error"])
            self.assertNotIn("output_sha256", record)
        finally:
            server.shutdown()

    def test_zerv_metrics(self):
        text = "\n".join(["# TYPE zerv_spec_verifies_total counter", "zerv_spec_verifies_total 7",
                          'zerv_spec_draft_tokens_total{stage="drafted"} 21', 'zerv_spec_draft_tokens_total{stage="verified"} 20',
                          'zerv_spec_draft_tokens_total{stage="accepted"} 15', "zerv_requests_total 3"])
        self.assertEqual(run_serving.spec_counters(text), dict(verifies=7, drafted=21, verified=20, accepted=15))
        self.assertIsNone(run_serving.spec_counters("zerv_requests_total 3"))


class EngineSuffixTests(unittest.TestCase):
    def test_disk_directory_log_names_are_bounded_basenames(self):
        import run_multiturn
        for engine in ("zerv-f16@prefix-cache-disk-dir=third_party/nvme-probe", "zerv@x=" + "a" * 300, "../escape"):
            name = run_multiturn.log_name(engine, 3)
            self.assertEqual(Path(name).name, name)
            self.assertLess(len(name.encode()), 255)
            self.assertEqual(name, run_multiturn.log_name(engine, 3))
            self.assertNotEqual(name, run_multiturn.log_name(engine, 4))

    def test_zerv_flags(self):
        table = {"zerv": dict(cmd=["/bin/zerv", "--model", "m"], env={}), "llama": dict(cmd=["/usr/bin/llama-server"], env={})}
        spec = run_serving.resolve_engine(table, "zerv@embedding-memory=device,spec-draft=3", "/bin/zerv")
        self.assertEqual(spec["cmd"], ["/bin/zerv", "--model", "m", "--embedding-memory", "device", "--spec-draft", "3"])
        self.assertEqual(table["zerv"]["cmd"], ["/bin/zerv", "--model", "m"])  # the table is unchanged
        self.assertEqual(run_serving.resolve_engine(table, "llama", "/bin/zerv"), table["llama"])
        with self.assertRaises(SystemExit):
            run_serving.resolve_engine(table, "llama@x=1", "/bin/zerv")
        with self.assertRaises(SystemExit):
            run_serving.resolve_engine(table, "zerv@novalue", "/bin/zerv")


if __name__ == "__main__":
    unittest.main()
