"""The serving benchmark client fails a stalled request instead of waiting (bench/run_serving.py).
Regression for 2026-09-24: llama-server stalled mid-stream and the run waited about an hour."""
import http.server
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock

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

    def test_rdna3_fusion_opt_out_is_inside_container(self):
        normal = run_serving.rdna3_engine(Path("model.gguf"), 18098, 24576, ["-b", "512"])
        tuned = run_serving.rdna3_engine(Path("model.gguf"), 18098, 24576, ["-b", "512"], disable_fusion=True)
        cmd = tuned["cmd"]
        i = cmd.index("GGML_CUDA_DISABLE_FUSION=1")
        self.assertEqual(cmd[i - 1], "-e")
        self.assertLess(i, cmd.index("--entrypoint"))
        self.assertLess(i, cmd.index(run_serving.VLLM_IMAGE))
        self.assertEqual(cmd[:i - 1] + cmd[i + 1:], normal["cmd"])
        self.assertEqual(tuned["env"], {})
        self.assertEqual(tuned["stop"], normal["stop"])

    def test_container_memory_uses_server_pid_not_launcher(self):
        import run_multiturn
        spec = dict(cmd=["docker", "run"], container="test-server")
        with mock.patch.object(run_multiturn.subprocess, "check_output", return_value="123\n") as inspect, mock.patch.object(Path, "read_text", autospec=True, return_value="Name:\tserver\nVmHWM:\t4096 kB\nVmRSS:\t2048 kB\n") as read:
            got = run_multiturn.host_memory(spec, 456)
            inspect.assert_called_once_with(["docker", "inspect", "--format", "{{.State.Pid}}", "test-server"], text=True)
            read.assert_called_once_with(Path("/proc/123/status"))
        self.assertEqual(got, dict(values=dict(VmHWM="4096 kB", VmRSS="2048 kB"), scope="container-init-process", pid=123))
        with mock.patch.object(run_multiturn.subprocess, "check_output", return_value="0\n"), mock.patch.object(Path, "read_text") as read:
            self.assertEqual(run_multiturn.host_memory(spec, 456)["scope"], "unavailable")
            read.assert_not_called()
        self.assertEqual(run_multiturn.host_memory(dict(cmd=["docker", "run"]), 456)["scope"], "unavailable")

    def test_phase_idle_runs_once_per_barrier_not_per_client(self):
        import io
        import run_multiturn
        workload = dict(conversations=[dict(turns=["one", "two"])] * 2)
        def conversation(port, w, conv, recs, engine, level, rnd, barrier):
            for turn in range(2):
                if turn: barrier.wait(timeout=5)
                recs.append(dict(turn=turn, send=1, times=[2, 3], usage=dict(prompt_tokens=1, completion_tokens=1)))
        with mock.patch.object(run_multiturn, "conversation", conversation), mock.patch.object(run_multiturn.time, "sleep") as sleep:
            result = run_multiturn.level_run(0, workload, 2, io.StringIO(), "fake", 0, True, 6)
            sleep.assert_called_once_with(6)
        self.assertEqual(result["completion_tokens"], 4)
        self.assertEqual(result["errors"], [])

    def test_fixed_history_keeps_requests_equal_despite_different_answers(self):
        import copy
        import run_multiturn
        w = dict(system="system", max_tokens=128, temperature=0, seed=1234, options={}, conversations=[])
        conv = dict(name="case", turns=["first", "second"], assistant_history=["canonical"])
        w["conversations"] = [conv]
        run_multiturn.validate_history(w)
        outputs = []
        for answer in ("answer A", "answer B"):
            bodies, recs = [], []
            def turn(port, body, rec):
                bodies.append(copy.deepcopy(body))
                return answer
            with mock.patch.object(run_multiturn, "turn", turn):
                run_multiturn.conversation(0, w, conv, recs, "engine", 1, 0)
            self.assertEqual(bodies[1]["messages"][2]["content"], "canonical")
            outputs.append([r["request_sha256"] for r in recs])
        self.assertEqual(*outputs)
        del conv["assistant_history"]
        with mock.patch.object(run_multiturn, "turn", turn):
            bodies.clear()
            run_multiturn.conversation(0, w, conv, [], "engine", 1, 0)
        self.assertEqual(bodies[1]["messages"][2]["content"], "answer B")
        for bad in (None, "text", [], [1], ["one", "two"]):
            conv["assistant_history"] = bad
            with self.assertRaises(ValueError): run_multiturn.validate_history(w)

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


class InferenceXObserverTests(unittest.TestCase):
    def test_role_reasoning_content_finish_and_failure_gates(self):
        from inferencex_observer import observe, validate
        r = dict(status=200, text_times=[], text=[], finishes=[], done=False)
        observe(r, b'data: {"choices":[{"delta":{"role":"assistant"}}]}', 1)
        self.assertEqual(r['text_times'], [])
        observe(r, b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}', 2)
        observe(r, b'data: {"choices":[{"delta":{"content":"answer"}}]}', 3)
        observe(r, b'data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"completion_tokens":4}}', 4)
        self.assertEqual(r['text_times'], [2, 3])
        self.assertFalse(validate(r, 4))
        observe(r, b'data: [DONE]', 5)
        self.assertTrue(validate(r, 4))
        self.assertFalse(validate(r, 5))
        for k, v in [('status', 500), ('finishes', ['stop']), ('done', False), ('text_times', []), ('error', 'failure')]:
            self.assertFalse(validate(dict(r, **{k: v}), 4), k)

    def test_proxy_preserves_bytes_and_request_and_observes_delay(self):
        import http.client
        import json
        from inferencex_observer import Observer
        received = []
        first = b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        rest = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                received.append(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(first)
                self.wfile.flush()
                time.sleep(.03)
                self.wfile.write(rest)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with Observer(server.server_port) as observer:
                conn = http.client.HTTPConnection('127.0.0.1', observer.port)
                body = b'{"messages":[]}'
                conn.request('POST', '/v1/chat/completions', body)
                self.assertEqual(conn.getresponse().read(), first + rest)
                conn.close()
            self.assertEqual(received, [body])
            r = observer.records[0]
            self.assertEqual(r['body'], json.loads(body))
            self.assertTrue(r['done'])
            self.assertGreater(r['text_times'][0] - r['role_s'], .01)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class DemandQualityScoring(unittest.TestCase):
    def test_strict_answers_and_negative_controls(self):
        from score_demand_quality import correct
        self.assertTrue(correct('  [ 42, "ID" ]\n', [42, 'ID']))
        for text in ('[42.0,"ID"]', '[true,"ID"]', '[42]', '[42,"ID",0]',
                     '["ID",42]', '```json\n[42,"ID"]\n```', '[42,"ID"] extra',
                     '{"answer":[42,"ID"]}', '[NaN,"ID"]', None):
            self.assertFalse(correct(text, [42, 'ID']), text)
        self.assertFalse(correct('[true]', [1]))
        self.assertFalse(correct('[42]', ['42']))

    def test_complete_gate_and_corruption_controls(self):
        import hashlib
        import json
        import tempfile
        from make_demand_quality import workload
        from score_demand_quality import evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'workload.json'
            w = workload()
            fixture.write_text(json.dumps(w))
            names = ['zerv-f16', 'zerv-f16@reuse-join=128', 'vulkan', 'hip']
            resource = dict(vram_peak=20 * 1024**3, host_memory={'VmHWM': '16000000 kB'})
            manifest = dict(status='passed', workload={str(fixture): hashlib.sha256(fixture.read_bytes()).hexdigest()}, rounds=3, levels=[1, 4], engines={n: dict(resources=[resource] * 3) for n in names})
            rows, summary = [], {}
            for name in names:
                summary[name] = []
                for rnd in range(3):
                    for level in [1, 4]:
                        summary[name].append(dict(level=level, wall_s=1, turns=[dict(ttft_p50_ms=1)] * 3))
                        for conv in w['conversations'][:level]:
                            for turn, answer in enumerate(conv['expected']):
                                text = json.dumps(answer)
                                rows.append(dict(engine=name, round=rnd, level=level, conversation=conv['name'], turn=turn, output_text=text, output_sha256=hashlib.sha256(text.encode()).hexdigest(), request_sha256='same', finish_reasons=['stop'], usage=dict(prompt_tokens=100, completion_tokens=10)))
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'summary.json').write_text(json.dumps(summary))
            def score(rs):
                (root / 'raw.jsonl').write_text('\n'.join(map(json.dumps, rs)))
                return evaluate(root)
            self.assertTrue(score(rows)['quality_passed'])
            self.assertTrue(all(p['passed'] for p in score(rows)['parity'].values()))
            for field, value in [('output_text', '["wrong"]'), ('finish_reasons', ['length']), ('request_sha256', 'different'), ('error', 'timeout')]:
                corrupt = [dict(r) for r in rows]
                corrupt[-1][field] = value
                self.assertFalse(score(corrupt)['quality_passed'], field)
            self.assertFalse(score(rows[:-1])['quality_passed'])
            self.assertFalse(score(rows + [rows[-1]])['quality_passed'])
            manifest['engines']['hip']['resources'][0]['vram_peak'] = 25 * 1024**3
            (root / 'manifest.json').write_text(json.dumps(manifest))
            self.assertFalse(score(rows)['quality_passed'])

    def test_fixture_answers_from_records(self):
        import json
        from make_demand_quality import workload
        w = workload()
        self.assertEqual(w, workload())
        self.assertEqual(len(w['conversations']), 4)
        for i, conv in enumerate(w['conversations']):
            records = [json.loads(line) for line in conv['system'].splitlines()[1:]]
            self.assertEqual(len({r['id'] for r in records}), 128)
            lookup, addition, ordering = (conv['expected'][-(i % 3):] + conv['expected'][:-(i % 3)]) if i % 3 else conv['expected']
            self.assertEqual(lookup, [records[17 + i * 23]['tag']])
            self.assertEqual(addition, [records[9 + i]['quantity'] + records[119 - i]['quantity']])
            selected = [records[j] for j in (4 + i, 63 + i, 124 - i)]
            self.assertEqual(ordering, [r['id'] for r in sorted(selected, key=lambda r: (r['priority'], r['id']))])
            self.assertEqual([json.loads(s) for s in conv['assistant_history']], conv['expected'][:2])


class InferenceXReportTests(unittest.TestCase):
    def test_matrix_and_statistics(self):
        import itertools
        from summarize_inferencex import ENGINES, check_matrix, stats, percentile
        cases = [dict(engine=e, round=r, input=i, concurrency=c, output=256, status='passed')
                 for e, r, i, c in itertools.product(ENGINES, range(3), (1024, 8192), (1, 4))]
        check_matrix(cases)
        native = [c for c in cases if c['engine'] == 'zerv-tiered']
        check_matrix(native, ['zerv-tiered'])
        paired = native + [dict(c, engine='zerv-untiered') for c in native]
        check_matrix(paired, ['zerv-tiered', 'zerv-untiered'])
        for invalid in ([], ['unknown'], ['zerv-tiered', 'zerv-tiered']):
            with self.assertRaises(AssertionError): check_matrix(native, invalid)
        with self.assertRaises(AssertionError): check_matrix(native[:-1], ['zerv-tiered'])
        from run_inferencex import ENGINE_SPECS
        self.assertIn('kv-swap-mib=0,prefix-cache-tier=off', ENGINE_SPECS['zerv-untiered'])
        self.assertNotIn('disk-dir', ENGINE_SPECS['zerv-untiered'])
        self.assertIn('prefix-cache-disk-dir=', ENGINE_SPECS['zerv-tiered'])
        for engine in ('zerv-tiered', 'zerv-untiered'):
            self.assertIn('kv-pool-pages=192', ENGINE_SPECS[engine])
        for broken in (cases[:-1], cases + [cases[0]], [dict(c, status='running') for c in cases],
                       [dict(c, output=255) for c in cases]):
            with self.assertRaises(AssertionError):
                check_matrix(broken)
        self.assertEqual(stats([1, 2, 3]), dict(mean=2, sd=1, trials=[1, 2, 3]))
        self.assertEqual(percentile([30, 10, 20], .5), 20)
        self.assertAlmostEqual(percentile([10, 20], .99), 19.9)

    def test_resume_rejects_mismatched_identity_workload_and_gate_only(self):
        import copy
        import json
        import tempfile
        from run_inferencex import resume_cases
        with tempfile.TemporaryDirectory() as tmp:
            prior_root, out = Path(tmp)/'prior', Path(tmp)/'new'
            prior_root.mkdir()
            out.mkdir()
            manifest = dict.fromkeys(('model_sha256', 'native_sha256', 'llama_sha256', 'hip_sha256', 'tokenizer'), 'same')
            manifest['sources'] = dict.fromkeys(('tools/inferencex_client.py', 'bench/inferencex_observer.py', 'bazel/inferencex.bzl', 'requirements_inferencex_lock.txt'), 'same')
            for name in ('workload-1024.json', 'requests-1024.json', 'prompts-1024.jsonl'):
                (prior_root/name).write_text('same')
                (out/name).write_text('same')
            case = dict(engine='zerv-tiered', round=0, input=1024, concurrency=1, output=256,
                        status='passed', client_command=['client'], path='case')
            prior = dict(manifest, cases=[case, dict(case, status='running')])
            def resume(data):
                (prior_root/'manifest.json').write_text(json.dumps(data))
                return resume_cases(manifest, prior_root, out, [1024], [1], 256, 3)
            inherited = resume(prior)
            self.assertEqual(len(inherited), 1)
            self.assertEqual(inherited[0]['path'], str(prior_root/'case'))
            for field in ('model_sha256', 'native_sha256', 'llama_sha256', 'hip_sha256', 'tokenizer'):
                with self.assertRaises(AssertionError):
                    resume(dict(prior, **{field: 'changed'}))
            for source in manifest['sources']:
                bad = copy.deepcopy(prior)
                bad['sources'][source] = 'changed'
                with self.assertRaises(AssertionError): resume(bad)
            for changes in ({'client_command': []}, {'output': 255}, {'round': -1}, {'engine': 'unknown'}):
                with self.assertRaises(AssertionError):
                    resume(dict(prior, cases=[dict(case, **changes)]))
            with self.assertRaises(AssertionError): resume(dict(prior, cases=[case, case]))
            for name in ('workload-1024.json', 'requests-1024.json', 'prompts-1024.jsonl'):
                (out/name).write_text('different')
                with self.assertRaises(AssertionError): resume(prior)
                (out/name).write_text('same')


if __name__ == "__main__":
    unittest.main()
