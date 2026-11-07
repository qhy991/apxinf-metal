"""Check the real HTTP binary with an original CPU protocol fixture.

No model, MLX package, GPU, or external service is required.
Each test starts its own process group and retains diagnostics under devlocal.
This runner owns fixture launch, process groups, and cleanup for every Rust
HTTP test target; `--select` and `--exclude` map its cases onto those targets.
"""

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[3]
BINARY = None
RUN_ROOT = None
MODEL = "original-cpu-fixture"


def body(scenario="text", **fields):
    return {"model": MODEL, "max_tokens": 8,
            "messages": [{"role": "user", "content": "fixture:" + scenario}], **fields}


def without_limit(scenario="text"):
    """Omit max_tokens so the gateway applies the deployment output budget."""
    request = body(scenario)
    del request["max_tokens"]
    return request


def until(probe, description, seconds=4):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = probe()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Timed out: " + description)


def frames(response):
    event, data = "message", []
    while line := response.readline():
        line = line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                text = "\n".join(data)
                yield event, text if text == "[DONE]" else json.loads(text)
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        raise AssertionError("The SSE response ended inside a frame.")


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def matches(test, entries):
    owner = type(test).__name__
    return any(entry == owner or entry == owner + "." + test._testMethodName for entry in entries)


class Gateway:
    def __init__(self, name, timeout=5, max_output_tokens=64, request_timeout=4):
        self.root = Path(tempfile.mkdtemp(prefix=name + "-", dir=RUN_ROOT))
        self.request_timeout = request_timeout
        adapter = self.root / "python/apxinf/apxinf/serving/text_worker.py"
        adapter.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "tests/fixtures/serving/http_worker.py", adapter)
        shutil.copyfile(ROOT / "python/apxinf/apxinf/serving/contracts.py", adapter.with_name("contracts.py"))
        pins = self.root / "scripts/apxinf_mlx_generate.py"
        pins.parent.mkdir()
        shutil.copyfile(ROOT / "scripts/apxinf_mlx_generate.py", pins)
        model = self.root / "model"
        model.mkdir()
        (model / "fixture.txt").write_text("Original CPU HTTP fixture. No model weights.\n")
        self.connections = []
        self.forced_cleanup = False
        self.closed = False
        self.log = (self.root / "gateway.log").open("wb")
        self.process = subprocess.Popen([
            BINARY, "--model", str(model), "--model-id", MODEL,
            "--python", str(Path(sys.executable).resolve()), "--worker", str(adapter),
            "--port", "0", "--host-pressure-policy", "disabled",
            "--max-context", "128", "--max-output-tokens", str(max_output_tokens),
            "--queue-capacity", "1", "--timeout-seconds", str(timeout),
            "--memory-budget-bytes", "8192", "--sequence-reservation-bytes", "1024",
        ], stdout=self.log, stderr=self.log, start_new_session=True, cwd=ROOT)
        try:
            self.port = until(self.listener, "gateway startup", seconds=15)
            until(lambda: self.request("/readyz")[0] == 200, "HTTP readiness", seconds=15)
        except BaseException:
            self.close()
            raise

    def listener(self):
        text = (self.root / "gateway.log").read_text()
        if self.process.poll() is not None:
            raise AssertionError("Gateway startup failed:\n" + text)
        match = re.search(r"listener bound at http://127\.0\.0\.1:(\d+)", text)
        return int(match[1]) if match else None

    def trace(self):
        path = self.root / "http-worker-events.jsonl"
        result = []
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrent append can expose the final incomplete line.
        return result

    def send(self, path, document=None, raw=None, method=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.request_timeout)
        self.connections.append(connection)
        payload = raw if raw is not None else json.dumps(document) if document is not None else None
        connection.request(method or ("GET" if payload is None else "POST"), path, payload,
                           {"Content-Type": "application/json"})
        return connection

    def request(self, path, document=None, raw=None, method=None):
        connection = self.send(path, document, raw, method)
        try:
            response = connection.getresponse()
            content = response.read()
            headers = dict(response.getheaders())
            if "application/json" in response.getheader("content-type", ""):
                content = json.loads(content)
            else:
                content = content.decode()
            return response.status, headers, content
        finally:
            connection.close()

    def stream(self, path, document):
        connection = self.send(path, document)
        response = connection.getresponse()
        assert response.status == 200, response.read()
        assert response.getheader("content-type").startswith("text/event-stream")
        return connection, response

    def metrics(self):
        status, _, text = self.request("/metrics")
        assert status == 200
        return {line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
                for line in text.splitlines() if line and not line.startswith("#")}

    def idle(self):
        names = ["apxinf_active_requests", "apxinf_queued_requests", "apxinf_capacity_reserved_bytes",
                 'apxinf_cleanup_pending{operation="generate"}',
                 'apxinf_cleanup_pending{operation="count_tokens"}']
        def settled():
            metrics = self.metrics()
            return all(metrics.get(name) == 0 for name in names)
        return until(settled, "resource settlement", seconds=15)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            connection.close()
        try:
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    self.forced_cleanup = True
        finally:
            # The owned process group can survive an unexpected gateway exit.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.forced_cleanup = True
            except ProcessLookupError:
                pass
            finally:
                self.process.wait(timeout=3)
                self.log.close()


class GatewayChecks:
    """Assertions shared by every case that drives a gateway."""

    def assert_error(self, response, status, code, error_type=None):
        actual, headers, content = response
        self.assertEqual(actual, status, content)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(content["type"], "error")
        self.assertEqual(content["error"]["code"], code)
        if error_type is not None:
            self.assertEqual(content["error"]["type"], error_type, code)

    def assert_reaped(self, gateway):
        gateway.close()
        self.assertEqual(gateway.process.returncode, 0, "Graceful shutdown failed.")
        self.assertFalse(gateway.forced_cleanup, "The CLI must reap its worker.")
        worker_pid = next(item["pid"] for item in gateway.trace() if item["event"] == "started")
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)


class HttpGatewayTests(GatewayChecks, unittest.TestCase):
    def setUp(self):
        self.gateway = Gateway(self._testMethodName, timeout=1 if "deadline" in self._testMethodName else 5)
        self.addCleanup(self.gateway.close)

    def stream_all(self, path, document):
        connection, response = self.gateway.stream(path, document)
        try:
            return dict(response.getheaders()), list(frames(response))
        finally:
            response.close()
            connection.close()

    def test_routes_counting_and_nonstreamed_responses(self):
        gateway = self.gateway
        self.assertEqual(gateway.request("/healthz")[2], {"status": "alive"})
        status, _, ready = gateway.request("/readyz")
        self.assertEqual(status, 200)
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["worker"]["runtime"]["mlx"], "absent")
        self.assertFalse(ready["worker"]["capabilities"]["exact_append"])
        self.assertEqual(ready["host_pressure"]["policy"], "disabled")
        models = gateway.request("/v1/models")[2]["data"]
        self.assertEqual([item["id"] for item in models], [MODEL])
        self.assertEqual(gateway.request("/v1/messages/count_tokens", body())[2], {"input_tokens": 2})
        self.assertFalse(any(item["event"] == "accepted" for item in gateway.trace()))
        for path in ("/v1/messages", "/v1/chat/completions"):
            status, headers, reply = gateway.request(path, body())
            self.assertEqual(status, 200, reply)
            self.assertNotIn("transfer-encoding", headers, path)
            self.assertGreater(int(headers["content-length"]), 0, path)
            uuid.UUID(headers["x-request-id"])
            self.assertEqual(headers["x-apxinf-model-revision"], ready["worker"]["model_revision"])
            self.assertEqual(reply["model"], MODEL)
            if path.endswith("messages"):
                self.assertEqual(reply["content"], [{"type": "text", "text": "Original reply."}])
                self.assertEqual(reply["stop_reason"], "end_turn")
                self.assertEqual(reply["usage"], {"input_tokens": 2, "output_tokens": 3})
            else:
                self.assertEqual(reply["choices"][0]["message"]["content"], "Original reply.")
                self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
                self.assertEqual(reply["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        gateway.idle()
        self.assertEqual(gateway.metrics()['apxinf_requests_total{status="completed"}'], 3)

    def test_anthropic_stream_indices_usage_and_terminal(self):
        headers, events = self.stream_all("/v1/messages", body(stream=True))
        self.assertEqual([name for name, _ in events], ["message_start", "content_block_start",
                         "content_block_delta", "content_block_delta", "content_block_stop",
                         "message_delta", "message_stop"])
        for name, value in events:
            self.assertEqual(value["type"], name)
            if "index" in value:
                self.assertEqual(value["index"], 0)
        self.assertEqual(events[0][1]["message"]["id"], "msg_" + headers["x-request-id"])
        self.assertEqual(events[-2][1]["usage"], {"output_tokens": 3})
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "end_turn")
        self.gateway.idle()

    def test_openai_stream_usage_modes(self):
        for include in (None, False, True):
            with self.subTest(include_usage=include):
                request = body(stream=True)
                if include is not None:
                    request["stream_options"] = {"include_usage": include}
                headers, events = self.stream_all("/v1/chat/completions", request)
                self.assertEqual(events[-1], ("message", "[DONE]"))
                chunks = [value for _, value in events[:-1]]
                self.assertTrue(all(chunk["id"] == "chatcmpl-" + headers["x-request-id"] for chunk in chunks))
                text = "".join(choice["delta"].get("content", "") for chunk in chunks for choice in chunk["choices"])
                self.assertEqual(text, "Original reply.")
                if include:
                    self.assertEqual(chunks[-1]["choices"], [])
                    self.assertEqual(chunks[-1]["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
                    self.assertTrue(all(chunk["usage"] is None for chunk in chunks[:-1]))
                    self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
                else:
                    self.assertTrue(all("usage" not in chunk for chunk in chunks))
                    self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_openai_content_part_arrays_are_normalized(self):
        gateway = self.gateway
        content = [{"type": "text", "text": "fixture:"}, {"type": "text", "text": "text"}]
        status, _, reply = gateway.request("/v1/chat/completions", {
            "model": MODEL, "max_tokens": 8,
            "messages": [{"role": "user", "content": content}]})
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["choices"][0]["message"]["content"], "Original reply.")
        prepared = [item for item in gateway.trace() if item["event"] == "prepared"]
        self.assertEqual(prepared[-1]["messages"], [{"role": "user", "content": "fixture:text"}])

        # OpenAI image parts remain unsupported on the text profile.
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        response = gateway.request("/v1/chat/completions", {
            "model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": [image]}]})
        self.assert_error(response, 400, "unsupported_feature")
        gateway.idle()

    def test_invalid_input_never_reaches_worker(self):
        gateway = self.gateway
        for path in ("/v1/messages", "/v1/chat/completions", "/v1/messages/count_tokens"):
            for raw in ('{"model":', '{"model":"a","model":"b"}'):
                self.assert_error(gateway.request(path, raw=raw), 400, "invalid_request")
            self.assert_error(gateway.request(path, body(model="absent")), 404, "model_not_found")
            self.assert_error(gateway.request(path, body(temperature=0.5)), 400, "unsupported_feature")
        self.assert_error(gateway.request("/v1/messages", body(thinking={"type": "adaptive"})), 400, "unsupported_feature")
        self.assert_error(gateway.request("/v1/chat/completions", body(max_completion_tokens=4)), 400, "invalid_request")
        self.assertEqual(gateway.request("/v1/messages", raw=" " * 1_000_001)[0], 413)
        self.assertEqual(gateway.request("/not-a-route")[0], 404)
        self.assertEqual(gateway.request("/v1/messages", method="GET")[0], 405)
        self.assertEqual([entry["event"] for entry in gateway.trace()], ["started"])
        self.assertEqual(gateway.request("/readyz")[0], 200)

    def test_tools_and_stop_sequences_use_public_completion(self):
        schema = {"type": "object", "properties": {"path": {"type": "string"}},
                  "required": ["path"], "additionalProperties": False}
        request = body("tool", stream=True, tools=[{"name": "read", "input_schema": schema}])
        _, events = self.stream_all("/v1/messages", request)
        starts = [value for name, value in events if name == "content_block_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["index"], 0)
        self.assertEqual(starts[0]["content_block"]["type"], "tool_use")
        self.assertEqual(starts[0]["content_block"]["name"], "read")
        arguments = "".join(value["delta"]["partial_json"] for name, value in events if name == "content_block_delta")
        self.assertEqual(json.loads(arguments), {"path": "marker.txt"})
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")
        reply = self.gateway.request("/v1/messages", body("stop", stop_sequences=["STOP"]))[2]
        self.assertEqual(reply["content"], [{"type": "text", "text": "visible "}])
        self.assertEqual((reply["stop_reason"], reply["stop_sequence"]), ("stop_sequence", "STOP"))
        self.gateway.idle()

    def test_preparation_and_stream_failures_keep_worker_usable(self):
        self.assert_error(self.gateway.request("/v1/messages", body("prepare-error")), 400, "invalid_request")
        for path in ("/v1/messages", "/v1/chat/completions"):
            _, events = self.stream_all(path, body("fail", stream=True))
            self.assertEqual(events[-1][0], "error")
            self.assertEqual(events[-1][1]["error"]["code"], "internal_error")
            self.assertNotIn(("message", "[DONE]"), events)
            self.assertFalse(any(name == "message_stop" for name, _ in events))
        self.gateway.idle()
        self.assertEqual(self.gateway.request("/readyz")[0], 200)
        self.assertEqual(self.gateway.request("/v1/messages", body())[0], 200)

    def test_worker_declared_failure_status_follows_the_contract(self):
        # The public status comes from the documented error-code table, not from
        # a generic backend-failure default.
        cases = (
            ("unavailable", 503, "model_unavailable", "overloaded_error"),
            ("capacity", 503, "capacity_unavailable", "overloaded_error"),
            ("prefix-mismatch", 409, "prefix_mismatch", "api_error"),
            ("session-missing", 404, "session_not_found", "not_found_error"),
            ("fail", 500, "internal_error", "api_error"),
        )
        for scenario, status, code, error_type in cases:
            with self.subTest(scenario=scenario):
                for path in ("/v1/messages", "/v1/chat/completions"):
                    response = self.gateway.request(path, body(scenario))
                    self.assert_error(response, status, code, error_type)
        # A worker-declared request failure does not fence a healthy worker.
        self.assertEqual(self.gateway.request("/readyz")[0], 200)
        self.assertEqual(self.gateway.request("/v1/messages", body())[0], 200)
        self.gateway.idle()

    def test_worker_declared_failure_stream_error_type_follows_the_status(self):
        # A stream error event reports the same status-derived error type as the
        # non-streamed body, and it never reports a successful finish.
        for path in ("/v1/messages", "/v1/chat/completions"):
            _, events = self.stream_all(path, body("unavailable", stream=True))
            errors = [value for name, value in events if name == "error"]
            self.assertEqual(len(errors), 1, path)
            self.assertEqual(errors[0]["error"]["code"], "model_unavailable", path)
            self.assertEqual(errors[0]["error"]["type"], "overloaded_error", path)
            self.assertFalse(any(name == "message_stop" for name, _ in events), path)
            self.assertNotIn(("message", "[DONE]"), events, path)
        self.assertEqual(self.gateway.request("/readyz")[0], 200)
        self.gateway.idle()

    def test_disconnects_reclaim_waiting_queue_and_active_reservation(self):
        gateway = self.gateway
        active, response = gateway.stream("/v1/messages", body("slow", stream=True, max_tokens=64))
        events = frames(response)
        until(lambda: next(events)[0] == "content_block_delta", "first active output")
        queued = gateway.send("/v1/messages", body())
        until(lambda: gateway.metrics()["apxinf_queued_requests"] == 1, "queued request")
        self.assert_error(gateway.request("/v1/messages", body()), 429, "queue_full",
                         "rate_limit_error")
        queued.close()
        until(lambda: gateway.metrics()["apxinf_queued_requests"] == 0, "queued disconnect")
        self.assertEqual(len([item for item in gateway.trace() if item["event"] == "prepared"]), 1)
        response.close()
        active.close()
        gateway.idle()
        self.assertEqual(gateway.metrics()['apxinf_requests_total{status="cancelled"}'], 2)
        controls = [item for item in gateway.trace() if item["event"] == "control"]
        self.assertEqual([item["kind"] for item in controls], ["cancel_request"])
        self.assertEqual(gateway.request("/v1/messages", body())[0], 200)

    def test_deadline_emits_error_and_releases_resources(self):
        _, events = self.stream_all("/v1/messages", body("slow", stream=True, max_tokens=64))
        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["error"]["code"], "deadline_exceeded")
        self.assertEqual(events[-1][1]["error"]["type"], "api_error")
        self.assertFalse(any(name == "message_stop" for name, _ in events))
        self.gateway.idle()
        self.assertEqual(self.gateway.metrics()['apxinf_requests_total{status="expired"}'], 1)
        self.assertEqual(self.gateway.request("/readyz")[0], 200)

    def test_worker_exit_fences_readiness_but_keeps_health(self):
        _, events = self.stream_all("/v1/messages", body("exit", stream=True))
        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["error"]["code"], "worker_lost")
        until(lambda: self.gateway.request("/readyz")[0] == 503, "worker failure readiness")
        self.gateway.idle()
        self.assertEqual(self.gateway.request("/healthz")[0], 200)
        self.assert_error(self.gateway.request("/v1/messages", body()), 503, "worker_lost")

    def test_count_tokens_protocol_fault_keeps_specific_error(self):
        self.assert_error(self.gateway.request("/v1/messages/count_tokens", body("prepare-fault")),
                          500, "protocol_fault")
        self.assertEqual(self.gateway.request("/readyz")[0], 503)
        self.gateway.idle()

    def test_stream_protocol_fault_has_one_error_without_success(self):
        for path in ("/v1/messages", "/v1/chat/completions"):
            if path.endswith("completions"):
                self.gateway.close()
                self.gateway = Gateway(self._testMethodName + "-openai")
                self.addCleanup(self.gateway.close)
            _, events = self.stream_all(path, body("stream-fault", stream=True))
            errors = [value for name, value in events if name == "error"]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["error"]["code"], "protocol_fault")
            self.assertFalse(any(name == "message_stop" or value == "[DONE]" for name, value in events))
            self.assertEqual(self.gateway.request("/readyz")[0], 503)
            self.gateway.idle()

    def test_openai_developer_messages_and_message_shape_errors(self):
        gateway = self.gateway
        # A developer message carries the system instructions; the worker sees
        # the normalized system role, not the OpenAI-only role name.
        request = {"model": MODEL, "max_tokens": 4, "messages": [
            {"role": "developer", "content": "fixture:text"},
            {"role": "user", "content": "fixture:text"},
        ]}
        self.assertEqual(gateway.request("/v1/chat/completions", request)[0], 200)
        prepared = [item for item in gateway.trace() if item["event"] == "prepared"]
        self.assertEqual(prepared[-1]["messages"], [
            {"role": "system", "content": "fixture:text"},
            {"role": "user", "content": "fixture:text"},
        ])
        # An unsupported role or content shape is an actionable request error.
        for message, expected in (
            ({"role": "function", "content": "hello"},
             "system, developer, user, assistant, or tool"),
            ({"role": "user", "content": {"text": "hello"}},
             "a string or a content-part array"),
        ):
            response = gateway.request("/v1/chat/completions",
                                       {"model": MODEL, "max_tokens": 4, "messages": [message]})
            self.assert_error(response, 400, "invalid_request", "invalid_request_error")
            self.assertIn(expected, response[2]["error"]["message"])
        # Only the successful request reached the worker, which stays usable.
        self.assertEqual(len([item for item in gateway.trace() if item["event"] == "accepted"]), 1)
        self.assertEqual(gateway.request("/v1/messages", body())[0], 200)
        gateway.idle()

    def test_cli_shutdown_reaps_worker(self):
        gateway = self.gateway
        gateway.close()
        self.assertEqual(gateway.process.returncode, 0)
        self.assertFalse(gateway.forced_cleanup, "The CLI must reap its worker without fixture intervention.")
        worker_pid = next(item["pid"] for item in gateway.trace() if item["event"] == "started")
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)


class OutputBudgetTests(GatewayChecks, unittest.TestCase):
    """The public output budget keeps ordinary requests usable."""

    def open_gateway(self, max_output_tokens):
        # The slow fixture emits one token per 0.15s, so both deadlines keep
        # room above a small deployment budget instead of racing the budget.
        gateway = Gateway(self._testMethodName, timeout=15,
                          max_output_tokens=max_output_tokens, request_timeout=15)
        self.addCleanup(gateway.close)
        return gateway

    def test_omitted_limit_is_admitted_with_the_deployment_budget(self):
        gateway = self.open_gateway(8)
        # A zero limit is a request error and never reaches queue admission.
        for path in ("/v1/messages", "/v1/chat/completions"):
            request = body("text")
            request["max_tokens"] = 0
            response = gateway.request(path, request)
            self.assert_error(response, 400, "invalid_request", "invalid_request_error")
            self.assertIn("at least 1", response[2]["error"]["message"], path)
        self.assertFalse(any(item["event"] == "accepted" for item in gateway.trace()))
        self.assertEqual(gateway.metrics()['apxinf_requests_total{status="failed"}'], 0)

        self.assertEqual(gateway.request("/v1/messages/count_tokens", body())[2], {"input_tokens": 2})

        # An omitted limit reserves the deployment budget, so the worker emits
        # exactly that many tokens instead of receiving 512 and failing it.
        status, _, reply = gateway.request("/v1/chat/completions", without_limit("slow"))
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["choices"][0]["finish_reason"], "length", reply)
        self.assertEqual(reply["usage"]["completion_tokens"], 8, reply)
        status, _, reply = gateway.request("/v1/messages", without_limit("slow"))
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["stop_reason"], "max_tokens", reply)
        self.assertEqual(reply["usage"]["output_tokens"], 8, reply)

        gateway.idle()
        self.assertEqual(len([item for item in gateway.trace() if item["event"] == "accepted"]), 2)
        self.assertEqual(gateway.metrics()['apxinf_requests_total{status="completed"}'], 3)
        self.assert_reaped(gateway)

    def test_exhausted_output_budget_rejects_generation_but_counts_tokens(self):
        gateway = self.open_gateway(0)
        self.assertEqual(gateway.request("/readyz")[0], 200)
        for path in ("/v1/messages", "/v1/chat/completions"):
            response = gateway.request(path, without_limit())
            self.assert_error(response, 400, "invalid_request")
            self.assertEqual(response[2]["error"]["message"],
                             "This deployment allows no output tokens.", path)
        # An explicit limit above an exhausted deployment keeps the older message.
        over = body("text")
        over["max_tokens"] = 1
        response = gateway.request("/v1/chat/completions", over)
        self.assert_error(response, 400, "invalid_request")
        self.assertEqual(response[2]["error"]["message"],
                         "max_tokens exceeds the deployment output limit.")

        self.assertEqual(gateway.request("/v1/messages/count_tokens", body())[2], {"input_tokens": 2})
        gateway.idle()
        self.assertFalse(any(item["event"] == "accepted" for item in gateway.trace()))
        self.assertEqual(gateway.metrics()['apxinf_requests_total{status="failed"}'], 0)
        self.assert_reaped(gateway)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--select", action="append", default=[],
                        help="Run only these case names or classes.")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Skip these case names or classes.")
    parser.add_argument("--expect-cases", type=int,
                        help="Fail unless exactly this many cases are selected.")
    parser.add_argument("--expect-total", type=int,
                        help="Fail unless the module defines exactly this many cases.")
    parser.add_argument("--count-only", action="store_true",
                        help="Validate the case mapping and exit without running a gateway.")
    args = parser.parse_args()
    BINARY = str(Path(args.binary).resolve(strict=True))
    cases = list(flatten(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])))
    # A new case that no target selects would otherwise never run in cargo test.
    if args.expect_total is not None and len(cases) != args.expect_total:
        raise AssertionError(f"Expected {args.expect_total} cases in the module, got {len(cases)}.")
    selected = unittest.TestSuite(
        test for test in cases
        if (not args.select or matches(test, args.select)) and not matches(test, args.exclude))
    count = selected.countTestCases()
    if args.select and not count:
        raise AssertionError("No test matched: " + ", ".join(args.select))
    # A stale --select/--exclude mapping must fail instead of silently
    # re-running other cases or an empty suite.
    if args.expect_cases is not None and count != args.expect_cases:
        raise AssertionError(f"Expected {args.expect_cases} selected cases, got {count}.")
    if args.count_only:
        print(f"Selected {count} cases.", file=sys.stderr)
        sys.exit(0)
    base = Path(os.environ.get("APXINF_HTTP_TEST_OUTPUT", str(ROOT / "devlocal/serving-http-integration"))).resolve()
    base.mkdir(parents=True, exist_ok=True)
    RUN_ROOT = Path(tempfile.mkdtemp(prefix="http-gateway-", dir=base))
    print("CPU HTTP fixture evidence: " + str(RUN_ROOT), file=sys.stderr)
    result = unittest.TextTestRunner(verbosity=2).run(selected)
    sys.exit(0 if result.wasSuccessful() else 1)
