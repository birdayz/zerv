import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_session", ROOT/"tools/check_session.py")
check_session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_session)
split = check_session.split_reference


class OutputReferenceTests(unittest.TestCase):
    """The Python rendering of llama-server's output split used by tools/check_session.py.
    The cases mirror tests/session.zig (independent Zig implementation)."""

    def test_split_rules(self):
        self.assertEqual(split("\n\n  Hello world\n</think>\n\nAnswer\n", True), ("Hello world\n", "Answer\n"))
        self.assertEqual(split("a <b", True), ("a <b", ""))
        self.assertEqual(split("abc</thi", True), ("abc", ""))
        self.assertEqual(split("abc<", True), ("abc", ""))
        self.assertEqual(split("r <tool_call>x", True), ("r ", "<tool_call>x"))
        self.assertEqual(split(" \n\t\x0b\x0c\r</think>  c", True), (None, "c"))
        self.assertEqual(split("x</think>\ny</think> <tool_call> ", True), ("x", "y</think> <tool_call> "))
        self.assertEqual(split("\n\nx</think> y<tool_call>", False), (None, "x</think> y<tool_call>"))
        self.assertEqual(split(" \nThink <b> </thin </think>\n\n Answer </think> end\n", True), ("Think <b> </thin ", "Answer </think> end\n"))

    def test_render_drops_control_tokens_and_incomplete_tail(self):
        table = [b""] * 248320
        table[0], table[1], table[2] = b"Hi", b"\xe4\xb8", b"\x96!"
        table[248045] = b"<|im_start|>"
        table[248069] = b"</think>"
        self.assertEqual(check_session.render(table, [0, 248045, 248069, 1, 2]), "Hi</think>\u4e16!")
        self.assertEqual(check_session.render(table, [0, 1]), "Hi")
        self.assertEqual(check_session.render(table, [0, 2]), "Hi\ufffd!")


if __name__ == "__main__":
    unittest.main()
