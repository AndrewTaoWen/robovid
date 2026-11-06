"""The GPU service: it matches the labeler, labels a group of clips in one pass,
and one bad clip doesn't fail the rest. Modal and vLLM are replaced by stand-ins."""
import types
import unittest
from unittest import mock

from robovid import annotate
from vlm_helpers import QWEN, import_with_fake_modal


def outputs(*texts):
    return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t)]) for t in texts]


class TestGpuService(unittest.TestCase):
    def setUp(self):
        self.vlm, _ = import_with_fake_modal("robovid.modal_vlm")

    def test_names_match_the_labeler(self):
        self.assertEqual(self.vlm.APP_NAME, annotate.VLLM_APP_NAME)
        self.assertEqual(self.vlm.CLS_NAME, annotate.VLLM_CLS_NAME)
        self.assertEqual(self.vlm.VLMServer.__name__, annotate.VLLM_CLS_NAME)
        self.assertEqual(self.vlm.MODEL_NAME, QWEN)
        self.assertEqual(self.vlm.MAX_IMAGES, 16)  # same as the labeler's frame limit

    def test_container_asks_for_enough_memory_and_cpu(self):
        # Modal's small defaults got vLLM's engine killed on the first clip.
        self.assertGreaterEqual(self.vlm.MEMORY_MB, 32768)
        self.assertGreaterEqual(self.vlm.CPU, 4)

    def test_a_group_of_clips_is_one_model_call(self):
        llm = mock.Mock()
        llm.chat.return_value = outputs("a", "b")
        out = self.vlm.generate_batch(llm, "sp", QWEN,
                                      [{"messages": ["m1"]}, {"messages": ["m2"]}])
        llm.chat.assert_called_once_with([["m1"], ["m2"]], "sp", use_tqdm=False)
        self.assertEqual(out, [{"model": QWEN, "text": "a"}, {"model": QWEN, "text": "b"}])

    def test_one_bad_clip_does_not_fail_the_group(self):
        def chat(convs, sp, use_tqdm):
            if len(convs) > 1 or convs[0] == ["bad"]:
                raise ValueError("prompt too long")
            return outputs("ok")

        llm = mock.Mock()
        llm.chat.side_effect = chat
        out = self.vlm.generate_batch(llm, "sp", QWEN,
                                      [{"messages": ["good"]}, {"messages": ["bad"]}])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], {"model": QWEN, "text": "ok"})
        self.assertIn("prompt too long", out[1]["error"])

    def test_a_single_failing_clip_returns_an_error(self):
        llm = mock.Mock()
        llm.chat.side_effect = ValueError("out of memory")
        out = self.vlm.generate_batch(llm, "sp", QWEN, [{"messages": ["x"]}])
        self.assertEqual(llm.chat.call_count, 1)  # not retried a second time
        self.assertIn("out of memory", out[0]["error"])


if __name__ == "__main__":
    unittest.main()
