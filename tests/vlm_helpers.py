"""Small helpers shared by the tests for labeling clips with different models.

None of these tests need a GPU, the network, or a Modal account.
"""
import json

from robovid.frames import Clip, Frame

QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"


def make_clip(n=20, start=10.0):
    """A clip with `n` one-second frames whose "JPEG" bytes are just labels."""
    frames = [Frame(t=start + i, jpeg=f"jpeg-{i}".encode()) for i in range(n)]
    return Clip(start, start + n - 1, frames)


def model_reply(**over):
    """A typical model answer: valid label JSON wrapped in a Markdown code block.

    It also includes a made-up source_url, so tests can check that we ignore it.
    """
    body = {
        "task": "loosening a bolt", "viewpoint": "egocentric", "has_human_hands": True,
        "usable_for_robotics": True,
        "steps": [{"start_s": 11, "end_s": 13, "verb": "unscrew", "objects": ["bolt"],
                   "hand": "right"},
                  {"start_s": 13, "end_s": 15, "verb": "grasp", "objects": ["wrench"]}],
        "source_url": "made-up-by-model",
    }
    body.update(over)
    return "```json\n" + json.dumps(body) + "\n```"


def fake_modal():
    """A stand-in for the `modal` package, with just enough for our code to import.

    Its decorators hand back the plain function or class, so tests can call the
    code directly. The rest (Volume, Secret, Cls, ...) are mocks we can inspect.
    """
    import types
    from unittest import mock

    m = types.ModuleType("modal")

    class _Image:
        def __getattr__(self, name):  # lets .pip_install(...).env(...) chain
            return lambda *a, **k: self

    class App:
        def __init__(self, name, **kw):
            self.name = name

        def function(self, **kw):
            return lambda f: f

        def cls(self, **kw):
            return lambda c: c

        def local_entrypoint(self, **kw):
            return lambda f: f

    m.App = App
    m.Image = types.SimpleNamespace(debian_slim=lambda *a, **k: _Image())
    m.Volume, m.Secret, m.Retries, m.Cls = mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    m.enter = m.method = lambda *a, **k: (lambda f: f)
    m.batched = lambda *a, **k: (lambda f: f)
    return m


def import_with_fake_modal(module, env=None):
    """Import `module` from scratch with the stand-in Modal; returns (module, stand-in)."""
    import importlib
    import os
    import sys
    from unittest import mock

    stub = fake_modal()
    with mock.patch.dict(sys.modules, {"modal": stub}), \
            mock.patch.dict(os.environ, env or {}):
        sys.modules.pop(module, None)
        mod = importlib.import_module(module)
    return mod, stub
