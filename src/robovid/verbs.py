"""A fixed verb vocabulary for manipulation steps.

Free-text verbs from a VLM drift ("point to", "point_to", "point at", "hold up"),
which fragments the label space a policy is trained on. Every step's verb is
mapped onto this closed set; the model's original wording is kept in
`Step.raw_verb` whenever it differs, so nothing is lost.
"""
from __future__ import annotations

import re

VERBS: tuple[str, ...] = (
    "reach", "grasp", "release", "lift", "place", "hold", "move",
    "push", "pull", "press", "turn", "open", "close",
    "insert", "remove", "attach", "detach", "tighten", "loosen",
    "pour", "scoop", "stir", "mix", "cut", "wipe", "shake", "flip", "fold",
    "touch", "point", "wait", "other",
)
_VERB_SET = frozenset(VERBS)

SYNONYMS: dict[str, str] = {
    # grasp
    "grab": "grasp", "grip": "grasp", "take": "grasp", "pick": "grasp",
    "pick up": "grasp", "catch": "grasp", "seize": "grasp",
    # release
    "let go": "release", "let go of": "release", "drop": "release", "unhand": "release",
    # lift / place
    "raise": "lift", "lift up": "lift", "elevate": "lift",
    "put": "place", "put down": "place", "set": "place", "set down": "place",
    "lay": "place", "lay down": "place", "position": "place", "rest": "place",
    # hold / move
    "hold up": "hold", "show": "hold", "present": "hold", "display": "hold",
    "carry": "move", "slide": "move", "transfer": "move", "bring": "move",
    "shift": "move", "drag": "move", "lower": "move", "adjust": "move",
    "align": "place",
    "withdraw": "remove", "separate": "detach", "swirl": "shake",
    # push / pull / press
    "shove": "push", "nudge": "push",
    "tug": "pull", "yank": "pull",
    "tap": "press", "click": "press", "push down": "press", "press down": "press",
    "squeeze": "press",
    # turn
    "rotate": "turn", "twist": "turn", "spin": "turn", "crank": "turn",
    # insert / remove / attach / detach
    "put in": "insert", "plug in": "insert", "slot": "insert",
    "take out": "remove", "pull out": "remove", "pull off": "remove",
    "take off": "remove", "extract": "remove", "unplug": "remove",
    "connect": "attach", "mount": "attach", "install": "attach",
    "fasten": "attach", "fit": "attach",
    "disconnect": "detach", "unmount": "detach", "unclip": "detach",
    # tighten / loosen
    "screw": "tighten", "screw in": "tighten",
    "unscrew": "loosen", "undo": "loosen",
    # kitchen and cleaning
    "fill": "pour", "spoon": "scoop", "whisk": "mix", "blend": "mix",
    "chop": "cut", "slice": "cut", "dice": "cut", "trim": "cut",
    "clean": "wipe", "scrub": "wipe", "wipe down": "wipe",
    "flip over": "flip",
    # non-manipulation
    "point at": "point", "point to": "point", "point towards": "point",
    "indicate": "point", "gesture": "point", "gesture at": "point",
    "pause": "wait", "idle": "wait",
}


def _clean(verb: str) -> str:
    v = verb.strip().lower().replace("_", " ").replace("-", " ")
    v = re.sub(r"[^a-z ]+", "", v)
    return re.sub(r"\s+", " ", v).strip()


def _lookup(phrase: str) -> str | None:
    if phrase in _VERB_SET:
        return phrase
    return SYNONYMS.get(phrase)


def normalize_verb(verb: str | None) -> str:
    """Map a free-text verb onto VERBS, falling back to "other".

    Tries the whole phrase, then its first two words, then its first word, so
    "grasp the handle" -> "grasp" and "pick up mug" -> "grasp".
    """
    v = _clean(verb or "")
    if not v:
        return "other"
    words = v.split()
    for candidate in (v, " ".join(words[:2]), words[0]):
        hit = _lookup(candidate)
        if hit:
            return hit
    # simple inflections: "grasping", "grasped", "pours"
    for suffix in ("ing", "ed", "es", "s"):
        if words[0].endswith(suffix) and len(words[0]) > len(suffix) + 2:
            stem = words[0][: -len(suffix)]
            for c in (stem, stem + "e"):
                hit = _lookup(c)
                if hit:
                    return hit
    return "other"
