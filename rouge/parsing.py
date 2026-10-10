from __future__ import annotations

import json
import re

from .items import RELIC, Item

FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
DONE = re.compile(r"^\s*DONE\s*$", re.M)


def strip_thinking(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip()


def parse_json_response(text: str) -> dict:
    text = strip_thinking(text)
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def parse_action(text: str) -> tuple[str, bool]:

    text = strip_thinking(text)
    if text.startswith("{"):
        try:
            value = parse_json_response(text)
            source, done = value.get("python", ""), value.get("done", False)
            if isinstance(source, str) and isinstance(done, bool):
                return source, done
        except ValueError:
            pass
    blocks = [body for lang, body in FENCE.findall(text) if lang.lower() in ("", "python", "py", "python3")]
    outside = FENCE.sub("", text)
    done = bool(DONE.search(outside))
    if len(blocks) > 1:
        raise ValueError("Send exactly one ```python block per turn")
    if not blocks:
        if "```" in outside:
            raise ValueError("The code block was not closed; send shorter code in one ```python block")
        if not done:
            raise ValueError("Reply with one ```python block, or the single word DONE when outputs are ready")
        return "", True
    return blocks[0], done


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    return (slug if slug[:1].isalpha() else "r_" + slug)[:31] or "tool"


def parse_item(text: str) -> Item:

    text = strip_thinking(text)
    blocks = FENCE.findall(text)
    header, content = None, None
    for lang, body in blocks:
        if lang.lower() == "json" and header is None:
            header = json.loads(body)
        else:
            content = body.rstrip()
    if header is None:
        match = re.search(r"\{.*?\}", FENCE.sub("", text), re.S)
        candidate = match.group(0) if match else text
        try:
            header = json.loads(candidate)
        except json.JSONDecodeError:
            header = parse_json_response(text)
    if not isinstance(header, dict):
        raise ValueError("Header must be a JSON object")
    if "content" in header and str(header["content"]).strip():
        content = str(header["content"])
    if content is None:
        raise ValueError("Missing item content (put it in one fenced block after the JSON header)")
    slot = str(header.get("slot", "")).strip().lower()
    name = str(header.get("name", "")).strip() or "unnamed"
    if slot == RELIC:
        name = _slug(name)
    return Item.from_dict({"slot": slot, "name": name, "content": content,
                           "needs": header.get("needs", [])})
