"""Load the credit policy markdown and split it into one chunk per "## N. Title" section."""

import hashlib
import re
from pathlib import Path

from .rules import POLICY_SECTIONS

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "credit_policy_en.md"

# Decision-framework sections injected in every prompt (not subject to retrieval).
PINNED_SECTIONS = (11, 12, 13)

_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")


def load_policy_chunks(path=DEFAULT_POLICY_PATH):
    """Return [{'id': 4, 'title': 'Repayment Capacity and DTI', 'ref': '§4 ...', 'text': body}, ...]."""
    text = Path(path).read_text(encoding="utf-8")
    chunks, current = [], None
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            if current:
                chunks.append(current)
            number, title = int(match.group(1)), match.group(2)
            current = {"id": number, "title": title, "ref": f"§{number} {title}", "lines": []}
        elif current is not None:
            current["lines"].append(line)
    if current:
        chunks.append(current)
    for chunk in chunks:
        chunk["text"] = "\n".join(chunk.pop("lines")).strip()
    _check_titles(chunks)
    return chunks


def policy_sha256(path=DEFAULT_POLICY_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_titles(chunks):
    """Fail fast if the policy headings drift from the references the rule engine emits."""
    by_id = {c["id"]: c["title"] for c in chunks}
    for number, title in POLICY_SECTIONS.items():
        if by_id.get(number) != title:
            raise ValueError(
                f"Policy section §{number} title mismatch: rules.py has {title!r}, "
                f"policy has {by_id.get(number)!r}. Update creditmaster/rules.py:POLICY_SECTIONS."
            )


def format_context(chunks):
    """Render retrieved chunks as the context block placed in the user prompt."""
    blocks = [f"[{c['ref']}]\n{c['text']}" for c in chunks]
    return "### Retrieved credit policy context\n\n" + "\n\n".join(blocks)
