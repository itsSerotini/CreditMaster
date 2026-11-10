"""Output schema (v2) and validation helpers."""

import json
import re

from .rules import AUTHORITY_LEVELS, RECOMMENDATIONS, RISK_LEVELS

OUTPUT_KEYS = [
    "recommendation", "risk_level", "authority_level", "strengths",
    "risk_factors", "mitigants", "policy_references", "opinion",
]
LIST_KEYS = ["strengths", "risk_factors", "mitigants", "policy_references"]

# v1 model labels -> v2 labels (used when evaluating the legacy fine-tuned model).
V1_LABEL_MAP = {
    "Conditional Approval": "Approve with Conditions",
    "Very High": "High",
}


def parse_json_output(text):
    """Parse model output; tolerates surrounding text by extracting the first {...} block."""
    if text is None:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def schema_errors(obj, required_keys=OUTPUT_KEYS):
    """Return a list of schema violations (empty list == valid)."""
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    errors = [f"missing key: {k}" for k in required_keys if k not in obj]
    if "recommendation" in obj and obj["recommendation"] not in RECOMMENDATIONS:
        errors.append(f"invalid recommendation: {obj['recommendation']!r}")
    if "risk_level" in obj and obj["risk_level"] not in RISK_LEVELS:
        errors.append(f"invalid risk_level: {obj['risk_level']!r}")
    if "authority_level" in obj and obj["authority_level"] not in AUTHORITY_LEVELS:
        errors.append(f"invalid authority_level: {obj['authority_level']!r}")
    for key in LIST_KEYS:
        if key in obj and not isinstance(obj[key], list):
            errors.append(f"{key} is not a list")
    if "opinion" in obj and not isinstance(obj["opinion"], str):
        errors.append("opinion is not a string")
    return errors


def section_numbers(references):
    """['§4 Repayment Capacity and DTI', '§11', 'Section 13'] -> {4, 11, 13}."""
    numbers = set()
    for ref in references or []:
        for match in re.findall(r"(?:§|section\s*)(\d+)", str(ref), flags=re.IGNORECASE):
            numbers.add(int(match))
    return numbers
