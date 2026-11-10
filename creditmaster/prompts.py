"""Prompt builders. Training, evaluation, inference and the Ollama Modelfile must all use these."""

from .policy import format_context
from .rules import (clean_text, integer, money, percent, translate_home,
                    translate_purpose, translate_term)

# Exact v1 system prompt (legacy_v1/generate_opinions_v1.py) — used to evaluate the legacy model.
SYSTEM_PROMPT_V1 = (
    "You are CreditMaster, an expert automated credit risk analyst. "
    "Analyze the applicant's financial and risk profile according to institutional credit policies. "
    "Output an objective, structured credit opinion in JSON format containing recommendation, risk_level, "
    "strengths, risk_factors, and executive opinion."
)

_SCHEMA_INSTRUCTIONS = (
    "Output a single JSON object with exactly these keys: "
    "recommendation (one of: Approve, Approve with Conditions, Suspend, Deny), "
    "risk_level (one of: Low, Low to Moderate, Moderate, High), "
    "authority_level (one of: Credit analyst, Credit coordination, Credit management, Credit committee), "
    "strengths (list of strings), risk_factors (list of strings), mitigants (list of strings), "
    "policy_references (list of policy sections, e.g. \"§4 Repayment Capacity and DTI\"), "
    "and opinion (executive credit opinion text). Do not invent data that is not provided."
)

SYSTEM_PROMPT_V2_RAG = (
    "You are CreditMaster, an expert automated credit risk analyst. "
    "Analyze the applicant's financial and risk profile strictly according to the institutional "
    "credit policy excerpts provided in the context, and cite the sections you apply. "
    + _SCHEMA_INSTRUCTIONS
)

SYSTEM_PROMPT_V2_NORAG = (
    "You are CreditMaster, an expert automated credit risk analyst. "
    "Analyze the applicant's financial and risk profile according to institutional credit policies, "
    "and cite the policy sections you apply. "
    + _SCHEMA_INSTRUCTIONS
)


def build_profile_text(profile):
    """Same field layout as the v1 training prompt."""
    return (
        "Prepare an objective credit opinion based on the financial and risk profile below.\n"
        f"Requested amount: {money(profile.get('loan_amnt'))}\n"
        f"Term: {translate_term(profile.get('term'))}\n"
        f"Annual income: {money(profile.get('annual_inc'))}\n"
        f"Purpose: {translate_purpose(profile.get('purpose'))}\n"
        f"Housing status: {translate_home(profile.get('home_ownership'))}\n"
        f"Employment length: {clean_text(profile.get('emp_length'))}\n"
        f"DTI: {percent(profile.get('dti'))}\n"
        f"Minimum FICO: {integer(profile.get('fico_range_low'))}\n"
        f"Delinquencies in 2 years: {integer(profile.get('delinq_2yrs'))}\n"
        f"Inquiries in 6 months: {integer(profile.get('inq_last_6mths'))}\n"
        f"Revolving utilization: {percent(profile.get('revol_util'))}\n"
        f"Open accounts: {integer(profile.get('open_acc'))}\n"
        f"Public records: {integer(profile.get('pub_rec'))}"
    )


def build_messages(profile, mode="rag", chunks=None):
    """mode: 'rag' (v2 + context), 'norag' (v2 without context), 'v1' (legacy prompt)."""
    profile_text = build_profile_text(profile)
    if mode == "v1":
        return [{"role": "system", "content": SYSTEM_PROMPT_V1},
                {"role": "user", "content": profile_text}]
    if mode == "norag":
        return [{"role": "system", "content": SYSTEM_PROMPT_V2_NORAG},
                {"role": "user", "content": profile_text}]
    if mode == "rag":
        if chunks is None:
            raise ValueError("mode='rag' requires retrieved chunks")
        user = format_context(chunks) + "\n\n### Applicant profile\n" + profile_text
        return [{"role": "system", "content": SYSTEM_PROMPT_V2_RAG},
                {"role": "user", "content": user}]
    raise ValueError(f"Unknown prompt mode: {mode}")
