"""CreditMaster v2 rule engine — single source of truth for gold labels.

Aligned with Dataset/credit_policy_en.md. Differences from the v1 engine
(legacy_v1/generate_opinions_v1.py), which is kept untouched for reproducibility:

* DTI > 30% blocks automatic approval (policy §4).
* Requested amount > 45% of annual income blocks automatic approval (§8).
* FICO < 660 is a relevant negative factor (§3); approval requires FICO >= 680 (§11).
* Missing essential data (FICO, income, DTI) -> "Suspend" (§11).
* Missing secondary data is never presumed positive (§2, §10).
* Recent delinquency combined with low FICO, high DTI or long term -> Deny (§5).
* New outputs: authority_level (§13), mitigants (§12), policy_references.
* Labels follow §14 wording: Approve / Approve with Conditions / Suspend / Deny,
  risk levels Low / Low to Moderate / Moderate / High.

Known gap: public records (pub_rec) are scored as a risk factor but the policy
has no section about them, so they carry no policy reference.
"""

import math
import re

PROFILE_FIELDS = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc", "purpose",
    "dti", "fico_range_low", "delinq_2yrs", "inq_last_6mths", "revol_util",
    "open_acc", "pub_rec",
]

ESSENTIAL_FIELDS = {"fico_range_low": "FICO score", "annual_inc": "annual income", "dti": "DTI"}

RECOMMENDATIONS = ["Approve", "Approve with Conditions", "Suspend", "Deny"]
RISK_LEVELS = ["Low", "Low to Moderate", "Moderate", "High"]
AUTHORITY_LEVELS = ["Credit analyst", "Credit coordination", "Credit management", "Credit committee"]

# Titles must match the "## N. Title" headings of credit_policy_en.md (checked in policy.py).
POLICY_SECTIONS = {
    3: "FICO Score Classification",
    4: "Repayment Capacity and DTI",
    5: "Delinquency History",
    6: "Recent Credit Inquiries",
    7: "Revolving Credit Utilization",
    8: "Requested Amount Relative to Income",
    9: "Loan Term",
    10: "Purpose, Housing Status, and Employment Length",
    11: "Recommendation Matrix",
    12: "Acceptable Mitigants",
    13: "Decision Authority Levels",
}

# Exact wording of policy §12.
MITIGANT_AMOUNT = "Reduction of the requested amount"
MITIGANT_TERM = "Reduction of the term"
MITIGANT_INCOME = "Additional evidence of recurring income"
MITIGANT_SETTLE = "Prior settlement of short-term debts"
MITIGANT_COLLATERAL = "Additional collateral, when applicable"
MITIGANT_COOBLIGOR = "Co-obligor with a stronger credit profile"

PURPOSES = {
    "car": "vehicle purchase",
    "credit_card": "credit card refinancing",
    "debt_consolidation": "debt consolidation",
    "educational": "education",
    "home_improvement": "home improvement",
    "house": "housing",
    "major_purchase": "major purchase",
    "medical": "medical expenses",
    "moving": "moving expenses",
    "other": "other purpose",
    "renewable_energy": "renewable energy",
    "small_business": "small business",
    "vacation": "travel",
    "wedding": "wedding",
}

HOME_OWNERSHIP = {
    "ANY": "not provided",
    "MORTGAGE": "mortgaged home",
    "NONE": "not provided",
    "OTHER": "other",
    "OWN": "owned home",
    "RENT": "rented home",
}

NOT_PROVIDED = "not provided"


def section_ref(number):
    return f"§{number} {POLICY_SECTIONS[number]}"


# ---------------------------------------------------------------------------
# Parsing / formatting helpers (same output format as v1 so prompts stay familiar)
# ---------------------------------------------------------------------------
def to_float(value):
    if value is None:
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def clean_text(value):
    value = (value or "").strip()
    return value if value else NOT_PROVIDED


def money(value):
    num = to_float(value)
    return NOT_PROVIDED if num is None else f"US$ {num:,.2f}"


def percent(value):
    num = to_float(value)
    return NOT_PROVIDED if num is None else f"{num:.2f}%"


def integer(value):
    num = to_float(value)
    return NOT_PROVIDED if num is None else str(int(round(num)))


def translate_purpose(value):
    val = clean_text(value)
    return PURPOSES.get(val, val.replace("_", " "))


def translate_home(value):
    val = clean_text(value)
    return HOME_OWNERSHIP.get(val.upper(), val.lower())


def translate_term(value):
    val = clean_text(value)
    if "36" in val:
        return "36 months"
    if "60" in val:
        return "60 months"
    return val


def parse_emp_years(value):
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, 'n/a'/'' -> None."""
    text = clean_text(value).lower()
    if text in (NOT_PROVIDED, "n/a"):
        return None
    if "<" in text:
        return 0
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
def assess(profile):
    """Apply the v2 policy rules to a raw profile dict (LendingClub column names).

    Returns a dict with the gold output fields plus internal diagnostics
    (`score`, `flags`, `risk_sections`, `missing_essential`).
    """
    fico = to_float(profile.get("fico_range_low"))
    dti = to_float(profile.get("dti"))
    delinq = to_float(profile.get("delinq_2yrs"))
    inq = to_float(profile.get("inq_last_6mths"))
    revol = to_float(profile.get("revol_util"))
    income = to_float(profile.get("annual_inc"))
    loan = to_float(profile.get("loan_amnt"))
    pub_rec = to_float(profile.get("pub_rec"))
    term = translate_term(profile.get("term"))
    emp_years = parse_emp_years(profile.get("emp_length"))
    purpose = clean_text(profile.get("purpose"))

    score = 0
    strengths, risks = [], []
    risk_sections = set()
    flags = set()

    def risk(text, section=None, points=0):
        nonlocal score
        risks.append(text)
        score += points
        if section is not None:
            risk_sections.add(section)

    def strength(text, points):
        nonlocal score
        strengths.append(text)
        score += points

    missing_essential = [label for field, label in ESSENTIAL_FIELDS.items()
                         if to_float(profile.get(field)) is None]

    # §3 FICO
    if fico is None:
        risk("FICO score not provided; treated as a risk factor until documented evidence is provided", 3)
    elif fico >= 760:
        strength(f"Very strong FICO score ({fico:.0f})", 3)
    elif fico >= 720:
        strength(f"Strong FICO score ({fico:.0f})", 2)
    elif fico >= 680:
        strength(f"Adequate FICO score ({fico:.0f})", 1)
    elif fico >= 660:
        flags.add("fico_moderate")
        risk(f"Moderate FICO score ({fico:.0f}) requiring caution and objective compensating factors", 3, -1)
    elif fico >= 640:
        flags.update({"fico_moderate", "fico_below_660"})
        risk(f"Moderate FICO score ({fico:.0f}) below 660, a relevant negative factor", 3, -2)
    else:
        flags.update({"fico_weak", "fico_below_660"})
        risk(f"Weak FICO score ({fico:.0f}), a restrictive indicator", 3, -3)

    # §4 DTI
    if dti is None:
        risk("DTI not reported; treated as a restriction until income and debts are validated", 4)
    elif dti <= 15:
        strength(f"Controlled DTI ({dti:.2f}%)", 2)
    elif dti <= 30:
        strength(f"Acceptable DTI ({dti:.2f}%)", 1)
    elif dti <= 40:
        flags.add("dti_high")
        risk(f"High DTI ({dti:.2f}%) above the 30% limit; automatic approval not permitted", 4, -2)
    else:
        flags.update({"dti_high", "dti_severe"})
        risk(f"Very high DTI ({dti:.2f}%), indicating excessive debt burden", 4, -4)

    # §5 Delinquency
    if delinq is None:
        flags.add("missing_secondary")
        risk("Delinquency history not provided; treated as incomplete information", 5, -1)
    elif delinq > 0:
        flags.add("delinq")
        n = int(delinq)
        risk(f"Recent delinquency history ({n} event{'s' if n != 1 else ''} in the last 24 months), "
             "a priority risk factor", 5, -(3 + n))
    else:
        strength("No delinquencies in the last two years", 2)

    # §6 Inquiries
    if inq is None:
        flags.add("missing_secondary")
        risk("Recent credit inquiries not provided; treated as incomplete information", 6, -1)
    elif inq >= 3:
        flags.add("inq")
        risk(f"{int(inq)} credit inquiries in the last 6 months, indicating recent credit pressure", 6, -2)

    # §7 Revolving utilization
    if revol is None:
        flags.add("missing_secondary")
        risk("Revolving utilization not provided; treated as incomplete information", 7, -1)
    elif revol >= 80:
        flags.add("revol_high")
        risk(f"High revolving utilization ({revol:.2f}%), indicating possible dependence on short-term credit", 7, -3)
    elif revol <= 35:
        strength(f"Low revolving utilization ({revol:.2f}%)", 2)

    # §8 Amount relative to income
    ratio = loan / income if (income and loan and income > 0) else None
    if ratio is not None:
        if ratio > 0.45:
            flags.add("lti_high")
            risk(f"Requested amount is {ratio * 100:.1f}% of annual income, above the 45% limit", 8, -3)
        elif ratio <= 0.20:
            strength(f"Requested amount is {ratio * 100:.1f}% of annual income, proportional to income", 2)

    # §9 Term
    if term == "60 months":
        flags.add("long_term")
        risk("60-month term increases risk exposure", 9, -1)

    # §10 Purpose, housing, employment (complementary signals)
    if emp_years is None:
        risk("Employment length not provided; not treated as a stability signal", 10)
    elif emp_years >= 8:
        strength(f"Stable employment tenure ({clean_text(profile.get('emp_length'))})", 1)
    if purpose in ("debt_consolidation", "credit_card") and flags & {"revol_high", "inq", "dti_high"}:
        risk(f"Purpose ({translate_purpose(purpose)}) combined with credit stress signals requires additional attention", 10, -1)

    # Public records — not covered by the policy (no section reference).
    if pub_rec is not None and pub_rec > 0:
        flags.add("pub_rec")
        risk(f"Derogatory public records ({int(pub_rec)})", None, -3)

    # -------------------------------------------------------------------
    # Decision (§5, §11)
    # -------------------------------------------------------------------
    severe = len(flags & {"fico_below_660", "delinq", "lti_high", "dti_severe", "pub_rec"})
    combo_deny = "delinq" in flags and bool(flags & {"fico_below_660", "dti_high", "long_term"})

    if missing_essential:
        recommendation, risk_level = "Suspend", "High"
    else:
        approve_ok = (
            fico is not None and fico >= 680
            and "dti_high" not in flags
            and delinq == 0
            and "lti_high" not in flags
            and "pub_rec" not in flags
            and score >= 4
        )
        if approve_ok:
            recommendation = "Approve"
            risk_level = "Low" if score >= 7 else "Low to Moderate"
        elif score < 0 or combo_deny:
            recommendation, risk_level = "Deny", "High"
        else:
            recommendation, risk_level = "Approve with Conditions", "Moderate"

    # §13 Authority level
    level = 0
    if flags & {"dti_high", "long_term", "inq"}:
        level = 1
    if flags & {"fico_below_660", "delinq", "lti_high"}:
        level = 2
    if severe >= 2 or missing_essential:
        level = 3
    authority = AUTHORITY_LEVELS[level]

    # §12 Mitigants
    mitigants = []
    if recommendation in ("Approve with Conditions", "Deny"):
        if "lti_high" in flags:
            mitigants.append(MITIGANT_AMOUNT)
        if "long_term" in flags:
            mitigants.append(MITIGANT_TERM)
        if "dti_high" in flags:
            mitigants.append(MITIGANT_INCOME)
        if flags & {"revol_high", "inq"}:
            mitigants.append(MITIGANT_SETTLE)
        if flags & {"fico_below_660", "delinq", "pub_rec"}:
            mitigants.append(MITIGANT_COOBLIGOR)
        if level >= 2:
            mitigants.append(MITIGANT_COLLATERAL)
        if not mitigants:
            mitigants.append(MITIGANT_INCOME)

    sections = set(risk_sections) | {11}
    if mitigants:
        sections.add(12)
    if level > 0:
        sections.add(13)
    policy_references = [section_ref(n) for n in sorted(sections)]

    result = {
        "recommendation": recommendation,
        "risk_level": risk_level,
        "authority_level": authority,
        "strengths": strengths,
        "risk_factors": risks,
        "mitigants": mitigants,
        "policy_references": policy_references,
    }
    result["opinion"] = _build_opinion(profile, result, flags, ratio, missing_essential, combo_deny)

    diagnostics = {
        "score": score,
        "flags": sorted(flags),
        "risk_sections": sorted(risk_sections),
        "missing_essential": missing_essential,
    }
    return result, diagnostics


def _build_opinion(profile, result, flags, ratio, missing_essential, combo_deny):
    rec = result["recommendation"]
    ratio_txt = f" ({ratio * 100:.1f}% of annual income)" if ratio is not None else ""
    parts = [
        f"Recommendation: {rec}. Risk level: {result['risk_level']}. "
        f"Decision authority: {result['authority_level']}.",
        f"The applicant presents a FICO score of {integer(profile.get('fico_range_low'))}, "
        f"DTI of {percent(profile.get('dti'))}, annual income of {money(profile.get('annual_inc'))}, "
        f"and requests {money(profile.get('loan_amnt'))}{ratio_txt} for "
        f"{translate_purpose(profile.get('purpose'))} over {translate_term(profile.get('term'))}.",
    ]
    if result["strengths"]:
        parts.append("Strengths: " + "; ".join(result["strengths"]) + ".")
    if result["risk_factors"]:
        parts.append("Risk factors: " + "; ".join(result["risk_factors"]) + ".")

    rationale = []
    if missing_essential:
        rationale.append(f"Essential information is missing ({', '.join(missing_essential)}); "
                         "per §11 the decision is suspended until data remediation.")
    else:
        if combo_deny:
            rationale.append("Per §5, recent delinquency combined with low score, high DTI or a long term "
                             "defaults to denial or strong mitigants.")
        elif "delinq" in flags:
            rationale.append("Per §5, recent delinquency is a priority risk factor.")
        if "fico_below_660" in flags:
            rationale.append("Per §3, a score below 660 is a relevant negative factor.")
        if "dti_high" in flags:
            rationale.append("Per §4, DTI above 30% precludes automatic approval.")
        if "lti_high" in flags:
            rationale.append("Per §8, a request above 45% of annual income requires higher-authority review, "
                             "amount reduction, additional collateral, or denial.")
        if "long_term" in flags:
            rationale.append("Per §9, the 60-month term requires consistent repayment capacity.")
        if "inq" in flags:
            rationale.append("Per §6, three or more recent inquiries reduce appetite for increased exposure.")
        if "revol_high" in flags:
            rationale.append("Per §7, utilization at or above 80% indicates possible dependence on short-term credit.")

    mitigants_txt = "; ".join(result["mitigants"])
    if rec == "Approve":
        rationale.append("Per §11, the profile meets the low-to-moderate risk conditions for approval with regular monitoring.")
    elif rec == "Approve with Conditions":
        rationale.append(f"Per §11 and §12, approval is recommended only with the following mitigants: {mitigants_txt}.")
    elif rec == "Deny":
        rationale.append(f"Per §11, the profile falls into the high-risk category and should be declined unless "
                         f"strong mitigants are provided ({mitigants_txt}).")
    if result["authority_level"] != AUTHORITY_LEVELS[0] and rec != "Suspend":
        rationale.append(f"Per §13, the decision requires {result['authority_level'].lower()} approval.")

    parts.extend(rationale)
    return " ".join(parts)
