import argparse
import csv
import json
import math
import random
from pathlib import Path

COLUMNS = [
    "loan_amnt",
    "term",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "purpose",
    "dti",
    "fico_range_low",
    "delinq_2yrs",
    "inq_last_6mths",
    "revol_util",
    "open_acc",
    "pub_rec",
    "loan_status",
]

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

def to_float(value):
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number

def clean_text(value):
    value = (value or "").strip()
    return value if value else "not provided"

def money(value):
    num = to_float(value)
    return "not provided" if num is None else f"US$ {num:,.2f}"

def percent(value):
    num = to_float(value)
    return "not provided" if num is None else f"{num:.2f}%"

def integer(value):
    num = to_float(value)
    return "not provided" if num is None else str(int(round(num)))

def translate_purpose(value):
    val = clean_text(value)
    return PURPOSES.get(val, val.replace("_", " "))

def translate_home(value):
    val = clean_text(value).upper()
    return HOME_OWNERSHIP.get(val, val.lower())

def translate_term(value):
    val = clean_text(value)
    if "36" in val:
        return "36 months"
    if "60" in val:
        return "60 months"
    return val

def evaluate_credit_profile(row):
    fico = to_float(row.get("fico_range_low"))
    dti = to_float(row.get("dti"))
    delinq = to_float(row.get("delinq_2yrs"))
    inq = to_float(row.get("inq_last_6mths"))
    revol = to_float(row.get("revol_util"))
    income = to_float(row.get("annual_inc"))
    loan = to_float(row.get("loan_amnt"))
    term = clean_text(row.get("term"))
    pub_rec = to_float(row.get("pub_rec"))
    emp = clean_text(row.get("emp_length"))

    score_points = 0
    strengths = []
    risk_factors = []

    # 1. FICO Score
    if fico is None:
        risk_factors.append("FICO score not provided")
        score_points -= 2
    elif fico >= 760:
        strengths.append(f"Very strong FICO score ({int(fico)})")
        score_points += 3
    elif fico >= 720:
        strengths.append(f"Strong FICO score ({int(fico)})")
        score_points += 2
    elif fico >= 680:
        strengths.append(f"Adequate FICO score ({int(fico)})")
        score_points += 1
    elif fico >= 640:
        risk_factors.append(f"Moderate/borderline FICO score ({int(fico)})")
        score_points -= 1
    else:
        risk_factors.append(f"Weak FICO score ({int(fico)}) below standard threshold")
        score_points -= 3

    # 2. DTI (Debt-to-Income)
    if dti is None:
        risk_factors.append("DTI ratio not reported")
        score_points -= 1
    elif dti <= 15:
        strengths.append(f"Controlled DTI ratio ({dti:.2f}%)")
        score_points += 2
    elif dti <= 30:
        strengths.append(f"Acceptable DTI ratio ({dti:.2f}%)")
        score_points += 1
    elif dti <= 40:
        risk_factors.append(f"Elevated DTI ratio ({dti:.2f}%)")
        score_points -= 2
    else:
        risk_factors.append(f"High debt burden with DTI above 40% ({dti:.2f}%)")
        score_points -= 4

    # 3. Delinquencies in past 2 years
    if delinq is not None and delinq > 0:
        risk_factors.append(f"Recent delinquency history ({int(delinq)} late payment events in 24 months)")
        score_points -= (3 + int(delinq))
    else:
        strengths.append("Clean recent credit history with no delinquencies in 2 years")
        score_points += 2

    # 4. Inquiries in last 6 months
    if inq is not None and inq >= 3:
        risk_factors.append(f"High frequency of recent credit inquiries ({int(inq)} inquiries)")
        score_points -= 2
    elif inq == 0:
        strengths.append("No credit inquiries in the last 6 months")
        score_points += 1

    # 5. Revolving Credit Utilization
    if revol is not None:
        if revol >= 80:
            risk_factors.append(f"High revolving credit utilization ({revol:.2f}%), indicating credit dependence")
            score_points -= 3
        elif revol <= 35:
            strengths.append(f"Low revolving utilization ({revol:.2f}%) demonstrating disciplined liquidity")
            score_points += 2

    # 6. Loan-to-Income Exposure
    if income and loan and income > 0:
        ratio = loan / income
        if ratio > 0.45:
            risk_factors.append(f"Requested loan represents {ratio*100:.1f}% of annual income (high leverage)")
            score_points -= 3
        elif ratio <= 0.20:
            strengths.append(f"Requested amount is conservative ({ratio*100:.1f}% of annual income)")
            score_points += 2

    # 7. Public Records & Employment
    if pub_rec is not None and pub_rec > 0:
        risk_factors.append(f"Presence of derogatory public records ({int(pub_rec)})")
        score_points -= 3

    if "10+" in emp or "9" in emp or "8" in emp:
        strengths.append(f"Stable employment tenure ({emp})")
        score_points += 1

    if "60" in term:
        risk_factors.append("Extended term (60 months) increases risk exposure over time")
        score_points -= 1

    # Final Decision Synthesis
    if score_points >= 4 and not any("Weak FICO" in rf or "High debt burden" in rf for rf in risk_factors):
        decision = "Approve"
        risk_level = "Low" if score_points >= 7 else "Low to Moderate"
        conclusion = "The applicant demonstrates robust repayment capacity and favorable credit history, meeting policy requirements for standard approval."
    elif score_points >= 0:
        decision = "Conditional Approval"
        risk_level = "Moderate"
        conclusion = "The proposal presents acceptable fundamentals but exhibits moderate risk points. Approval is recommended subject to standard documentation validation or exposure monitoring."
    else:
        decision = "Deny"
        risk_level = "High" if score_points >= -4 else "Very High"
        conclusion = "The credit application exceeds policy risk tolerance due to adverse credit indicators or overextended leverage, warranting a decline."

    return decision, risk_level, strengths[:3], risk_factors[:3], conclusion

def build_input(row):
    return (
        "Prepare an objective credit opinion based on the financial and risk profile below.\n"
        f"Requested amount: {money(row.get('loan_amnt'))}\n"
        f"Term: {translate_term(row.get('term'))}\n"
        f"Annual income: {money(row.get('annual_inc'))}\n"
        f"Purpose: {translate_purpose(row.get('purpose'))}\n"
        f"Housing status: {translate_home(row.get('home_ownership'))}\n"
        f"Employment length: {clean_text(row.get('emp_length'))}\n"
        f"DTI: {percent(row.get('dti'))}\n"
        f"Minimum FICO: {integer(row.get('fico_range_low'))}\n"
        f"Delinquencies in 2 years: {integer(row.get('delinq_2yrs'))}\n"
        f"Inquiries in 6 months: {integer(row.get('inq_last_6mths'))}\n"
        f"Revolving utilization: {percent(row.get('revol_util'))}\n"
        f"Open accounts: {integer(row.get('open_acc'))}\n"
        f"Public records: {integer(row.get('pub_rec'))}"
    )

def build_opinion(row):
    decision, risk_level, strengths, risk_factors, conclusion = evaluate_credit_profile(row)
    
    strengths_text = "; ".join(strengths) if strengths else "No relevant positive factors identified in the examined fields."
    factors_text = "; ".join(risk_factors) if risk_factors else "No critical risk flags detected in primary indicators."

    fico_str = integer(row.get('fico_range_low'))
    dti_str = percent(row.get('dti'))
    inc_str = money(row.get('annual_inc'))
    loan_str = money(row.get('loan_amnt'))
    purpose_str = translate_purpose(row.get('purpose'))

    opinion_text = (
        f"Recommendation: {decision}. Risk Level: {risk_level}. "
        f"The applicant presents a FICO score of {fico_str}, DTI of {dti_str}, "
        f"annual income of {inc_str}, and requested loan of {loan_str} for {purpose_str}. "
        f"Strengths: {strengths_text}. Risk factors: {factors_text}. {conclusion}"
    )

    return {
        "recommendation": decision,
        "risk_level": risk_level,
        "strengths": strengths,
        "risk_factors": risk_factors,
        "opinion": opinion_text,
    }

def sample_rows(input_path, total_samples, seed=42):
    rng = random.Random(seed)
    
    with input_path.open(newline="", encoding="utf-8", errors="replace") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [column for column in COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in CSV: {', '.join(missing)}")

        all_rows = []
        for i, row in enumerate(reader):
            if row.get("annual_inc") and row.get("loan_amnt") and row.get("fico_range_low"):
                all_rows.append(row)
            if len(all_rows) >= total_samples * 5:
                break

    rng.shuffle(all_rows)
    
    target_per_class = total_samples // 3
    approved, conditional, denied = [], [], []
    for row in all_rows:
        dec, _, _, _, _ = evaluate_credit_profile(row)
        if dec == "Approve" and len(approved) < target_per_class:
            approved.append(row)
        elif dec == "Conditional Approval" and len(conditional) < target_per_class:
            conditional.append(row)
        elif dec == "Deny" and len(denied) < target_per_class:
            denied.append(row)
        
        if len(approved) >= target_per_class and len(conditional) >= target_per_class and len(denied) >= target_per_class:
            break

    sampled = approved + conditional + denied
    rng.shuffle(sampled)
    return sampled

def write_jsonl(rows, output_path):
    system_prompt = (
        "You are CreditMaster, an expert automated credit risk analyst. "
        "Analyze the applicant's financial and risk profile according to institutional credit policies. "
        "Output an objective, structured credit opinion in JSON format containing recommendation, risk_level, "
        "strengths, risk_factors, and executive opinion."
    )
    with output_path.open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            opinion_data = build_opinion(row)
            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_input(row)},
                    {"role": "assistant", "content": json.dumps(opinion_data, ensure_ascii=False)},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_csv(rows, output_path):
    fields = [
        "recommendation",
        "risk_level",
        "input",
        "opinion",
        *[c for c in COLUMNS if c != "loan_status"],
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            op = build_opinion(row)
            writer.writerow({
                "recommendation": op["recommendation"],
                "risk_level": op["risk_level"],
                "input": build_input(row),
                "opinion": op["opinion"],
                **{c: row.get(c, "") for c in COLUMNS if c != "loan_status"},
            })

def main():
    parser = argparse.ArgumentParser(description="Generate CreditMaster dataset for fine-tuning.")
    parser.add_argument("--input", default="/home/serotini/TCC/Dataset/accepted_2007_to_2018Q4.csv", help="Input loans CSV path.")
    parser.add_argument("--total", type=int, default=2000, help="Total samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--jsonl", default="/home/serotini/TCC/Dataset/credit_opinions_2000.jsonl", help="Output JSONL.")
    parser.add_argument("--csv", default="/home/serotini/TCC/Dataset/credit_opinions_audit.csv", help="Audit CSV.")
    args = parser.parse_args()

    print(f"Sampling and generating {args.total} credit analysis opinions...")
    input_file = Path(args.input)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {args.input} not found.")

    rows = sample_rows(input_file, args.total, args.seed)
    write_jsonl(rows, Path(args.jsonl))
    write_csv(rows, Path(args.csv))
    print(f"Successfully generated {len(rows)} samples.")
    print(f"JSONL saved to: {args.jsonl}")
    print(f"Audit CSV saved to: {args.csv}")

if __name__ == "__main__":
    main()
