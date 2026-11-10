"""Policy retriever for CreditMaster RAG.

Design
------
The applicant profile is structured (13 fixed fields), so embedding the raw
profile text would retrieve the same sections for every applicant. Instead,
`build_query()` expands the profile into natural-language descriptors of the
signals present (e.g. "DTI above 30%", "60 months long term"), using the
policy's own vocabulary. The query is then matched against the indicator
sections of the policy.

* Pinned sections (§11 Recommendation Matrix, §12 Mitigants, §13 Authority)
  are always injected: every decision needs them.
* The remaining sections are ranked by BM25 (default, no dependencies) or by
  cosine similarity of Ollama embeddings (`backend="ollama"`).

The SAME retriever configuration must be used for dataset generation and for
inference, otherwise the fine-tuned model sees a context distribution it was
not trained on.
"""

import json
import math
import re
import urllib.request
from collections import Counter

from .policy import PINNED_SECTIONS, load_policy_chunks
from .rules import clean_text, parse_emp_years, to_float, translate_term

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "with", "be", "is", "are",
    "must", "as", "at", "it", "its", "this", "that", "when", "from", "not", "no", "up", "than",
    "should", "may", "only", "any", "such", "which", "their", "other", "there", "has", "have",
}


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def build_query(profile):
    """Describe the profile's notable signals in policy vocabulary (descriptors joined by '. ')."""
    return ". ".join(build_descriptors(profile))


def build_descriptors(profile):
    """One descriptor per detected signal; their count drives the dynamic top-k."""
    terms = []
    fico = to_float(profile.get("fico_range_low"))
    dti = to_float(profile.get("dti"))
    delinq = to_float(profile.get("delinq_2yrs"))
    inq = to_float(profile.get("inq_last_6mths"))
    revol = to_float(profile.get("revol_util"))
    income = to_float(profile.get("annual_inc"))
    loan = to_float(profile.get("loan_amnt"))
    purpose = clean_text(profile.get("purpose"))

    missing = [name for name, val in (("score", fico), ("income", income), ("DTI", dti)) if val is None]
    if missing:
        terms.append("incomplete essential information missing " + " ".join(missing)
                     + " suspend decision until data remediation")

    if fico is None:
        terms.append("FICO score not provided no score provided")
    elif fico < 680:
        terms.append("moderate weak FICO score below 660 caution compensating factors restrictive")
    elif fico >= 720:
        terms.append("strong FICO score strength")

    if dti is None:
        terms.append("DTI not provided not measured income debts validated")
    elif dti > 30:
        terms.append("high DTI above 30% debt burden repayment capacity compensating factors")

    if delinq is None:
        terms.append("delinquency history missing data request evidence")
    elif delinq > 0:
        terms.append("delinquencies in the last two years recent delinquency history behavioral fragility")

    if inq is not None and inq >= 3:
        terms.append("three or more recent credit inquiries last 6 months liquidity demand")

    if revol is not None and revol >= 80:
        terms.append("high revolving utilization 80% dependence on short-term credit")

    if income and loan and income > 0 and loan / income > 0.45:
        terms.append("requested amount above 45% of annual income high exposure")

    if translate_term(profile.get("term")) == "60 months":
        terms.append("60 months long term loan term risk exposure")

    if purpose in ("debt_consolidation", "credit_card"):
        terms.append("debt consolidation credit card refinancing purpose additional attention")
    if parse_emp_years(profile.get("emp_length")) is None:
        terms.append("employment length missing stability signals")

    if not terms:
        terms.append("strong FICO score controlled DTI repayment capacity eligible for approval")
    return terms


class _BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in documents]
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query):
        q = tokenize(query)
        out = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            out.append(s)
        return out


def _ollama_embed(texts, model, host):
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/embed", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["embeddings"]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class PolicyRetriever:
    def __init__(self, policy_path=None, backend="bm25", min_k=2, max_k=8,
                 embed_model="nomic-embed-text", ollama_host="http://localhost:11434"):
        self.chunks = load_policy_chunks(policy_path) if policy_path else load_policy_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.pinned = [self.by_id[i] for i in PINNED_SECTIONS]
        self.candidates = [c for c in self.chunks if c["id"] not in PINNED_SECTIONS]
        self.backend, self.min_k, self.max_k = backend, min_k, max_k
        self.embed_model, self.host = embed_model, ollama_host
        docs = [f"{c['title']}\n{c['text']}" for c in self.candidates]
        if backend == "bm25":
            self._bm25 = _BM25(docs)
        elif backend == "ollama":
            self._doc_vecs = _ollama_embed(["search_document: " + d for d in docs], embed_model, ollama_host)
        else:
            raise ValueError(f"Unknown retriever backend: {backend}")

    def config(self):
        return {"backend": self.backend, "min_k": self.min_k, "max_k": self.max_k,
                "pinned": list(PINNED_SECTIONS),
                "embed_model": self.embed_model if self.backend == "ollama" else None}

    def rank(self, query):
        if self.backend == "bm25":
            scores = self._bm25.scores(query)
        else:
            qv = _ollama_embed(["search_query: " + query], self.embed_model, self.host)[0]
            scores = [_cosine(qv, dv) for dv in self._doc_vecs]
        return sorted(zip(self.candidates, scores), key=lambda x: -x[1])

    def retrieve(self, profile, top_k=None):
        """Return (chunks, retrieved_ids): top-k indicator sections followed by pinned sections.

        Dynamic k: one section per detected signal, clamped to [min_k, max_k].
        """
        descriptors = build_descriptors(profile)
        k = top_k if top_k is not None else max(self.min_k, min(self.max_k, len(descriptors)))
        ranked = self.rank(". ".join(descriptors))[:k]
        retrieved = [c for c, _ in ranked]
        return retrieved + self.pinned, [c["id"] for c in retrieved]
