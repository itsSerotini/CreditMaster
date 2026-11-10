"""CreditMaster v2 — shared core used by dataset generation, training, evaluation and inference.

Modules:
    rules      Rule engine aligned with Dataset/credit_policy_en.md (gold labels).
    policy     Loads and chunks the credit policy by section.
    retriever  Policy retriever (BM25 or Ollama embeddings) + query builder.
    prompts    System/user prompt builders (v1 legacy and v2 RAG).
    schema     Output schema constants and validation.
"""
