phase: IMPLEMENTATION
feature: LLM-augmented save flow
status: implemented
changes:
  - insts are newline-joined into insts_text and saved alongside original list
  - LLM integration: generate_natural_language_program + variates called on insts_text
  - saved record now includes: nlp_program, variations, llm_error (on failure)
  - correct-answer canonicalization: when ok=True and GT hashes available, output_hash set to GT hash for that test_index
  - deduplication: previous entries for same (split, task_id, test_index) removed by rewriting JSONL atomically before appending new record
  - safe fallback: lazy import of llm; if unavailable or errors, request still succeeds and llm_error notes reason
api_impacts:
  - POST /api/save returns the augmented record with new fields
  - GET /api/saved surfaces those fields in items[]
ops_notes:
  - Requires OpenAI key in env (.env) for LLM usage; otherwise llm_error will indicate unavailability
  - LLM calls are synchronous and may add latency; consider async/queue later if needed
verification_next:
  - Switch to TEST phase and add pytest cases:
    * save: verifies newline joining, presence of fields, deduplication behavior, canonical hash when ok=True
    * mock LLM to avoid network; ensure fallback path works when LLM missing
    * saved listing reflects latest record only for a given (split, task_id, test_index)
implementation_complete: true
next_phase_recommendation: TEST