from __future__ import annotations
import json
import os
import time
import hashlib
from typing import Any, Dict, List, Optional
from collections import deque

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Try to import LLM helpers safely (do not crash server if deps/keys missing)
HAVE_LLM = False
LLM_IMPORT_ERROR: Optional[str] = None
try:
    from llm import generate_natural_language_program as _gen_nlp, variates as _variates
    HAVE_LLM = True
except Exception as _e:  # noqa: BLE001
    LLM_IMPORT_ERROR = str(_e)
    _gen_nlp = None  # type: ignore
    _variates = None  # type: ignore

# ---------- Paths ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "training": {
        "ch": os.path.join(ROOT, "arc-agi_training_challenges.json"),
        "sol": os.path.join(ROOT, "arc-agi_training_solutions.json"),
    },
    "evaluation": {
        "ch": os.path.join(ROOT, "arc-agi_evaluation_challenges.json"),
        "sol": os.path.join(ROOT, "arc-agi_evaluation_solutions.json"),
    },
}
SAVED_PATH = os.path.join(ROOT, "saved_annotations.jsonl")

# ---------- Caches ----------
CACHE: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {
    "training": {"ch": None, "sol": None},
    "evaluation": {"ch": None, "sol": None},
}

# ---------- Utils ----------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(split: str) -> Dict[str, Dict[str, Any]]:
    if split not in CACHE:
        raise ValueError("split must be 'training' or 'evaluation'")
    if CACHE[split]["ch"] is None:
        CACHE[split]["ch"] = load_json(FILES[split]["ch"])
    # Solutions may be missing for evaluation; load lazily and tolerate absence
    if CACHE[split]["sol"] is None:
        sol_path = FILES[split]["sol"]
        if os.path.exists(sol_path):
            CACHE[split]["sol"] = load_json(sol_path)
        else:
            CACHE[split]["sol"] = {}
    return {"ch": CACHE[split]["ch"], "sol": CACHE[split]["sol"]}


def js_style_grid_hash(grid: List[List[int]]) -> str:
    # Mirrors the JS function used in the UI: hash over flattened comma-joined string
    flat = ",".join(str(cell) for row in grid for cell in row)
    h = 0
    for ch in flat:
        h = ((h << 5) - h) + ord(ch)
        h &= 0xFFFFFFFF  # 32-bit
    return f"{h:08x}"


def robust_hash(obj: Any) -> str:
    s = json.dumps(obj, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def validate_output(split: str, task_id: str, test_index: int, output: List[List[int]]) -> Optional[bool]:
    data = load_split(split)
    sol = data["sol"] or {}
    sol_task = (sol or {}).get(task_id)
    if not sol_task:
        return None

    # Support two possible formats:
    # 1) list-of-outputs: {task_id: [output0, output1, ...]}
    # 2) dict with "test" examples: {task_id: {"test":[{"output": ...}, ...]}}
    gt = None
    if isinstance(sol_task, list):
        if not (0 <= test_index < len(sol_task)):
            return None
        gt = sol_task[test_index]
    elif isinstance(sol_task, dict):
        test = sol_task.get("test")
        if not isinstance(test, list) or test_index >= len(test):
            return None
        gt = test[test_index].get("output")
    else:
        return None

    if gt is None:
        return None

    # shape check
    if len(output) != len(gt) or (len(output) > 0 and len(output[0]) != len(gt[0])):
        return False
    for r in range(len(gt)):
        for c in range(len(gt[0])):
            if int(output[r][c]) != int(gt[r][c]):
                return False
    return True


def task_test_hashes(split: str, task_id: str) -> List[str]:
    data = load_split(split)
    sol = data["sol"] or {}
    sol_task = (sol or {}).get(task_id)
    if not sol_task:
        return []

    hashes: List[str] = []
    if isinstance(sol_task, list):
        for out in sol_task:
            if out is not None:
                try:
                    hashes.append(js_style_grid_hash(out))
                except Exception:
                    hashes.append(robust_hash(out))
    elif isinstance(sol_task, dict):
        for ex in sol_task.get("test", []) or []:
            out = ex.get("output")
            if out is not None:
                try:
                    hashes.append(js_style_grid_hash(out))
                except Exception:
                    hashes.append(robust_hash(out))
    return hashes


def read_saved_last(limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(SAVED_PATH):
        return []
    dq: deque[str] = deque(maxlen=max(1, limit))
    with open(SAVED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    out: List[Dict[str, Any]] = []
    for line in dq:
        try:
            out.append(json.loads(line))
        except Exception:
            # skip malformed
            pass
    # return newest-last in file order; for API, return newest first
    out.reverse()
    return out


def saved_hashes_for_task(task_id: str) -> List[str]:
    if not os.path.exists(SAVED_PATH):
        return []
    hashes: set[str] = set()
    with open(SAVED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("task_id") == task_id:
                h = rec.get("output_hash")
                if isinstance(h, str):
                    hashes.add(h)
    return sorted(hashes)


def get_saved_status_record(split: str, task_id: str, test_index: int) -> Optional[Dict[str, Any]]:
    """Return the saved record for (split, task_id, test_index) if present, else None.
    File is deduplicated per key on save, but we still scan defensively.
    """
    if not os.path.exists(SAVED_PATH):
        return None
    found: Optional[Dict[str, Any]] = None
    with open(SAVED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if (
                rec.get("split") == split
                and rec.get("task_id") == task_id
                and int(rec.get("test_index", -1)) == int(test_index)
            ):
                found = rec
    return found


# ---------- App ----------
app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "ts": time.time()})


@app.get("/api/tasks")
def list_tasks():
    split = request.args.get("split", "training")
    data = load_split(split)
    ch = data["ch"]
    ids = sorted(ch.keys())
    return jsonify({
        "split": split,
        "count": len(ids),
        "task_ids": ids,
    })


@app.get("/api/task/<task_id>")
def get_task(task_id: str):
    split = request.args.get("split", "training")
    data = load_split(split)
    ch = data["ch"]
    sol = data["sol"] or {}

    if task_id not in ch:
        return jsonify({"error": "task_not_found", "task_id": task_id}), 404

    task = ch[task_id]
    train = task.get("train", [])
    test = task.get("test", [])

    # Extract ground-truth for test if available
    gt: Optional[List[List[List[int]]]] = None
    sol_task = sol.get(task_id)
    if isinstance(sol_task, dict) and isinstance(sol_task.get("test"), list):
        gt = []
        for ex in sol_task.get("test", []):
            gt.append(ex.get("output"))

    return jsonify({
        "task_id": task_id,
        "split": split,
        "train": train,
        "test": test,
        "gt": gt,  # may be None or list of outputs
    })


@app.post("/api/save")
def save_annotation():
    payload = request.get_json(force=True, silent=True) or {}
    task_id = payload.get("task_id")
    split = payload.get("split", "training")
    test_index = int(payload.get("test_index", 0))
    output = payload.get("output")  # expected 2D list of ints
    insts = payload.get("insts", [])  # list of strings

    if not task_id or not isinstance(output, list):
        return jsonify({"error": "invalid_request"}), 400

    # Compute hashes
    try:
        grid_hash = js_style_grid_hash(output)
    except Exception:
        grid_hash = robust_hash(output)

    ok = validate_output(split, task_id, test_index, output)
    all_test_hashes = task_test_hashes(split, task_id)

    # If correct and GT hashes are available for this test index, use canonical GT hash
    if ok is True and 0 <= test_index < len(all_test_hashes):
        grid_hash = all_test_hashes[test_index]

    # Join instructions with newlines for LLM processing and save alongside list
    if isinstance(insts, list):
        insts_text = "\n".join(str(s) for s in insts)
    elif isinstance(insts, str):
        insts_text = insts
    else:
        insts_text = str(insts)

    # Call LLM augmentation (graceful on errors)
    nlp_program: Optional[str] = None
    variations: Optional[List[str]] = None
    llm_error: Optional[str] = None
    try:
        if insts_text.strip():
            if HAVE_LLM and _gen_nlp and _variates:
                nlp_obj = _gen_nlp(insts_text)
                nlp_program = getattr(nlp_obj, "program", None)
                vars_obj = _variates(insts_text)
                variations = getattr(vars_obj, "variations", None)
            else:
                llm_error = f"llm_unavailable: {LLM_IMPORT_ERROR}" if LLM_IMPORT_ERROR else "llm_unavailable"
    except Exception as e:  # noqa: BLE001
        llm_error = str(e)

    rec = {
        "task_id": task_id,
        "split": split,
        "test_index": test_index,
        "output": output,
        "output_hash": grid_hash,
        "output_sha1": robust_hash(output),
        "ok": ok,  # True/False or None if unknown
        "insts": insts,
        "insts_text": insts_text,  # newline-joined instructions
        "nlp_program": nlp_program,
        "variations": variations,
        "llm_error": llm_error,
        "all_test_hashes": all_test_hashes,  # hashes for all GT test outputs (when available)
        "timestamp": time.time(),
    }

    # Rewrite saved_annotations.jsonl to remove older entries for same (split, task_id, test_index)
    # then append the new record. This effectively "replaces" prior saved hashes/records.
    os.makedirs(os.path.dirname(SAVED_PATH), exist_ok=True) if os.path.dirname(SAVED_PATH) else None
    existing_lines: List[str] = []
    if os.path.exists(SAVED_PATH):
        with open(SAVED_PATH, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    tmp_path = SAVED_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for line in existing_lines:
            keep = True
            try:
                prev = json.loads(line)
                if (
                    prev.get("split") == split
                    and prev.get("task_id") == task_id
                    and int(prev.get("test_index", -1)) == int(test_index)
                ):
                    keep = False  # drop older record for same key
            except Exception:
                # If malformed, keep line to avoid data loss
                keep = True
            if keep:
                out_f.write(line if line.endswith("\n") else (line + "\n"))
        # Append the new record at the end
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Atomic replace
    os.replace(tmp_path, SAVED_PATH)

    return jsonify({"saved": True, "record": rec})


@app.get("/api/saved")
def list_saved():
    try:
        limit = int(request.args.get("limit", 50))
    except Exception:
        limit = 50
    limit = max(1, min(limit, 1000))
    items = read_saved_last(limit)
    return jsonify({
        "limit": limit,
        "count": len(items),
        "items": items,
    })


@app.get("/api/saved/status")
def saved_status():
    task_id = request.args.get("task_id", "")
    split = request.args.get("split", "training")
    try:
        test_index = int(request.args.get("test_index", "0"))
    except Exception:
        test_index = 0
    if not task_id:
        return jsonify({"error": "invalid_request"}), 400

    rec = get_saved_status_record(split, task_id, test_index)
    if rec is None:
        return jsonify({"exists": False, "ok": None})
    # Return a small summary for convenience
    return jsonify({
        "exists": True,
        "ok": rec.get("ok"),
        "output_hash": rec.get("output_hash"),
        "timestamp": rec.get("timestamp"),
    })


@app.get("/api/task_hashes/<task_id>")
def get_task_hashes(task_id: str):
    split = request.args.get("split", "training")
    gt_hashes = task_test_hashes(split, task_id)
    saved = saved_hashes_for_task(task_id)
    return jsonify({
        "task_id": task_id,
        "split": split,
        "gt_hashes": gt_hashes,          # hashes of GT test outputs when available
        "saved_output_hashes": saved,     # hashes of outputs saved by users for this task
        "counts": {"gt": len(gt_hashes), "saved": len(saved)},
    })


@app.get("/api/download/saved")
def download_saved():
    if not os.path.exists(SAVED_PATH):
        # create empty file for convenience
        open(SAVED_PATH, "a").close()
    return send_file(SAVED_PATH, as_attachment=True, download_name="saved_annotations.jsonl")


@app.get("/")
@app.get("/ui")
def serve_ui():
    # Serve the labeller UI
    path = os.path.join(ROOT, "arc_labeller_v2.html")
    if not os.path.exists(path):
        return jsonify({"error": "ui_not_found"}), 404
    return send_file(path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)