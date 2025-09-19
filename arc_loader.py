from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any

Grid = List[List[int]]
Pair = Tuple[Grid, Grid]


class ArcAgiDatasetLoader:
    """
    Loader for ARC-AGI training challenges and solutions.

    Files expected (defaults match repository root):
    - arc-agi_training_challenges.json: {task_hash: {"train": [{"input": Grid, "output": Grid}, ...], "test": [{"input": Grid}, ...]}}
    - arc-agi_training_solutions.json: {task_hash: [Grid, Grid, ...]} (solutions for each test input in order)

    Primary goal:
    - Given a task hash, return a list of (input, output) pairs. Train pairs
      come directly from the challenges. Test pairs use the solutions file to
      provide outputs.

    Usage:
        loader = ArcAgiDatasetLoader()
        pairs = loader["00576224"]  # list[tuple[Grid, Grid]]

        # Or explicitly:
        train_pairs = loader.get_train_pairs("00576224")
        test_pairs = loader.get_test_pairs("00576224")  # uses solutions
        all_pairs = loader.get_all_pairs("00576224")
    """

    def __init__(
        self,
        challenges_path: str = "arc-agi_training_challenges.json",
        solutions_path: str = "arc-agi_training_solutions.json",
        lazy: bool = False,
    ) -> None:
        self._challenges_path = challenges_path
        self._solutions_path = solutions_path
        self._lazy = lazy
        self._challenges: Optional[Dict[str, Any]] = None
        self._solutions: Optional[Dict[str, Any]] = None

        if not self._lazy:
            self._load_all()

    # -------------------- Loading --------------------
    def _load_all(self) -> None:
        self._load_challenges()
        self._load_solutions()

    def _ensure_loaded(self) -> None:
        if self._challenges is None:
            self._load_challenges()
        if self._solutions is None:
            self._load_solutions()

    def _load_challenges(self) -> None:
        with open(self._challenges_path, "r", encoding="utf-8") as f:
            self._challenges = json.load(f)

    def _load_solutions(self) -> None:
        with open(self._solutions_path, "r", encoding="utf-8") as f:
            self._solutions = json.load(f)

    # -------------------- Public API --------------------
    def available_hashes(self) -> List[str]:
        """Return sorted list of available task hashes from the challenges file."""
        if self._challenges is None:
            self._load_challenges()
        assert self._challenges is not None
        return sorted(self._challenges.keys())

    def has_hash(self, task_hash: str) -> bool:
        if self._challenges is None:
            self._load_challenges()
        assert self._challenges is not None
        return task_hash in self._challenges

    def get_train_pairs(self, task_hash: str) -> List[Pair]:
        """Return list of (input, output) pairs from the training split for the given hash."""
        self._ensure_loaded()
        assert self._challenges is not None

        if task_hash not in self._challenges:
            raise KeyError(f"Task hash not found in challenges: {task_hash}")

        entry = self._challenges[task_hash]
        train = entry.get("train", [])
        pairs: List[Pair] = []
        for ex in train:
            inp = ex.get("input")
            out = ex.get("output")
            if inp is None or out is None:
                raise ValueError(f"Malformed training example for {task_hash}: missing input or output")
            pairs.append((inp, out))
        return pairs

    def get_test_pairs(
        self,
        task_hash: str,
        require_solutions: bool = True,
    ) -> List[Pair]:
        """
        Return list of (input, output) pairs for the test split, using solutions
        to populate outputs.

        - If require_solutions is True (default), raises if solutions are missing
          or count does not match the number of test inputs.
        - If require_solutions is False, returns pairs where output may be None
          (type: ignore) when unavailable; if counts mismatch, pairs up to the
          smaller length and fills remaining outputs with None.
        """
        self._ensure_loaded()
        assert self._challenges is not None and self._solutions is not None

        if task_hash not in self._challenges:
            raise KeyError(f"Task hash not found in challenges: {task_hash}")

        entry = self._challenges[task_hash]
        test = entry.get("test", []) or []
        test_inputs: List[Grid] = [ex.get("input") for ex in test]
        if any(inp is None for inp in test_inputs):
            raise ValueError(f"Malformed test example for {task_hash}: missing input")

        sols: Optional[List[Grid]] = self._solutions.get(task_hash) if task_hash in self._solutions else None

        if require_solutions:
            if sols is None:
                raise KeyError(f"No solutions found for task hash: {task_hash}")
            if len(sols) != len(test_inputs):
                raise ValueError(
                    f"Solutions count mismatch for {task_hash}: {len(sols)} solutions vs {len(test_inputs)} test inputs"
                )
            return list(zip(test_inputs, sols))  # type: ignore[arg-type]
        else:
            pairs: List[Pair] = []
            if sols is None:
                # type: ignore[assignment]
                return [(inp, None) for inp in test_inputs]  # type: ignore[return-value]
            n = min(len(test_inputs), len(sols))
            pairs.extend((test_inputs[i], sols[i]) for i in range(n))
            # Fill remaining with None outputs if any
            for j in range(n, len(test_inputs)):
                # type: ignore[misc]
                pairs.append((test_inputs[j], None))  # type: ignore[arg-type]
            return pairs

    def get_all_pairs(self, task_hash: str, require_solutions: bool = True) -> List[Pair]:
        """Return train pairs followed by test pairs (test outputs sourced from solutions)."""
        train = self.get_train_pairs(task_hash)
        test = self.get_test_pairs(task_hash, require_solutions=require_solutions)
        return train + test

    def __getitem__(self, task_hash: str) -> List[Pair]:
        """Alias for get_all_pairs(task_hash)."""
        return self.get_all_pairs(task_hash)

    def raw_task(self, task_hash: str) -> Dict[str, Any]:
        """Return raw JSON entry for the given hash (train/test dict)."""
        if self._challenges is None:
            self._load_challenges()
        assert self._challenges is not None
        if task_hash not in self._challenges:
            raise KeyError(f"Task hash not found in challenges: {task_hash}")
        return self._challenges[task_hash]

    def raw_solution(self, task_hash: str) -> Optional[List[Grid]]:
        """Return raw solutions list for the given hash (or None if missing)."""
        if self._solutions is None:
            self._load_solutions()
        assert self._solutions is not None
        return self._solutions.get(task_hash)

    # -------------------- Iteration helpers --------------------
    def iter_hashes(self) -> Iterator[str]:
        """Iterate over available task hashes."""
        for h in self.available_hashes():
            yield h

    def iter_pairs(self, include_test: bool = True, require_solutions: bool = True) -> Iterator[Tuple[str, Pair]]:
        """Iterate over (hash, (input, output)) for all tasks.

        include_test: include test pairs (using solutions) after train pairs.
        require_solutions: if True, raise on missing/mismatched solutions for test.
        """
        for h in self.iter_hashes():
            # Train
            for p in self.get_train_pairs(h):
                yield h, p
            if include_test:
                for p in self.get_test_pairs(h, require_solutions=require_solutions):
                    yield h, p

