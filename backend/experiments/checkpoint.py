"""
checkpoint.py

Resumable, cache-everything store for the experiment harness. Designed for the
Groq free-tier reality: a run may be interrupted by a daily-token cap and must
restart without repeating any paid LLM call.

Layout under results/<run_id>/:
    config_snapshot.json          frozen config + git SHA + seeds + args
    trajectories/<qid>.json       full-depth Writer→Critic trajectory (drafts,
                                  critic feedback, distances, per-round tokens)
    trajectories/<qid>.emb.json   draft embeddings (bulky; kept separate)
    judge_cache/<sha1>.json       RAGAS result for one distinct draft (judged once)
    rows.jsonl                    append-only ResultRow records (one per qid×policy×seed)

Resume semantics:
    * a trajectory file present → generation skipped for that question
    * a judge_cache entry present (keyed on the draft hash) → draft not re-judged
    * a (qid, policy, seed) already in rows.jsonl → that cell skipped
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Set

from shp.trajectory import Trajectory, RoundRecord
from experiments.metrics_schema import ResultRow, RoundScore


def draft_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class CheckpointStore:
    def __init__(self, root: str):
        self.root = root
        self.traj_dir = os.path.join(root, "trajectories")
        self.judge_dir = os.path.join(root, "judge_cache")
        self.rows_path = os.path.join(root, "rows.jsonl")
        os.makedirs(self.traj_dir, exist_ok=True)
        os.makedirs(self.judge_dir, exist_ok=True)
        self._row_keys: Set[str] = self._load_row_keys()

    # ── config snapshot ───────────────────────────────────────
    def save_config_snapshot(self, snapshot: Dict) -> None:
        with open(os.path.join(self.root, "config_snapshot.json"), "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2)

    # ── trajectories ──────────────────────────────────────────
    def _traj_path(self, qid: str) -> str:
        return os.path.join(self.traj_dir, f"{qid}.json")

    def _emb_path(self, qid: str) -> str:
        return os.path.join(self.traj_dir, f"{qid}.emb.json")

    def has_trajectory(self, qid: str) -> bool:
        return os.path.exists(self._traj_path(qid))

    def save_trajectory(self, traj: Trajectory) -> None:
        meta = {
            "scenario_id": traj.scenario_id,
            "topic": traj.topic,
            "question": traj.question,
            "contexts": traj.contexts,
            "ground_truth": traj.ground_truth,
            "rounds": [r.public() for r in traj.rounds],
        }
        with open(self._traj_path(traj.scenario_id), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        embeddings = [r.embedding for r in traj.rounds]
        with open(self._emb_path(traj.scenario_id), "w", encoding="utf-8") as fh:
            json.dump(embeddings, fh)

    def load_trajectory(self, qid: str) -> Trajectory:
        with open(self._traj_path(qid), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        embeddings: List[List[float]] = []
        if os.path.exists(self._emb_path(qid)):
            with open(self._emb_path(qid), "r", encoding="utf-8") as fh:
                embeddings = json.load(fh)
        traj = Trajectory(
            scenario_id=meta["scenario_id"], topic=meta["topic"],
            question=meta["question"], contexts=meta["contexts"],
            ground_truth=meta["ground_truth"],
        )
        for i, r in enumerate(meta["rounds"]):
            traj.rounds.append(RoundRecord(
                round=r["round"], draft=r["draft"], feedback=r["feedback"],
                is_approved=r["is_approved"], word_count=r["word_count"],
                distance=r["distance"],
                embedding=embeddings[i] if i < len(embeddings) else [],
                writer_tokens=r["writer_tokens"], critic_tokens=r["critic_tokens"],
            ))
        return traj

    # ── judge cache ───────────────────────────────────────────
    def _judge_path(self, dhash: str) -> str:
        return os.path.join(self.judge_dir, f"{dhash}.json")

    def get_score(self, draft: str) -> Optional[RoundScore]:
        path = self._judge_path(draft_hash(draft))
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return RoundScore.from_dict(json.load(fh))

    def put_score(self, draft: str, score: RoundScore) -> None:
        with open(self._judge_path(draft_hash(draft)), "w", encoding="utf-8") as fh:
            json.dump(score.to_dict(), fh, indent=2)

    # ── result rows ───────────────────────────────────────────
    def _load_row_keys(self) -> Set[str]:
        keys: Set[str] = set()
        if os.path.exists(self.rows_path):
            with open(self.rows_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    keys.add(ResultRow.key(d["scenario_id"], d["policy"], d["seed"]))
        return keys

    def has_row(self, scenario_id: str, policy: str, seed: int) -> bool:
        return ResultRow.key(scenario_id, policy, seed) in self._row_keys

    def append_row(self, row: ResultRow) -> None:
        with open(self.rows_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.to_dict()) + "\n")
        self._row_keys.add(row.row_key)

    def load_rows(self) -> List[ResultRow]:
        rows: List[ResultRow] = []
        if os.path.exists(self.rows_path):
            with open(self.rows_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(ResultRow.from_dict(json.loads(line)))
        return rows
