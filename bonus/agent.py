"""Bonus challenge — HybridMemoryAgent: vector episodic memory + Feast profile.

Design decisions live in bonus/ARCHITECTURE.md; this file is the minimal POC.
Patterns reused from the main lab: app/embeddings.py (pluggable embedder),
app/search.py (BM25 + vector + RRF k=60), app/feast_repo (user_profile +
query_velocity feature views, already materialized by NB4).
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter,
                                  MatchValue, PointStruct, VectorParams)
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

COLLECTION = "bonus_memory"
CHUNK_MAX_WORDS = 60          # ARCHITECTURE.md decision 1: fixed-size chunks
RRF_K = 60                    # same default as app/search.py

# Fallback if Feast is not materialized yet (fresh clone before NB4 runs).
# Real system would fail loudly; POC degrades so `python bonus/demo.py` exits 0.
_PROFILE_FALLBACK = {
    "reading_speed_wpm": 200, "preferred_language": "vi",
    "topic_affinity": "cloud", "queries_last_hour": 0,
    "distinct_topics_24h": 0,
}


def chunk_text(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    """Split sentences, pack into <= max_words chunks (no external tokenizer)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
    chunks, cur = [], []
    for s in sentences:
        while len(s.split()) > max_words:       # one giant sentence: hard window
            words = s.split()
            cur.append(" ".join(words[:max_words]))
            chunks.append(" ".join(cur)); cur = []
            s = " ".join(words[max_words:])
        if len(" ".join(cur + [s]).split()) > max_words and cur:
            chunks.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class HybridMemoryAgent:
    """Episodic memory in Qdrant (user_id payload filter), profile in Feast."""

    def __init__(self) -> None:
        self.embedder = Embedder()
        mode = os.getenv("QDRANT_MODE", "memory")
        if mode == "server":
            self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
            if not self.client.collection_exists(COLLECTION):
                self.client.create_collection(
                    collection_name=COLLECTION,
                    vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE))
        else:
            self.client = QdrantClient(":memory:")
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE))
        self._chunks: dict[str, list[dict]] = {}   # user_id -> [{"text": ...}]
        self._store = self._load_feast()

    def _load_feast(self):
        try:
            from feast import FeatureStore
            return FeatureStore(repo_path=str(_REPO_ROOT / "app" / "feast_repo"))
        except Exception:                                   # registry missing, etc.
            return None

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        parts = chunk_text(text)
        vectors = list(self.embedder.embed(parts))
        points = []
        for i, (part, vec) in enumerate(zip(parts, vectors)):
            chunk_id = f"{user_id}:{uuid.uuid4().hex[:8]}:{i}"
            points.append(PointStruct(id=uuid.uuid4().hex, vector=vec.tolist(),
                                      payload={"user_id": user_id, "text": part,
                                               "chunk_id": chunk_id}))
            self._chunks.setdefault(user_id, []).append({"text": part, "chunk_id": chunk_id})
        self.client.upsert(collection_name=COLLECTION, points=points)

    # ── read path ───────────────────────────────────────────────────────
    def _profile(self, user_id: str) -> dict:
        if self._store is None:
            return dict(_PROFILE_FALLBACK)
        try:
            feats = self._store.get_online_features(
                features=["user_profile_features:reading_speed_wpm",
                          "user_profile_features:preferred_language",
                          "user_profile_features:topic_affinity",
                          "query_velocity_features:queries_last_hour",
                          "query_velocity_features:distinct_topics_24h"],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {k: (feats[k][0] if feats[k][0] is not None else _PROFILE_FALLBACK[k])
                    for k in _PROFILE_FALLBACK}
        except Exception:
            return dict(_PROFILE_FALLBACK)

    def _hybrid_search(self, query: str, user_id: str, top_k: int = 3):
        """BM25 over user chunks + Qdrant vector filtered by user_id, RRF-fused."""
        chunks = self._chunks.get(user_id, [])
        kw_ranked: list[str] = []
        if chunks:
            bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])
            scores = bm25.get_scores(query.lower().split())
            kw_ranked = [chunks[i]["chunk_id"]
                         for i in sorted(range(len(scores)), key=lambda i: -scores[i])
                         if scores[i] > 0]

        filt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        q_vec = next(self.embedder.embed([query])).tolist()
        sem_ranked = [p.payload["chunk_id"] for p in self.client.query_points(
            collection_name=COLLECTION, query=q_vec, query_filter=filt,
            limit=max(top_k * 5, 20)).points]

        rrf: dict[str, float] = {}
        for ranked in (kw_ranked, sem_ranked):
            for rank, cid in enumerate(ranked, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        # POC scope: chunk text lives in-process (lite mode). Server mode with
        # memories written by another process would need a payload fetch here.
        text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
        return [(cid, text_by_id.get(cid, ""), score) for cid, score in top]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features -> assembled context."""
        profile = self._profile(user_id)
        memories = self._hybrid_search(query, user_id, top_k=3)
        lines = [
            f"User likes {profile['topic_affinity']} reading at "
            f"{profile['reading_speed_wpm']}wpm ({profile['preferred_language']}).",
            f"Recent activity: {profile['queries_last_hour']} queries in the last hour, "
            f"{profile['distinct_topics_24h']} distinct topics in 24h.",
        ]
        if memories:
            lines.append("Top memories:")
            for rank, (cid, text, score) in enumerate(memories, start=1):
                lines.append(f"  {rank}. [{score:.4f}] {text}")
        else:
            lines.append("Top memories: (none stored for this user yet)")
        return "\n".join(lines)
