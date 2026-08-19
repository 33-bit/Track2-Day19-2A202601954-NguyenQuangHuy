"""Bonus challenge demo — 5 queries showing when each memory layer carries the answer.

Run from anywhere:  python bonus/demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import HybridMemoryAgent

USER = "u_001"

# Episodic memory seed — Vietnamese with code-switching (vi/en mix), the way
# real users type. One chunk per memory keeps each inside the 60-word window.
MEMORIES = [
    "Tôi đã đọc bài về Kubernetes cluster: cách setup node pool, cấu hình "
    "kubectl context cho staging và prod, và cách dùng Helm chart để deploy.",
    "Ghi chú về autoscaling: Horizontal Pod Autoscaler trong Kubernetes scale "
    "theo CPU và custom metrics. Khi traffic tăng đột ngột, cluster tự động "
    "mở rộng hạ tầng thêm node mới, không cần can thiệp tay.",
    "Đã đọc checklist cloud security: tách VPC riêng cho từng môi trường, "
    "IAM least privilege, mã hoá at-rest bằng KMS, và bật audit log cho "
    "mọi truy cập console.",
    "Note cuộc họp tuần trước: team quyết định chọn Qdrant làm vector store "
    "và Feast làm feature store cho dự án trợ lý ảo tiếng Việt.",
    "Sáng nay note lại latency budget của search API: p95 dưới 200ms, "
    "dùng semantic cache cho các query lặp lại để giảm chi phí embedding.",
]

# (query, which layer the spec says it exercises)
QUERIES = [
    ("Tôi đã đọc gì về Kubernetes?", "vector hit"),
    ("Recommend đọc gì tiếp?", "needs topic_affinity profile"),
    ("Tôi đang quan tâm gì gần đây?", "needs fresh activity (queries_last_hour)"),
    ("Tài liệu về tự động mở rộng hạ tầng?", "paraphrase — vector wins"),
    ("Cho tôi summary cloud security", "hybrid + profile"),
]


def main() -> None:
    print(f"[demo] booting HybridMemoryAgent "
          f"(QDRANT_MODE={os.getenv('QDRANT_MODE', 'memory')}) ...")
    agent = HybridMemoryAgent()

    print(f"[demo] remembering {len(MEMORIES)} episodic memories for {USER}")
    for m in MEMORIES:
        agent.remember(m, user_id=USER)

    for i, (q, layer) in enumerate(QUERIES, start=1):
        print("\n" + "=" * 72)
        print(f"Q{i}: {q}")
        print(f"    (spec: {layer})")
        print("-" * 72)
        print(agent.recall(q, user_id=USER))
    print("\n[demo] done — 5/5 queries answered")


if __name__ == "__main__":
    main()
