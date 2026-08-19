# Reflection — Lab 19

**Tên:** _Nguyễn Quang Huy_
**Cohort:** _A20-K3_
**Path đã chạy:** _lite_

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên golden set 50 queries, hybrid (RRF k=60) thắng tổng thể: Precision@10
78,6% so với keyword 77,8% và semantic 73,2%. Tách theo loại: hybrid cùng
keyword hòa ở `exact` (96,7%), hybrid một mình lên 100% ở `mixed` (semantic
98,5%), còn `paraphrase` thì cả ba mode đều sụp (keyword 33,3%, semantic
24,0%, hybrid 32,0%) — vì embedder mặc định bge-small chỉ tiếng Anh, vector
không bắt được paraphrase tiếng Việt như "co giãn linh hoạt". **Không** dùng
hybrid khi: (1) query exact-match (tên, mã, doc_id) — pure BM25 đủ, P50
1,6ms so với hybrid 24,1ms, tiết kiệm ~15× latency; (2) budget latency quá
chặt; (3) query thuần paraphrase khái niệm cross-lingual và có model
multilingual tốt — lúc đó BM25 chỉ thêm noise, pure vector gọn hơn.

---

## Điều ngạc nhiên nhất khi làm lab này

Mode vector "hiện đại" thua BM25 ở query paraphrase tiếng Việt (trung bình
73,2% so với 77,8%; riêng loại paraphrase 24% so với 33,3%) — chỉ vì embedder
mặc định là English-only. Bài học: chọn model đúng ngôn ngữ dữ liệu quan
trọng hơn chọn thuật toán.

---

## Bonus challenge

- [X] Đã làm bonus (xem `bonus/`) — `bonus/ARCHITECTURE.md` (3 quyết định
  kiến trúc + tradeoff), `bonus/agent.py` (HybridMemoryAgent: Qdrant +
  Feast + RRF), `bonus/demo.py` (5 queries, `python bonus/demo.py` exits 0)
- [X] Pair work với:  _làm cá nhân_
