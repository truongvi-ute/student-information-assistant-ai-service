import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.services.inference.retriever import vector_retriever


def evaluate_retrieval(dataset_path: Path, top_k: int = 5) -> None:
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    answerable_items = [item for item in items if item.get("category") == "answerable"]

    hits = 0
    latencies: list[float] = []

    print("=" * 60)
    print(f"ĐÁNH GIÁ BASELINE RETRIEVAL (Top-K = {top_k})")
    print(f"Tổng số mẫu: {len(items)} | Số câu có đáp án (answerable): {len(answerable_items)}")
    print("=" * 60)

    for item in answerable_items:
        question = item["question"]
        keywords = [kw.lower() for kw in item.get("expected_keywords", [])]

        start_t = time.perf_counter()
        chunks = vector_retriever.retrieve(query=question, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        latencies.append(elapsed_ms)

        combined_text = " ".join(c.content.lower() for c in chunks)
        is_hit = all(kw in combined_text for kw in keywords) if keywords else bool(chunks)
        if is_hit:
            hits += 1

        status_label = "HIT" if is_hit else "MISS"
        print(f"[{item['id']}] {status_label} ({elapsed_ms:.1f} ms) - {question}")

    hit_rate = (hits / len(answerable_items) * 100) if answerable_items else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    print("\n--- KẾT QUẢ BASELINE ---")
    print(f"Hit-Rate@{top_k}:       {hits}/{len(answerable_items)} ({hit_rate:.1f}%)")
    print(f"Độ trễ trung bình:  {avg_latency:.1f} ms/query")


def main():
    parser = argparse.ArgumentParser(description="Chạy đánh giá baseline cho Retrieval")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/dataset.json",
        help="Đường dẫn tới file dataset.json",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Giá trị K cho Hit-Rate@K")
    args = parser.parse_args()

    evaluate_retrieval(Path(args.dataset), top_k=args.top_k)


if __name__ == "__main__":
    main()
