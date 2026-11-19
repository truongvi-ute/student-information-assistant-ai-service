import argparse
import contextlib
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.schemas.request import ChatHistoryMessage
from app.services.inference.generator import rag_generator
from app.services.inference.retriever import vector_retriever


def _normalize_text(value: str) -> str:
    """Chuẩn hóa chuỗi tiếng Việt về chữ thường không dấu để đối chiếu tên file ổn định."""
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.replace("đ", "d").replace("Đ", "d").lower().split())


def _compact_alnum(value: str) -> str:
    """Rút gọn chuỗi chỉ giữ chữ cái/chữ số không dấu để khớp cả tên file viết liền."""
    return "".join(ch for ch in _normalize_text(value) if ch.isalnum())


def _filename_matches_group(filename: str, patterns: list[str]) -> bool:
    """Kiểm tra tên tài liệu có khớp với ít nhất một mẫu nhận diện trong nhóm hay không."""
    normalized_filename = _normalize_text(filename)
    compact_filename = _compact_alnum(filename)
    for pattern in patterns:
        normalized_pattern = _normalize_text(pattern)
        compact_pattern = _compact_alnum(pattern)
        if (normalized_pattern and normalized_pattern in normalized_filename) or (
            compact_pattern and compact_pattern in compact_filename
        ):
            return True
    return False


def compute_citation_precision(
    returned_filenames: list[str],
    expected_source_groups: list[list[str]],
    expect_empty_sources: bool,
) -> float:
    """Tính Citation Precision cho một mẫu đánh giá."""
    if expect_empty_sources:
        return 1.0 if len(returned_filenames) == 0 else 0.0

    if not returned_filenames:
        return 0.0

    if not expected_source_groups:
        return 1.0

    matched_sources = sum(
        1
        for filename in returned_filenames
        if any(_filename_matches_group(filename, group) for group in expected_source_groups)
    )
    return matched_sources / len(returned_filenames)


def compute_multi_doc_coverage(
    returned_filenames: list[str],
    expected_source_groups: list[list[str]],
) -> bool:
    """Kiểm tra tập nguồn trả về có bao phủ đủ mọi nhóm tài liệu kỳ vọng hay không."""
    if not expected_source_groups:
        return bool(returned_filenames)

    return all(
        any(_filename_matches_group(filename, group) for filename in returned_filenames)
        for group in expected_source_groups
    )


def evaluate_retrieval(dataset_path: Path, top_k: int = 5) -> None:
    """Đánh giá riêng tầng VectorRetriever (Hit-Rate@K)."""
    items: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    retrieval_items = [
        item
        for item in items
        if item.get("category") in {"answerable", "multi_turn", "comparison"}
    ]

    hits = 0
    latencies: list[float] = []

    print("=" * 68)
    print(f"ĐÁNH GIÁ RETRIEVAL (Top-K = {top_k})")
    print(f"Tổng số mẫu: {len(items)} | Số câu cần truy xuất: {len(retrieval_items)}")
    print("=" * 68)

    for item in retrieval_items:
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
        print(f"[{item['id']}] ({item['category']}) {status_label} ({elapsed_ms:.1f} ms) - {question}")

    hit_rate = (hits / len(retrieval_items) * 100) if retrieval_items else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    print("\n--- KẾT QUẢ RETRIEVAL ---")
    print(f"Hit-Rate@{top_k}:       {hits}/{len(retrieval_items)} ({hit_rate:.1f}%)")
    print(f"Độ trễ trung bình:  {avg_latency:.1f} ms/query")


def evaluate_rag(dataset_path: Path, top_k: int = 5) -> None:
    """Đánh giá toàn trình Adaptive RAG: single-turn, multi-turn, so sánh đa tài liệu và Citation Precision."""
    items: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))

    latencies: list[float] = []
    keyword_hits = 0
    keyword_total = 0
    comparison_hits = 0
    comparison_total = 0
    empty_source_hits = 0
    empty_source_total = 0
    citation_precisions: list[float] = []

    print("=" * 72)
    print(f"ĐÁNH GIÁ ADAPTIVE RAG TOÀN TRÌNH (Top-K = {top_k})")
    print(f"Tổng số mẫu đánh giá: {len(items)}")
    print("=" * 72)

    for item in items:
        category = str(item.get("category", "answerable"))
        question = str(item["question"])
        raw_history = item.get("history") or []
        history = [
            ChatHistoryMessage(role=msg["role"], content=msg["content"])
            for msg in raw_history
        ]
        expected_keywords = [kw.lower() for kw in item.get("expected_keywords", [])]
        expected_groups: list[list[str]] = item.get("expected_source_groups", [])
        expect_empty = bool(
            item.get("expect_empty_sources", category in {"unanswerable", "no_retrieval", "non_academic"})
        )

        start_time = time.perf_counter()
        response = rag_generator.answer_question(
            question=question,
            top_k=top_k,
            history=history,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)

        returned_filenames = [source.filename for source in response.sources]
        precision = compute_citation_precision(
            returned_filenames=returned_filenames,
            expected_source_groups=expected_groups,
            expect_empty_sources=expect_empty,
        )
        citation_precisions.append(precision)

        if expect_empty:
            empty_source_total += 1
            empty_sources_matched = len(returned_filenames) == 0
            if empty_sources_matched:
                empty_source_hits += 1
            status_label = (
                "PASS (sources=[])"
                if empty_sources_matched
                else f"FAIL (got {len(returned_filenames)} sources)"
            )
        else:
            keyword_total += 1
            searchable_text = " ".join(
                [response.answer.lower(), *(source.content.lower() for source in response.sources)]
            )
            keywords_matched = (
                all(keyword in searchable_text for keyword in expected_keywords)
                if expected_keywords
                else bool(response.answer)
            )
            if keywords_matched:
                keyword_hits += 1

            coverage_note = ""
            if category == "comparison" or len(expected_groups) > 1:
                comparison_total += 1
                coverage_matched = compute_multi_doc_coverage(returned_filenames, expected_groups)
                if coverage_matched:
                    comparison_hits += 1
                coverage_note = f" | multi_doc={'OK' if coverage_matched else 'MISS'}"

            status_label = (
                f"{'HIT' if keywords_matched else 'MISS'} | cite_prec={precision:.2f}{coverage_note}"
            )

        print(
            f"[{item['id']}] ({category}) {status_label} "
            f"({elapsed_ms:.1f} ms | sources={len(returned_filenames)}) - {question}"
        )

    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    mean_precision = (
        sum(citation_precisions) / len(citation_precisions) * 100
        if citation_precisions
        else 0.0
    )
    keyword_hit_rate = (keyword_hits / keyword_total * 100) if keyword_total else 0.0
    comparison_hit_rate = (comparison_hits / comparison_total * 100) if comparison_total else 0.0
    empty_source_accuracy = (empty_source_hits / empty_source_total * 100) if empty_source_total else 0.0

    print("\n--- TỔNG HỢP KẾT QUẢ ADAPTIVE RAG ---")
    print(f"Answer/Context Hit-Rate: {keyword_hits}/{keyword_total} ({keyword_hit_rate:.1f}%)")
    print(f"Multi-Doc Comparison:    {comparison_hits}/{comparison_total} ({comparison_hit_rate:.1f}%)")
    print(f"Empty-Source Accuracy:   {empty_source_hits}/{empty_source_total} ({empty_source_accuracy:.1f}%)")
    print(f"Mean Citation Precision: {mean_precision:.1f}%")
    print(f"Độ trễ trung bình:       {avg_latency:.1f} ms/query")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy bộ đánh giá cho Adaptive RAG / Retrieval")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/dataset.json",
        help="Đường dẫn tới file dataset.json",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Giá trị K khi truy xuất")
    parser.add_argument(
        "--mode",
        choices=["rag", "retrieval"],
        default="rag",
        help="Chế độ đánh giá: 'rag' (toàn trình Adaptive RAG + Citation Precision) hoặc 'retrieval'",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if args.mode == "retrieval":
        evaluate_retrieval(dataset_path, top_k=args.top_k)
    else:
        evaluate_rag(dataset_path, top_k=args.top_k)


if __name__ == "__main__":
    main()
