import re
from dataclasses import dataclass

from app.core.config import settings
from app.schemas.document import Chunk, ExtractedDocument

_HEADING_RE = re.compile(r"^#{1,6}\s+.+")


@dataclass
class _MarkdownBlock:
    kind: str  # "heading" | "table" | "text"
    content: str


class TextChunker:
    """Bộ chia đoạn hỗ trợ cấu trúc Markdown (heading, paragraph) và bảo toàn bảng biểu."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|") and len(stripped) >= 2

    def _parse_markdown_blocks(self, text: str) -> list[_MarkdownBlock]:
        """Tách văn bản Markdown thành các khối heading, table và text thường."""
        lines = text.splitlines()
        blocks: list[_MarkdownBlock] = []
        text_buf: list[str] = []
        idx = 0

        def flush_text_buf() -> None:
            raw = "\n".join(text_buf).strip()
            text_buf.clear()
            if not raw:
                return
            for para in re.split(r"\n{2,}", raw):
                cleaned = para.strip()
                if not cleaned:
                    continue
                kind = (
                    "heading"
                    if _HEADING_RE.match(cleaned) and "\n" not in cleaned
                    else "text"
                )
                blocks.append(_MarkdownBlock(kind=kind, content=cleaned))

        while idx < len(lines):
            line = lines[idx]
            if self._is_table_line(line):
                table_lines: list[str] = []
                while idx < len(lines) and self._is_table_line(lines[idx]):
                    table_lines.append(lines[idx].strip())
                    idx += 1
                if len(table_lines) >= 2:
                    flush_text_buf()
                    blocks.append(
                        _MarkdownBlock(kind="table", content="\n".join(table_lines))
                    )
                    continue
                text_buf.extend(table_lines)
                continue

            if _HEADING_RE.match(line.strip()):
                flush_text_buf()
                blocks.append(_MarkdownBlock(kind="heading", content=line.strip()))
                idx += 1
                continue

            text_buf.append(line)
            idx += 1

        flush_text_buf()
        return blocks

    def _split_oversized_table(self, table_text: str) -> list[str]:
        """Nếu bảng vượt quá chunk_size, tách theo dòng và lặp lại header bảng ở mỗi phần."""
        lines = [ln for ln in table_text.splitlines() if ln.strip()]
        if len(lines) <= 2 or len(table_text) <= self.chunk_size:
            return [table_text]

        has_separator = (
            len(lines) >= 2
            and set(lines[1].replace("|", "").replace("-", "").replace(":", "").strip())
            == set()
        )
        header_lines = lines[:2] if has_separator else lines[:1]
        body_lines = lines[len(header_lines) :]
        header_block = "\n".join(header_lines)

        parts: list[str] = []
        current_rows: list[str] = []
        current_len = len(header_block)

        for row in body_lines:
            added_len = len(row) + 1
            if current_rows and current_len + added_len > self.chunk_size:
                parts.append(header_block + "\n" + "\n".join(current_rows))
                current_rows = []
                current_len = len(header_block)
            current_rows.append(row)
            current_len += added_len

        if current_rows:
            parts.append(header_block + "\n" + "\n".join(current_rows))

        return parts

    def _split_text_by_separators(self, text: str, separators: list[str]) -> list[str]:
        """Tách đệ quy đoạn văn dài theo danh sách ký tự phân cách ưu tiên."""
        if len(text) <= self.chunk_size or not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]
        parts = text.split(sep)
        result: list[str] = []

        for idx, part in enumerate(parts):
            if not part:
                continue
            candidate = (
                f"{part}{sep.rstrip()}"
                if sep in {". ", "; ", ", "} and idx < len(parts) - 1
                else part
            )
            if len(candidate) <= self.chunk_size:
                result.append(candidate)
            else:
                result.extend(self._split_text_by_separators(candidate, remaining_seps))

        return result

    def _split_prose_with_overlap(self, text: str) -> list[str]:
        """Chia đoạn văn bản thường quá dài thành nhiều mảnh có gối đầu (overlap)."""
        separators = ["\n", ". ", "; ", ", ", " "]
        raw_parts = self._split_text_by_separators(text, separators)

        pieces: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in raw_parts:
            part_len = len(part)
            sep_len = 1 if current else 0
            if current and current_len + sep_len + part_len > self.chunk_size:
                joined = " ".join(current).strip()
                if joined:
                    pieces.append(joined)

                overlap_items: list[str] = []
                overlap_len = 0
                for item in reversed(current):
                    extra = len(item) + (1 if overlap_items else 0)
                    if overlap_len + extra <= self.chunk_overlap:
                        overlap_items.insert(0, item)
                        overlap_len += extra
                    else:
                        break
                current = overlap_items
                current_len = overlap_len

            current.append(part)
            current_len += part_len + (1 if len(current) > 1 else 0)

        if current:
            joined = " ".join(current).strip()
            if joined:
                pieces.append(joined)

        return pieces

    def split_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Chia nhỏ một tài liệu Markdown thành danh sách Chunk mà vẫn giữ bảng biểu & ngữ cảnh."""
        text = text.strip()
        if not text:
            return []

        metadata = metadata or {}
        blocks = self._parse_markdown_blocks(text)

        units: list[_MarkdownBlock] = []
        for block in blocks:
            if block.kind == "table":
                for sub_table in self._split_oversized_table(block.content):
                    units.append(_MarkdownBlock(kind="table", content=sub_table))
            elif block.kind == "text" and len(block.content) > self.chunk_size:
                for sub_text in self._split_prose_with_overlap(block.content):
                    units.append(_MarkdownBlock(kind="text", content=sub_text))
            else:
                units.append(block)

        chunks: list[Chunk] = []
        current_units: list[_MarkdownBlock] = []
        current_len = 0
        chunk_idx = 0
        last_heading: str | None = None

        def flush_chunk() -> None:
            nonlocal chunk_idx
            content = "\n\n".join(u.content for u in current_units).strip()
            if content:
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_index=chunk_idx,
                        metadata=metadata.copy(),
                    )
                )
                chunk_idx += 1

        for unit in units:
            if unit.kind == "heading":
                last_heading = unit.content

            sep_len = 2 if current_units else 0
            unit_len = len(unit.content)

            if current_units and current_len + sep_len + unit_len > self.chunk_size:
                if len(current_units) == 1 and current_units[0].kind == "heading":
                    pass
                else:
                    flush_chunk()
                    if (
                        last_heading
                        and unit.kind != "heading"
                        and len(last_heading) + 2 + unit_len <= self.chunk_size
                    ):
                        current_units = [
                            _MarkdownBlock(kind="heading", content=last_heading)
                        ]
                        current_len = len(last_heading)
                    else:
                        current_units = []
                        current_len = 0

            sep_len = 2 if current_units else 0
            current_units.append(unit)
            current_len += sep_len + unit_len

        if current_units:
            flush_chunk()

        return chunks

    def split_documents(self, documents: list[ExtractedDocument]) -> list[Chunk]:
        """Tách nhiều ExtractedDocument thành danh sách Chunk."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.split_text(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks


text_chunker = TextChunker()
