import re
from dataclasses import dataclass

from app.core.config import settings
from app.schemas.document import Chunk, ExtractedDocument

_HEADING_RE = re.compile(r"^#{1,6}\s+.+")


@dataclass
class _MarkdownBlock:
    kind: str  # "heading" | "table" | "text"
    content: str
    char_start: int | None = None
    char_end: int | None = None
    is_synthetic: bool = False


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

    @staticmethod
    def _iter_lines_with_offsets(text: str) -> list[tuple[str, int, int]]:
        """Tách từng dòng kèm vị trí ký tự [start, end) trên chuỗi text chuẩn hóa."""
        entries: list[tuple[str, int, int]] = []
        pos = 0
        for raw_line in text.splitlines(keepends=True):
            line_no_eol = raw_line.rstrip("\r\n")
            entries.append((line_no_eol, pos, pos + len(line_no_eol)))
            pos += len(raw_line)
        return entries

    def _parse_markdown_blocks(self, text: str) -> list[_MarkdownBlock]:
        """Tách văn bản Markdown thành các khối heading, table và text thường kèm offset."""
        line_entries = self._iter_lines_with_offsets(text)
        blocks: list[_MarkdownBlock] = []
        text_buf: list[tuple[str, int, int]] = []
        idx = 0

        def flush_text_buf() -> None:
            if not text_buf:
                return
            buf_start = text_buf[0][1]
            buf_end = text_buf[-1][2]
            text_buf.clear()

            segment = text[buf_start:buf_end]
            if not segment.strip():
                return

            para_pos = 0
            for sep_match in re.finditer(r"\n{2,}", segment):
                para_raw = segment[para_pos : sep_match.start()]
                _append_paragraph_block(para_raw, buf_start + para_pos)
                para_pos = sep_match.end()
            _append_paragraph_block(segment[para_pos:], buf_start + para_pos)

        def _append_paragraph_block(para_raw: str, base_offset: int) -> None:
            cleaned = para_raw.strip()
            if not cleaned:
                return
            leading = len(para_raw) - len(para_raw.lstrip())
            trailing = len(para_raw) - len(para_raw.rstrip())
            p_start = base_offset + leading
            p_end = base_offset + len(para_raw) - trailing
            kind = (
                "heading"
                if _HEADING_RE.match(cleaned) and "\n" not in cleaned
                else "text"
            )
            blocks.append(
                _MarkdownBlock(
                    kind=kind,
                    content=cleaned,
                    char_start=p_start,
                    char_end=p_end,
                )
            )

        while idx < len(line_entries):
            line, line_start, line_end = line_entries[idx]
            if self._is_table_line(line):
                table_entries: list[tuple[str, int, int]] = []
                while idx < len(line_entries) and self._is_table_line(line_entries[idx][0]):
                    table_entries.append(line_entries[idx])
                    idx += 1
                if len(table_entries) >= 2:
                    flush_text_buf()
                    table_content = "\n".join(ln.strip() for ln, _, _ in table_entries)
                    first_raw = table_entries[0][0]
                    last_raw = table_entries[-1][0]
                    t_start = table_entries[0][1] + (len(first_raw) - len(first_raw.lstrip()))
                    t_end = table_entries[-1][2] - (len(last_raw) - len(last_raw.rstrip()))
                    blocks.append(
                        _MarkdownBlock(
                            kind="table",
                            content=table_content,
                            char_start=t_start,
                            char_end=t_end,
                        )
                    )
                    continue
                text_buf.extend(table_entries)
                continue

            if _HEADING_RE.match(line.strip()):
                flush_text_buf()
                leading = len(line) - len(line.lstrip())
                trailing = len(line) - len(line.rstrip())
                blocks.append(
                    _MarkdownBlock(
                        kind="heading",
                        content=line.strip(),
                        char_start=line_start + leading,
                        char_end=line_end - trailing,
                    )
                )
                idx += 1
                continue

            text_buf.append((line, line_start, line_end))
            idx += 1

        flush_text_buf()
        return blocks

    def _split_oversized_table(
        self,
        table_text: str,
        block_start: int | None = None,
        block_end: int | None = None,
        source_text: str | None = None,
    ) -> list[_MarkdownBlock]:
        """Nếu bảng vượt quá chunk_size, tách theo dòng và lặp lại header bảng ở mỗi phần."""
        lines = [ln for ln in table_text.splitlines() if ln.strip()]
        if len(lines) <= 2 or len(table_text) <= self.chunk_size:
            return [
                _MarkdownBlock(
                    kind="table",
                    content=table_text,
                    char_start=block_start,
                    char_end=block_end,
                )
            ]

        has_separator = (
            len(lines) >= 2
            and set(lines[1].replace("|", "").replace("-", "").replace(":", "").strip())
            == set()
        )
        header_lines = lines[:2] if has_separator else lines[:1]
        body_lines = lines[len(header_lines) :]
        header_block = "\n".join(header_lines)

        row_groups: list[list[str]] = []
        current_rows: list[str] = []
        current_len = len(header_block)

        for row in body_lines:
            added_len = len(row) + 1
            if current_rows and current_len + added_len > self.chunk_size:
                row_groups.append(current_rows)
                current_rows = []
                current_len = len(header_block)
            current_rows.append(row)
            current_len += added_len

        if current_rows:
            row_groups.append(current_rows)

        units: list[_MarkdownBlock] = []
        cursor = block_start if block_start is not None else 0
        for group_idx, rows in enumerate(row_groups):
            sub_content = header_block + "\n" + "\n".join(rows)
            sub_start = block_start
            sub_end = block_end
            if source_text is not None and block_start is not None and block_end is not None:
                first_idx = source_text.find(rows[0], cursor, block_end)
                last_search_from = first_idx if first_idx != -1 else cursor
                last_idx = source_text.find(rows[-1], last_search_from, block_end)
                if first_idx != -1:
                    sub_start = block_start if group_idx == 0 else first_idx
                if last_idx != -1:
                    sub_end = last_idx + len(rows[-1])
                    cursor = sub_end
            units.append(
                _MarkdownBlock(
                    kind="table",
                    content=sub_content,
                    char_start=sub_start,
                    char_end=sub_end,
                )
            )

        return units

    def _split_text_by_separators(self, text: str, separators: list[str]) -> list[str]:
        """Tách đệ quy đoạn văn dài theo danh sách ký tự phân cách ưu tiên."""
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

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

    def _split_prose_with_overlap(
        self,
        text: str,
        block_start: int | None = None,
        block_end: int | None = None,
        source_text: str | None = None,
    ) -> list[_MarkdownBlock]:
        """Chia đoạn văn bản thường quá dài thành nhiều mảnh có gối đầu (overlap) kèm offset."""
        separators = ["\n", ". ", "; ", ", ", " "]
        raw_parts = self._split_text_by_separators(text, separators)

        part_spans: list[tuple[str, int | None, int | None]] = []
        if source_text is not None and block_start is not None and block_end is not None:
            cursor = block_start
            for part in raw_parts:
                idx = source_text.find(part, cursor, block_end)
                if idx != -1:
                    part_spans.append((part, idx, idx + len(part)))
                    cursor = idx + len(part)
                else:
                    part_spans.append((part, cursor, min(cursor + len(part), block_end)))
        else:
            part_spans = [(part, None, None) for part in raw_parts]

        pieces: list[_MarkdownBlock] = []
        current: list[tuple[str, int | None, int | None]] = []
        current_len = 0

        for part, p_start, p_end in part_spans:
            part_len = len(part)
            sep_len = 1 if current else 0
            if current and current_len + sep_len + part_len > self.chunk_size:
                joined = " ".join(item[0] for item in current).strip()
                if joined:
                    pieces.append(
                        _MarkdownBlock(
                            kind="text",
                            content=joined,
                            char_start=current[0][1],
                            char_end=current[-1][2],
                        )
                    )

                overlap_items: list[tuple[str, int | None, int | None]] = []
                overlap_len = 0
                for item in reversed(current):
                    extra = len(item[0]) + (1 if overlap_items else 0)
                    if overlap_len + extra <= self.chunk_overlap:
                        overlap_items.insert(0, item)
                        overlap_len += extra
                    else:
                        break
                current = overlap_items
                current_len = overlap_len

            current.append((part, p_start, p_end))
            current_len += part_len + (1 if len(current) > 1 else 0)

        if current:
            joined = " ".join(item[0] for item in current).strip()
            if joined:
                pieces.append(
                    _MarkdownBlock(
                        kind="text",
                        content=joined,
                        char_start=current[0][1],
                        char_end=current[-1][2],
                    )
                )

        return pieces

    def split_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Chia nhỏ một tài liệu Markdown thành danh sách Chunk mà vẫn giữ bảng biểu, ngữ cảnh & offset."""
        text = text.strip()
        if not text:
            return []

        metadata = metadata or {}
        blocks = self._parse_markdown_blocks(text)

        units: list[_MarkdownBlock] = []
        for block in blocks:
            if block.kind == "table":
                units.extend(
                    self._split_oversized_table(
                        block.content,
                        block_start=block.char_start,
                        block_end=block.char_end,
                        source_text=text,
                    )
                )
            elif (
                block.kind in {"text", "heading"}
                and len(block.content) > self.chunk_size
            ):
                units.extend(
                    self._split_prose_with_overlap(
                        block.content,
                        block_start=block.char_start,
                        block_end=block.char_end,
                        source_text=text,
                    )
                )
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
                source_units = [
                    u
                    for u in current_units
                    if not u.is_synthetic
                    and u.char_start is not None
                    and u.char_end is not None
                ]
                if not source_units:
                    source_units = [
                        u
                        for u in current_units
                        if u.char_start is not None and u.char_end is not None
                    ]
                char_start = (
                    min(u.char_start for u in source_units if u.char_start is not None)
                    if source_units
                    else None
                )
                char_end = (
                    max(u.char_end for u in source_units if u.char_end is not None)
                    if source_units
                    else None
                )
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_index=chunk_idx,
                        char_start=char_start,
                        char_end=char_end,
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
                            _MarkdownBlock(
                                kind="heading",
                                content=last_heading,
                                is_synthetic=True,
                            )
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
