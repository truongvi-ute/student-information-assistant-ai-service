from app.core.config import settings
from app.schemas.document import Chunk, ExtractedDocument


class TextChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")

    def _split_text_by_separators(self, text: str, separators: list[str]) -> list[str]:
        """Tách đệ quy văn bản dựa trên danh sách các ký tự phân cách ưu tiên."""
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep)
        result: list[str] = []

        for part in parts:
            if not part:
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                sub_parts = self._split_text_by_separators(part, remaining_seps)
                result.extend(sub_parts)

        return result

    def split_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Chia nhỏ một đoạn văn bản thành danh sách Chunk."""
        text = text.strip()
        if not text:
            return []

        metadata = metadata or {}
        # Danh sách dấu ngắt ưu tiên: đoạn văn -> xuống dòng -> câu -> từ
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        raw_parts = self._split_text_by_separators(text, separators)

        chunks: list[Chunk] = []
        current_chunk: list[str] = []
        current_len = 0
        chunk_idx = 0

        for part in raw_parts:
            part_len = len(part)
            if current_len + part_len > self.chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk).strip()
                if chunk_str:
                    chunk_meta = metadata.copy()
                    chunks.append(Chunk(content=chunk_str, chunk_index=chunk_idx, metadata=chunk_meta))
                    chunk_idx += 1

                # Tạo overlap từ cuối chunk trước
                overlap_chunk: list[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break

                current_chunk = overlap_chunk
                current_len = overlap_len

            current_chunk.append(part)
            current_len += part_len

        if current_chunk:
            final_str = " ".join(current_chunk).strip()
            if final_str:
                chunk_meta = metadata.copy()
                chunks.append(Chunk(content=final_str, chunk_index=chunk_idx, metadata=chunk_meta))

        return chunks

    def split_documents(self, documents: list[ExtractedDocument]) -> list[Chunk]:
        """Tách nhiều ExtractedDocument thành danh sách Chunk."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.split_text(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks


text_chunker = TextChunker()