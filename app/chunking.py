from __future__ import annotations

from dataclasses import dataclass

from app.loaders import DocumentSection

BOUNDARY_CHARS = "。！？!?；;\n"


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
    file_type: str
    page: int | None
    paragraph: int | None
    chunk_index: int
    start_char: int
    end_char: int


def chunk_sections(
    sections: list[DocumentSection],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    chunks: list[TextChunk] = []
    for section in sections:
        text = section.text.strip()
        if not text:
            continue

        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length:
                end = _preferred_end(text, start, end, chunk_size)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source=section.source,
                        file_type=section.file_type,
                        page=section.page,
                        paragraph=section.paragraph,
                        chunk_index=len(chunks),
                        start_char=start,
                        end_char=end,
                    )
                )

            if end >= text_length:
                break
            start = max(end - overlap, start + 1)

    return chunks


def _preferred_end(text: str, start: int, window_end: int, chunk_size: int) -> int:
    search_start = max(start + chunk_size // 2, start + 1)
    boundary = max(text.rfind(char, search_start, window_end) for char in BOUNDARY_CHARS)
    return boundary + 1 if boundary >= 0 else window_end
