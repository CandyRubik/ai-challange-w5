from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from .corpus import Page, SOURCE_URL, TITLE


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False, verbose: bool = False) -> list[int]: ...
    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    strategy: str
    source: str
    title: str
    section: str
    page_start: int
    page_end: int
    text: str
    token_count: int

    def metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "strategy": self.strategy,
            "source": self.source, "title": self.title,
            "section": self.section, "page_start": self.page_start,
            "page_end": self.page_end, "text": self.text,
            "token_count": self.token_count,
        }


def _windows(tokens: list[int], size: int, overlap: int) -> list[list[int]]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Некорректные размер чанка или перекрытие")
    step = size - overlap
    return [tokens[start:start + size] for start in range(0, len(tokens), step)]


def fixed_chunks(pages: list[Page], tokenizer: Tokenizer, size: int = 300, overlap: int = 50) -> list[Chunk]:
    """Token windows over the whole corpus; page span is recovered from token offsets."""
    tokens: list[int] = []
    page_numbers: list[int] = []
    sections: list[str] = []
    for page in pages:
        page_tokens = tokenizer.encode(page.text, add_special_tokens=False, verbose=False)
        tokens.extend(page_tokens)
        page_numbers.extend([page.number] * len(page_tokens))
        sections.extend([page.section] * len(page_tokens))
    result: list[Chunk] = []
    step = size - overlap
    for start, window in ((i, tokens[i:i + size]) for i in range(0, len(tokens), step)):
        if not window or (start != 0 and len(window) <= overlap):
            continue
        result.append(Chunk(
            f"fixed-{len(result):04d}", "fixed", SOURCE_URL, TITLE,
            sections[start], page_numbers[start], page_numbers[start + len(window) - 1],
            tokenizer.decode(window).strip(), len(window),
        ))
    return result


HEADING = re.compile(r"^(6(?:\.\d+){0,2})\s+([A-Z][\w,;:()\-\s]{3,90})$")


def structured_chunks(pages: list[Page], tokenizer: Tokenizer, size: int = 300) -> list[Chunk]:
    """Respect headings and page boundaries, then cap long sections by tokens."""
    result: list[Chunk] = []
    section = pages[0].section if pages else ""
    for page in pages:
        blocks: list[tuple[str, str]] = []
        buffer: list[str] = []
        for line in page.text.splitlines():
            match = HEADING.match(line.strip())
            if match:
                if buffer:
                    blocks.append((section, "\n".join(buffer)))
                    buffer = []
                section = f"{match.group(1)} {match.group(2).strip()}"
            else:
                buffer.append(line)
        if buffer:
            blocks.append((section, "\n".join(buffer)))
        for block_section, text in blocks:
            tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            for window in _windows(tokens, size, 0):
                result.append(Chunk(
                    f"structure-{len(result):04d}", "structure", SOURCE_URL, TITLE,
                    block_section, page.number, page.number,
                    tokenizer.decode(window).strip(), len(window),
                ))
    return result
