from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SOURCE_URL = "https://jcip.net/jcip-sample.pdf"
DEFAULT_PDF = Path(__file__).resolve().parents[2] / "data/documents/jcip-sample.pdf"
TITLE = "Java Concurrency in Practice — sample chapter"
MIN_TEXT_PAGES = 20
PDF_SPACING_FIXES = {
    "ser ver": "server",
    "per iodic": "periodic",
    "Result-bear ing": "Result-bearing",
    "reser vations": "reservations",
}


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    section: str


def extract_pages(path: Path = DEFAULT_PDF) -> list[Page]:
    if not path.is_file():
        raise FileNotFoundError(f"PDF не найден: {path}. Поместите официальный sample PDF в data/documents/jcip-sample.pdf")
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    section = "Chapter 6: Task Execution"
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text(extraction_mode="layout") or ""
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        for index, line in enumerate(lines):
            if not re.match(r"^6(?:\.\d+){0,2}\s+", line):
                continue
            for broken, fixed in PDF_SPACING_FIXES.items():
                line = line.replace(broken, fixed)
            lines[index] = line
        lines = [line for line in lines if line and not re.fullmatch(r"\d+", line)]
        text = "\n".join(lines)
        if len(text) < 500:
            continue
        for line in lines:
            if match := re.match(r"^(6(?:\.\d+){0,2})\s+([A-Z][\w,;:()\-\s]{3,90})$", line):
                section = f"{match.group(1)} {match.group(2).strip()}"
                break
        pages.append(Page(number, text, section))
    if len(pages) < MIN_TEXT_PAGES:
        raise ValueError(f"Нужно не менее {MIN_TEXT_PAGES} страниц текста; в PDF найдено {len(pages)}")
    return pages
