# -*- coding: utf-8 -*-
"""EPUB/TXT ingestion that preserves narrative and chapter order."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from .models import Chapter, Passage, stable_id


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;\.])")
_SPACE = re.compile(r"[ \t\u00a0]+")


def source_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_book_path(book_id: int) -> Optional[str]:
    from ... import calibre_db, config

    book = calibre_db.get_book(int(book_id))
    if not book:
        return None
    priority = {"epub": 0, "kepub": 1, "txt": 2}
    candidates = sorted(
        (item for item in book.data if item.format.lower() in priority),
        key=lambda item: priority[item.format.lower()],
    )
    for item in candidates:
        path = os.path.normpath(
            os.path.join(config.config_calibre_dir, book.path, item.name + "." + item.format.lower())
        )
        if os.path.isfile(path):
            return path
    return None


def load_chapters(path: str) -> List[Chapter]:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".txt":
        return _load_text(path)
    if extension in {".epub", ".kepub"}:
        return _load_epub(path)
    raise ValueError("Unsupported novel format: {}".format(extension or "unknown"))


def _load_text(path: str) -> List[Chapter]:
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        text = unicodedata.normalize("NFKC", stream.read())
    paragraphs = _normalize_paragraphs(text.splitlines())
    if not paragraphs:
        return []
    return [Chapter(chapter_id=stable_id("chapter", path, 0), index=0, title="正文", paragraphs=paragraphs)]


def _load_epub(path: str) -> List[Chapter]:
    book = epub.read_epub(path)
    items = {item.get_id(): item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
    ordered = []
    for spine_entry in book.spine:
        item_id = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        item = items.get(item_id)
        if item is not None:
            ordered.append(item)
    if not ordered:
        ordered = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    chapters: List[Chapter] = []
    seen = set()
    for item in ordered:
        identity = item.get_id() or item.get_name()
        if identity in seen:
            continue
        seen.add(identity)
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for node in soup(["script", "style", "nav", "noscript"]):
            node.decompose()
        title = _chapter_title(soup, len(chapters))
        block_texts = []
        for node in soup.find_all(["p", "blockquote", "li", "h1", "h2", "h3", "h4"]):
            value = node.get_text(" ", strip=True)
            if value:
                block_texts.append(value)
        if not block_texts:
            block_texts = soup.get_text("\n", strip=True).splitlines()
        paragraphs = _normalize_paragraphs(block_texts)
        if not paragraphs:
            continue
        chapters.append(
            Chapter(
                chapter_id=str(identity),
                index=len(chapters),
                title=title,
                paragraphs=paragraphs,
            )
        )
    return chapters


def _normalize_paragraphs(values: Iterable[str]) -> List[str]:
    paragraphs = []
    for value in values:
        normalized = unicodedata.normalize("NFKC", str(value)).replace("\r", " ").replace("\n", " ")
        normalized = _SPACE.sub(" ", normalized).strip()
        if normalized and (not paragraphs or normalized != paragraphs[-1]):
            paragraphs.append(normalized)
    return paragraphs


def _chapter_title(soup: BeautifulSoup, fallback_index: int) -> str:
    for name in ("h1", "h2", "h3", "title"):
        node = soup.find(name)
        if node and node.get_text(strip=True):
            return _SPACE.sub(" ", node.get_text(" ", strip=True))[:240]
    return "Chapter {}".format(fallback_index + 1)


def _split_long_paragraph(text: str, target_chars: int) -> List[str]:
    if len(text) <= target_chars:
        return [text]
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    if len(sentences) <= 1:
        return [text[start : start + target_chars] for start in range(0, len(text), target_chars)]
    result: List[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > target_chars:
            result.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        result.append(current)
    return result


def segment_chapters(
    book_id: int,
    chapters: Sequence[Chapter],
    target_chars: int = 2400,
    overlap_paragraphs: int = 1,
) -> List[Passage]:
    """Build stable passages without crossing chapter boundaries."""
    passages: List[Passage] = []
    order_id = 0
    for chapter in chapters:
        units: List[Tuple[int, str]] = []
        for paragraph_index, paragraph in enumerate(chapter.paragraphs):
            units.extend((paragraph_index, value) for value in _split_long_paragraph(paragraph, target_chars))
        cursor = 0
        while cursor < len(units):
            end = cursor
            selected: List[str] = []
            size = 0
            while end < len(units):
                candidate = units[end][1]
                extra = len(candidate) + (2 if selected else 0)
                if selected and size + extra > target_chars:
                    break
                selected.append(candidate)
                size += extra
                end += 1
            text = "\n\n".join(selected).strip()
            if text:
                passages.append(
                    Passage.create(
                        book_id=int(book_id),
                        order_id=order_id,
                        chapter=chapter,
                        text=text,
                        paragraph_start=units[cursor][0],
                        paragraph_end=units[end - 1][0],
                    )
                )
                order_id += 1
            if end >= len(units):
                break
            overlap_start = end
            remaining = max(0, int(overlap_paragraphs))
            previous_paragraph = None
            while overlap_start > cursor and remaining:
                overlap_start -= 1
                paragraph_index = units[overlap_start][0]
                if paragraph_index != previous_paragraph:
                    remaining -= 1
                    previous_paragraph = paragraph_index
            cursor = max(cursor + 1, overlap_start)
    return passages
