import hashlib
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..ai_db import BookChunk, get_session
from .. import calibre_db, config


@dataclass
class Chunk:
    """
    Lightweight chunk representation before it is persisted to the database.
    """

    text: str
    book_id: int
    chunk_index: int
    chapter_index: int
    chapter_title: str
    chapter_id: str
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    def __post_init__(self):
        cleaned = self.text.strip()
        if not cleaned:
            raise ValueError("Chunk text cannot be empty")
        self.text = cleaned
        self.word_count = len(cleaned.split())
        self.char_count = len(cleaned)

    @property
    def chunk_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_model(self) -> BookChunk:
        return BookChunk(
            chunk_id=self.chunk_id,
            book_id=self.book_id,
            chunk_index=self.chunk_index,
            chunk_hash=self.chunk_hash,
            text=self.text,
            word_count=self.word_count,
            char_count=self.char_count,
            chapter_id=self.chapter_id,
            chapter_index=self.chapter_index,
            chapter_title=self.chapter_title,
            previous_chunk_id=self.previous_chunk_id,
            next_chunk_id=self.next_chunk_id,
        )


class BookParser:
    """
    Parse an EPUB file into discrete chunks and persist them into the chunk table.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        )

    def chunk_book(self, book_id: int, file_path: Optional[str] = None) -> List[str]:
        """
        Given a book id (and optionally an explicit file path), split the EPUB contents
        into chunks and persist them in the database.

        Returns:
            List of chunk_ids that were created.
        """
        epub_path = file_path or self._resolve_book_path(book_id)
        if not epub_path:
            raise ValueError("Book file path is required to chunk content")

        book = epub.read_epub(epub_path)
        chapter_docs = self._extract_chapters(book)
        chunks = self._chunk_chapters(book_id, chapter_docs)

        if not chunks:
            return []

        session = get_session()
        try:
            self._delete_existing_chunks(session, book_id)
            for chunk in chunks:
                session.add(chunk.to_model())
            session.commit()
            return [chunk.chunk_id for chunk in chunks]
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _extract_chapters(self, book) -> List[dict]:
        chapters = []
        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text("\n", strip=True)
            if not text:
                continue
            chapters.append(
                {
                    "id": item.get_id() or item.get_name(),
                    "index": len(chapters),
                    "title": self._guess_title(soup, len(chapters)),
                    "text": text,
                }
            )
        return chapters

    def _chunk_chapters(self, book_id: int, chapters: List[dict]) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_index = 0
        for chapter in chapters:
            pieces = self.text_splitter.split_text(chapter["text"])
            for piece in pieces:
                cleaned = piece.strip()
                if not cleaned:
                    continue
                chunks.append(
                    Chunk(
                        text=cleaned,
                        book_id=book_id,
                        chunk_index=chunk_index,
                        chapter_index=chapter["index"],
                        chapter_title=chapter["title"],
                        chapter_id=str(chapter["id"]),
                    )
                )
                chunk_index += 1

        for idx, chunk in enumerate(chunks):
            if idx > 0:
                chunk.previous_chunk_id = chunks[idx - 1].chunk_id
            if idx < len(chunks) - 1:
                chunk.next_chunk_id = chunks[idx + 1].chunk_id
        return chunks

    @staticmethod
    def _delete_existing_chunks(session, book_id: int) -> None:
        session.query(BookChunk).filter(BookChunk.book_id == book_id).delete()

    @staticmethod
    def _guess_title(soup: BeautifulSoup, fallback_index: int) -> str:
        for selector in ("h1", "h2", "h3", "title"):
            node = soup.find(selector)
            if node and node.get_text(strip=True):
                return node.get_text(strip=True)
        return f"Chapter {fallback_index + 1}"

    @staticmethod
    def _resolve_book_path(book_id: int) -> Optional[str]:
        """
        Placeholder for mapping a Calibre book_id to the actual file path.
        Implementers should override or extend this method.
        """
        book = calibre_db.get_book(book_id)

        if not book:
            return None

        readable_formats = ["epub", "kepub", "txt"]
        book_format = None
        book_data = None

        for data in book.data:
            if data.format.lower() in readable_formats:
                book_format = data.format.lower()
                book_data = data
                break

        if not book_format:
            return None

        file_path = os.path.normpath(
            os.path.join(config.config_calibre_dir, book.path, book_data.name + "." + book_format)
        )

        if not os.path.exists(file_path):
            return None

        return file_path
