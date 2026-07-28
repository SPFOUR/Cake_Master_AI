"""
Ingestion: load raw documents from disk and split them into overlapping chunks.

Upgrade path (for your final project):
- Add PDF/HTML/Markdown loaders (e.g. pypdf, BeautifulSoup) alongside plain .txt
- Swap the naive word-count chunker below for a sentence- or token-aware chunker
- Store document metadata (source URL, author, date) alongside each chunk
"""

import os
import re
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader
from .sources import SOURCE_MAP


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str
    section: str = None
    source_url: str = None
    citation: str = None



def load_documents(folder: str) -> List[dict]:
    """Load every .txt file in `folder` into {"title": ..., "text": ...} dicts."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        title = os.path.splitext(filename)[0].replace("_", " ").title()

        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            doc_type = "recipe"

        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            doc_type = "article"
        else:
            continue

        source_info = SOURCE_MAP.get(filename, {"url": None, "citation": "Unknown source"})
        docs.append({
            "title": title,
            "text": text,
            "doc_type": doc_type,
            "source_url": source_info["url"],
            "citation": source_info["citation"],
        })
    return docs

def clean_recipe_text(text: str) -> str:
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'^\S+\.(com|org|net)\S*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

#-------------------------

def chunk_recipe(text: str) -> List[tuple]:
    """Section-aware: split on Ingredients / Directions / Nutrition Facts.
    Returns a list of (section_name, section_text) pairs."""
    text = clean_recipe_text(text)
    section_pattern = r'\n(Ingredients|Directions|Nutrition Facts)\n'
    parts = re.split(section_pattern, text)
    result = []
    if parts[0].strip():
        result.append(("Overview", parts[0].strip()))
    for i in range(1, len(parts), 2):
        result.append((parts[i], parts[i + 1].strip()))
    return result

def chunk_prose(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[tuple]:
    """Sentence-aware with overlap, for history/article-style content.
    Returns a list of (section_name, section_text) pairs (section is always "Body")."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + sentences_per_chunk
        chunks.append(("Body", " ".join(sentences[start:end])))
        if end >= len(sentences):
            break
        start = end - overlap_sentences
    return chunks

def build_chunk_records(docs: List[dict]) -> List[Chunk]:
    records = []
    for doc in docs:
        if doc["doc_type"] == "recipe":
            pieces = chunk_recipe(doc["text"])
        else:
            pieces = chunk_prose(doc["text"])
        for i, (section, chunk_text_piece) in enumerate(pieces):
            records.append(Chunk(
                chunk_id=f"{doc['title']}::{i}",
                doc_title=doc["title"],
                text=chunk_text_piece,
                section=section,
                source_url=doc.get("source_url"),
                citation=doc.get("citation"),
            ))
    return records
#-------------------------