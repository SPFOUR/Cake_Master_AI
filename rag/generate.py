"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality before wiring
  up an LLM.
- "llm": calls an LLM to write a grounded answer from the retrieved context.
  TODO: fill in your provider of choice (Anthropic, OpenAI, a local model via
  Ollama, etc). A minimal Anthropic example is sketched below — install the
  `anthropic` package and set the ANTHROPIC_API_KEY environment variable to use it.
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()


from .ingest import Chunk


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: \u201c{query}\u201d\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """TODO: replace this with a real LLM call once retrieval is working well."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "[LLM mode not configured] Set ANTHROPIC_API_KEY (or wire up your own "
            "provider in rag/generate.py) to enable grounded LLM answers. "
            "Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )

    context = "\n\n".join(f"Source: {c.doc_title}\n{c.text}" for c, _ in retrieved)
    prompt = (
        "Answer the question using ONLY the sources below. Cite the source title(s) "
        f"you used.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    )

#Uses Google's Gemini-2.5 
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive") -> str:
    if mode == "llm":
        return llm_answer(query, retrieved)
    return extractive_answer(query, retrieved)
