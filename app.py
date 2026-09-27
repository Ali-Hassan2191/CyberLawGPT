import os
import io
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import requests
import numpy as np
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


APP_NAME = "CyberLawGPT"
MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_DRIVE_FILE_ID = "1Alve7SH7pEtCyK3o-9B_uem7ATGQ8Dda"
CACHE_DIR = Path(".cyberlaw_cache")
CACHE_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .source-box {border: 1px solid #d9d9d9; border-radius: 10px; padding: 10px; margin: 6px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def drive_download_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def download_pdf() -> bytes:
    uploaded = st.session_state.get("uploaded_pdf")
    if uploaded is not None:
        return uploaded.getvalue()

    pdf_url = os.getenv("CYBERLAW_PDF_URL", "").strip()
    file_id = os.getenv("CYBERLAW_GOOGLE_DRIVE_FILE_ID", DEFAULT_DRIVE_FILE_ID)

    if not pdf_url:
        pdf_url = drive_download_url(file_id)

    response = requests.get(
        pdf_url,
        timeout=90,
        headers={"User-Agent": "CyberLawGPT/1.0"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()

    if not response.content.startswith(b"%PDF") and "pdf" not in content_type:
        raise ValueError(
            "The PDF download did not return a PDF. "
            "Check that the Google Drive file is publicly accessible."
        )

    return response.content


def extract_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append({"page": page_number, "text": text})

    if not pages:
        raise ValueError(
            "No selectable text was extracted. This PDF may be scanned. "
            "Use an OCR-enabled PDF."
        )

    return pages


def make_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end]).strip()

            if text:
                chunks.append(
                    {
                        "text": text,
                        "page": page["page"],
                        "chunk_id": len(chunks) + 1,
                    }
                )

            if end >= len(words):
                break

            start = max(end - overlap, start + 1)

    return chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def build_index(pdf_hash: str, pdf_bytes: bytes):
    pages = extract_pages(pdf_bytes)
    chunks = make_chunks(pages)

    if not chunks:
        raise ValueError("No usable text chunks were created from the PDF.")

    model = load_embedding_model()
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks, len(pages), len(embeddings)


def retrieve(
    question: str,
    index,
    chunks: List[Dict[str, Any]],
    top_k: int,
    min_score: float,
) -> List[Dict[str, Any]]:
    model = load_embedding_model()
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    scores, ids = index.search(query_vector, min(top_k, len(chunks)))
    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        if score >= min_score:
            results.append(item)

    return results


def make_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "NO_RELEVANT_CONTEXT_FOUND"

    return "\n\n".join(
        f"[Source {i} | PDF page {item['page']} | similarity {item['score']:.3f}]\n"
        f"{item['text']}"
        for i, item in enumerate(results, start=1)
    )


def answer_question(
    question: str,
    context: str,
    technicality: str,
    response_size: str,
    language: str,
    include_citations: bool,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit secrets or your environment."
        )

    client = Groq(api_key=api_key)

    system_prompt = f"""
You are CyberLawGPT, a retrieval-augmented assistant focused on Pakistani cyber law.

Your job is to answer questions using ONLY the supplied PDF context.
The PDF is the source of truth for this answer. Do not invent sections,
penalties, dates, case law, amendments, procedures, or legal conclusions.

Rules:
1. First determine whether the question is related to Pakistani cyber law.
2. If it is unrelated, politely say that you only answer questions related
   to Pakistani cyber law.
3. If the supplied context does not support the answer, say:
   "I could not find sufficient support for this answer in the uploaded PDF."
   Then suggest what exact provision or document the user should provide.
4. Never fabricate a section number or quote.
5. Explain uncertainty and distinguish the PDF's text from general explanation.
6. Do not claim that the PDF is the latest law unless the PDF itself establishes that.
7. If the user asks for legal advice, provide general legal information only
   and recommend consulting a qualified lawyer for a real case.
8. Answer in {language}.
9. Technicality level: {technicality}.
10. Response size: {response_size}.
11. Use headings and bullet points when useful.
12. When citations are enabled, cite source labels exactly as [Source 1],
    [Source 2], etc., based only on the supplied context.

Retrieved PDF context:
{context}
"""

    user_prompt = f"User question:\n{question}"

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1800 if response_size == "Short" else (
            3500 if response_size == "Medium" else 6000
        ),
    )

    return completion.choices[0].message.content.strip()


def initialize_pdf():
    if "pdf_bytes" in st.session_state and "index" in st.session_state:
        return

    with st.spinner("Downloading PDF and creating embeddings..."):
        pdf_bytes = download_pdf()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        index, chunks, page_count, chunk_count = build_index(pdf_hash, pdf_bytes)

        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_hash = pdf_hash
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.page_count = page_count
        st.session_state.chunk_count = chunk_count


def render_sources(results: List[Dict[str, Any]]):
    if not results:
        return

    with st.expander("Retrieved legal sources"):
        for i, item in enumerate(results, start=1):
            st.markdown(
                f"""
                <div class="source-box">
                <b>[Source {i}]</b> — PDF page {item['page']} —
                similarity {item['score']:.3f}<br>
                {item['text']}
                </div>
                """,
                unsafe_allow_html=True,
            )


st.title("⚖️ CyberLawGPT")
st.caption(
    "Ask questions about Pakistani cyber law using the uploaded legal PDF. "
    "Answers are grounded in retrieved document passages."
)

with st.sidebar:
    st.header("Settings")

    technicality = st.select_slider(
        "Technicality level",
        options=["Simple", "Balanced", "Technical", "Legal-professional"],
        value="Balanced",
    )

    response_size = st.select_slider(
        "Response size",
        options=["Short", "Medium", "Detailed"],
        value="Medium",
    )

    language = st.selectbox(
        "Response language",
        ["English", "Urdu", "Roman Urdu"],
        index=0,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=12,
        value=5,
        help="More chunks provide broader context but may increase token usage.",
    )

    min_score = st.slider(
        "Minimum similarity score",
        min_value=0.0,
        max_value=0.9,
        value=0.15,
        step=0.05,
    )

    include_citations = st.checkbox("Include source labels", value=True)

    st.divider()
    st.subheader("Optional PDF override")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF to use instead of the default Drive PDF",
        type=["pdf"],
    )

    if uploaded_pdf is not None:
        if st.button("Use uploaded PDF"):
            st.session_state.uploaded_pdf = uploaded_pdf
            for key in [
                "pdf_bytes", "pdf_hash", "index", "chunks",
                "page_count", "chunk_count",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

    if st.button("Rebuild embeddings"):
        for key in [
            "pdf_bytes", "pdf_hash", "index", "chunks",
            "page_count", "chunk_count",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.markdown("**Model**")
    st.code(MODEL_NAME, language="text")

try:
    initialize_pdf()
except Exception as exc:
    st.error(f"Startup error: {exc}")
    st.info(
        "Make sure the Google Drive file is public and GROQ_API_KEY is configured. "
        "You can also upload the PDF from the sidebar."
    )
    st.stop()

with st.sidebar:
    st.success("PDF indexed")
    st.metric("PDF pages", st.session_state.page_count)
    st.metric("Text chunks", st.session_state.chunk_count)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input(
    "Ask a question about Pakistani cyber law, sections, offences, or penalties..."
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the legal PDF and preparing an answer..."):
            try:
                results = retrieve(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k,
                    min_score,
                )
                context = make_context(results)

                answer = answer_question(
                    question=question,
                    context=context,
                    technicality=technicality,
                    response_size=response_size,
                    language=language,
                    include_citations=include_citations,
                )

                st.markdown(answer)
                render_sources(results)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results,
                    }
                )
            except Exception as exc:
                error_message = f"Error: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

st.divider()
st.caption(
    "Legal information only — verify the current law and consult a qualified "
    "Pakistani lawyer for case-specific advice."
)

if st.session_state.messages:
    export_data = {
        "app": APP_NAME,
        "model": MODEL_NAME,
        "pdf_sha256": st.session_state.get("pdf_hash"),
        "messages": st.session_state.messages,
    }
    st.download_button(
        "Download chat as JSON",
        data=json.dumps(export_data, ensure_ascii=False, indent=2),
        file_name="cyberlawgpt_chat.json",
        mime="application/json",
    )
