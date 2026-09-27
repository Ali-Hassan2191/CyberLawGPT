# CyberLawGPT

CyberLawGPT is a Retrieval-Augmented Generation (RAG) application for asking questions about Pakistani cyber law using a legal PDF.

## Files

Only three files are required:

- `app.py` — Streamlit application, PDF downloading, extraction, chunking, embeddings, FAISS search, Groq generation, chat UI, and JSON export.
- `requirements.txt` — Python dependencies.
- `readme.md` — Setup and deployment instructions.

## Features

- Downloads the default Google Drive PDF at startup.
- Extracts text with `pypdf`.
- Splits text into overlapping chunks.
- Creates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores vectors in a FAISS inner-product index.
- Retrieves relevant passages for each question.
- Uses Groq model `openai/gpt-oss-120b`.
- UI controls for:
  - Technicality: Simple, Balanced, Technical, Legal-professional
  - Response size: Short, Medium, Detailed
  - Language: English, Urdu, Roman Urdu
  - Number of retrieved chunks
  - Minimum similarity score
  - Source-label display
- Allows a user to upload a replacement PDF.
- Exports the conversation as JSON.

## 1. Get the project

Keep the three files in one folder:

```text
CyberLawGPT/
├── app.py
├── requirements.txt
└── readme.md
```

## 2. Configure the Groq API key

Create a Groq API key from the Groq Console.

### Local or Colab

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Or in a terminal:

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Never commit your API key to GitHub.

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will download the default PDF and build embeddings on startup.

## 4. Run in Google Colab

Run this in a Colab cell:

```python
!pip install -r requirements.txt
```

Set your API key:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Start Streamlit:

```python
!streamlit run app.py &>/content/streamlit.log &
```

Expose the local Streamlit port using a tunneling method available in your Colab environment. For example, use a trusted tunnel provider and forward port `8501`.

## 5. Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload only:
   - `app.py`
   - `requirements.txt`
   - `readme.md`
3. Open Streamlit Community Cloud.
4. Create a new app and select `app.py`.
5. Add the secret:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

6. Deploy.

## 6. Google Drive PDF requirements

The default file ID is configured in `app.py`:

```python
DEFAULT_DRIVE_FILE_ID = "1HLE_EnH9rjxhqBUSIMUGj7gSTJTI8NI-"
```

The Drive file must be publicly accessible. If the default link does not work, use the sidebar PDF uploader or set a custom PDF URL.

### Optional environment variables

```bash
GROQ_API_KEY="your-key"
CYBERLAW_GOOGLE_DRIVE_FILE_ID="your-drive-file-id"
CYBERLAW_PDF_URL="https://example.com/file.pdf"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

`CYBERLAW_PDF_URL` takes priority over the default Drive URL.

## Legal and technical limitations

- The application answers from the supplied PDF, not from a live legal database.
- It does not automatically verify amendments, repealed provisions, court decisions, or the current legal status of the document.
- A scanned PDF without a text layer may fail extraction. Use an OCR-enabled PDF.
- FAISS is rebuilt when the PDF changes or when the app is restarted.
- The first startup can be slow because the embedding model must be downloaded.
- The app is designed for free-tier deployment, but resource limits on Colab and Streamlit Cloud may vary.
- Do not rely on generated answers as legal advice. Verify the relevant provision and consult a qualified Pakistani lawyer for actual legal matters.
