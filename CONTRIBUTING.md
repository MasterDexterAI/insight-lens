# Contributing to Insight Lens

Insight Lens is an offline multimodal RAG app. It reads PDFs and images, indexes them locally with ChromaDB, and answers questions using Ollama and Florence-2. No document content or query ever leaves the machine it runs on. Keep that property intact in every change you make.

## Ground rules

1. Nothing gets sent off the local machine. No remote API calls for embeddings, captions, OCR, or answers. If a dependency you want to add makes a network call at runtime, it does not belong here.
2. Prefer small, focused pull requests over large ones. One feature or one fix per PR.
3. Match the existing code style before introducing a new one.
4. If you are not sure whether a change fits the project, open an issue first and describe what you want to do.

## Getting set up

1. Install Python 3.11.
2. Create and activate a virtual environment:
   ```
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. Install [Ollama](https://ollama.com) and pull the model referenced in `.env.example` (`llava:7b` by default):
   ```
   ollama pull llava:7b
   ```
4. Copy `.env.example` to `.env` and adjust values if needed.
5. Install the Python dependencies used by the app (streamlit, chromadb, ollama, transformers, torch, torchvision, pymupdf, pillow, python-dotenv). There is no committed lockfile yet; if you add a new dependency, note it in your PR description so the maintainer can pin it.
6. Start Ollama, then run the app:
   ```
   streamlit run app.py
   ```

## Project layout

```
app.py                     Streamlit UI and top-level app flow
config.py                  Paths, env vars, model and retrieval settings
src/ingestion/              PDF and image loading, normalization
src/models/                 Florence-2 (captioning, OCR) and Ollama client
src/rag/pipeline.py         Ingestion and question-answering pipeline
src/vectorstore/            ChromaDB collection setup and queries
```

Understand the pipeline before making changes: `app.py` calls into `src/rag/pipeline.py`, which calls the ingestion and model modules and writes to the vector store. Keep that direction of dependency. Do not import `app.py` from inside `src/`.

## Making changes

- Load heavy models once at module load time, not per request. See `src/models/florence_engine.py` for the pattern.
- Keep functions in `src/` free of Streamlit imports. UI code belongs in `app.py`.
- Use type hints on new functions, matching the style already in the codebase.
- Add a short docstring when a function's behavior is not obvious from its name and signature. Skip docstrings that just restate the function name.
- Comment only where the reasoning is not obvious from the code itself, such as the EXIF rotation note in `src/ingestion/image_processor.py`.
- If you change a config default in `config.py`, update `.env.example` to match.

## Testing your change

There is no automated test suite yet. Before opening a PR, verify manually:

1. Run `streamlit run app.py` with Ollama running.
2. Ingest at least one PDF and one image, and confirm the page count updates.
3. Ask a question that should be answerable from the ingested documents and confirm the answer and source image are correct.
4. If you touched ingestion or the vector store, delete `data/chromadb` and `data/chroma_db` and re-ingest from scratch to confirm nothing depends on stale state.

If you add meaningful logic that can be tested without a running Ollama instance or a GPU, consider adding a pytest test alongside it and mention in your PR that you started the test suite.

## Commit messages

Write commit messages in the imperative mood, for example "Add PDF page caching" rather than "Added PDF page caching" or "Adds PDF page caching." Keep the summary line under about 70 characters. Add a body if the change needs explanation that is not obvious from the diff.

## Pull requests

- Describe what changed and why, not just what files were touched.
- Mention any new dependency you introduced.
- Include the manual test steps you ran, since there is no CI test suite yet.
- Keep unrelated formatting changes out of functional PRs.

## Reporting issues

When filing an issue, include your OS, Python version, the Ollama model in use, and the exact error message or traceback. If the issue is about answer quality rather than a crash, include the document type (PDF or image) and the question you asked.
