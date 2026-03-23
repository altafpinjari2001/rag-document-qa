# RAG Document Q&A

A production-style Retrieval-Augmented Generation (RAG) project for querying uploaded documents with context-aware answers.

[![CI](https://github.com/altafpinjari2001/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/altafpinjari2001/rag-document-qa/actions/workflows/ci.yml)

## Why this project matters
Recruiters and engineering teams evaluate whether a candidate can build **LLM systems that are reliable, testable, and maintainable**. This repo demonstrates exactly that.

## Core capabilities
- Document ingestion and preprocessing
- Chunking + embedding pipeline
- Retrieval + answer generation flow
- CI-backed quality checks

## Architecture (high level)
1. Load document(s)
2. Split and embed content
3. Store vectors
4. Retrieve relevant chunks for each query
5. Generate grounded answer

## Run locally
```bash
pip install -r requirements.txt
# run the project entry point (see app.py / docs)
```

## Quality signals
- GitHub Actions CI enabled and passing
- Lint/format/test checks integrated
- Modular source structure under `src/`

## Next improvements (recommended)
- Add latency + retrieval quality metrics in README
- Include a demo GIF/video
- Add benchmark queries and expected outputs