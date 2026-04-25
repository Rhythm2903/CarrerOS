# CareerOS

CareerOS is an AI career copilot for resume evaluation, job-fit analysis, skill-gap detection, market intelligence, and targeted career planning.

It lets a user:
- upload a resume PDF
- paste a job description
- get a structured match analysis
- see which job skills are actually present versus missing
- understand strengths, concerns, and gap severity
- view India-focused salary guidance
- receive a 90-day roadmap
- chat with resume context

## What This Project Does

The system is built as a multi-agent workflow rather than a single LLM prompt.

Main pipeline:
1. Parse the resume into structured sections and career signals
2. Decompose the job description into role-specific requirements
3. Retrieve relevant resume context using embeddings and vector search
4. Match job skills against resume evidence
5. Analyze skill gaps and severity
6. Add market and salary intelligence
7. Build a personalized 90-day roadmap
8. Run a critique pass on output quality

## Key Features

- Resume upload and PDF text extraction
- Resume-aware job matching
- Explicit matched versus missing skill detection
- Role-specific skill handling from the JD
- Soft-skill evidence checks from the resume
- India-focused salary guidance in INR LPA
- Three-phase 90-day roadmap
- Resume-grounded chat assistant
- Session switching so a new uploaded resume replaces the old one without refreshing the app

## Architecture

Backend:
- FastAPI API
- OpenAI chat models for analysis
- OpenAI embeddings for retrieval
- FAISS for vector similarity search

Frontend:
- Streamlit app

Core backend modules:
- `backend/main.py`: API routes and session lifecycle
- `backend/services/resume_parser.py`: PDF parsing and resume section extraction
- `backend/rag_pipeline/rag_engine.py`: chunking, embeddings, retrieval
- `backend/agents/`: orchestrated analysis agents
- `backend/services/llm_service.py`: resume-grounded chat

## Agent Flow

Agents in `backend/agents/`:
- `ParserAgent`
- `JDAgent`
- `MatchAgent`
- `GapAgent`
- `MarketAgent`
- `RoadmapAgent`
- `CriticAgent`

Shared state is passed through `ReasoningContext`, so each stage builds on previous results instead of starting over.

## Output Quality Focus

The current version is tuned to:
- use JD-required skills as the source of truth for matching
- check those skills against real resume evidence
- avoid generic praise when the resume does not support it
- avoid inflated entry-level salary predictions
- show strengths and concerns more honestly

## Project Structure

```text
CareerOS/
|-- backend/
|   |-- agents/
|   |-- models/
|   |-- rag_pipeline/
|   |-- services/
|   |-- utils/
|   |-- config.py
|   `-- main.py
|-- frontend/
|   `-- app.py
|-- .env.example
|-- .gitignore
|-- Dockerfile
|-- README.md
`-- requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create environment file

```bash
copy .env.example .env
```

Then fill in your own API keys in `.env`.

### 3. Run the backend

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the frontend

```bash
streamlit run frontend/app.py
```

## Environment Variables

See `.env.example`.

Expected fields:
- `LLM_PROVIDER`
- `EMBEDDING_PROVIDER`
- `MODEL_TIER`
- `FALLBACK_PROVIDER`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `CHROMA_PERSIST_DIR`

Only `.env.example` should be committed. Never commit your real `.env`.

## API Endpoints

- `GET /`
- `GET /debug-env`
- `POST /upload_resume`
- `POST /analyze_job`
- `POST /chat`
- `DELETE /session/{session_id}`

## Notes

- The repository is set up so secrets stay out of source control.
- Resume analysis quality depends heavily on resume content and JD clarity.
- For best results, re-upload a resume after major parsing or matching logic changes.

## Suggested Future Improvements

- stronger synonym mapping for job-skill matching
- persistent DB instead of in-memory session storage
- analytics for token usage and latency
- recruiter-style downloadable report
- score explanations with evidence citations from the resume
