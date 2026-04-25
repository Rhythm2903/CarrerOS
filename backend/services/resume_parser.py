import PyPDF2
import re
from typing import Optional


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes."""
    import io
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def parse_resume(text: str) -> dict:
    """Parse resume text into structured sections."""
    sections = {
        "name": _extract_name(text),
        "email": _extract_email(text),
        "phone": _extract_phone(text),
        "skills": _extract_skills(text),
        "experience": _extract_section(text, ["experience", "work experience", "employment"]),
        "education": _extract_section(text, ["education", "academic"]),
        "projects": _extract_section(text, ["projects", "personal projects"]),
        "certifications": _extract_section(text, ["certifications", "certificates", "courses"]),
        "summary": _extract_section(text, ["summary", "objective", "profile"]),
        "raw_text": text
    }
    return sections


def _extract_email(text: str) -> Optional[str]:
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None


def _extract_phone(text: str) -> Optional[str]:
    match = re.search(r'(\+?\d[\d\s\-().]{7,}\d)', text)
    return match.group(0).strip() if match else None


def _extract_name(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line.split()) <= 5 and not any(c in first_line for c in ['@', '|', '/']):
            return first_line
    return None


def _extract_skills(text: str) -> list[str]:
    skills_keywords = [
        "python", "java", "javascript", "typescript", "react", "node", "fastapi",
        "django", "flask", "sql", "postgresql", "mysql", "mongodb", "redis",
        "docker", "kubernetes", "aws", "gcp", "azure", "git", "linux",
        "machine learning", "deep learning", "nlp", "computer vision",
        "llm", "llms", "large language model", "large language models", "generative ai",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
        "langchain", "langgraph", "llamaindex", "openai", "rag", "vector database",
        "chromadb", "faiss", "pinecone", "weaviate", "qdrant", "hugging face",
        "huggingface", "transformers", "prompt engineering", "embeddings",
        "openai api", "gemini api", "ai agents", "agentic ai", "semantic search", "streamlit",
        "rest api", "graphql", "microservices", "agile", "scrum",
        "html", "css", "tailwind", "bootstrap", "figma", "ci/cd"
    ]
    found = []
    text_lower = text.lower()
    for skill in skills_keywords:
        if skill in text_lower:
            found.append(skill)
    return list(set(found))


def _extract_section(text: str, section_names: list[str]) -> str:
    lines = text.split('\n')
    section_content = []
    in_section = False

    section_headers = [
        "experience", "education", "skills", "projects", "certifications",
        "summary", "objective", "profile", "achievements", "awards",
        "publications", "references", "languages", "interests", "work"
    ]

    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        if any(name in line_lower for name in section_names) and len(line.strip()) < 50:
            in_section = True
            continue
        if in_section:
            if any(header in line_lower for header in section_headers if header not in section_names) \
                    and len(line.strip()) < 50 and line.strip():
                break
            section_content.append(line)

    return '\n'.join(section_content).strip()
