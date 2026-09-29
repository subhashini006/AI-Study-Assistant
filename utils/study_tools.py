"""Prompt-building functions for study features (summary, quiz, flashcards)."""
"""Prompt-building functions for study features (summary, quiz, flashcards)."""

import json
import re

from utils.llm import ask_claude


SUMMARY_SYSTEM_PROMPT = (
    "You are an expert study assistant. You turn study material into clear, "
    "well-organized summaries that help students learn efficiently. "
    "Always respond in Markdown."
)


def generate_summary(text: str) -> str:
    """
    Ask the AI to summarize the study material into headings and bullet points.
    Long PDFs are trimmed so we don't send too much text at once.
    """
    trimmed_text = text[:12000]

    prompt = f"""Summarize the study material below.

Rules:
- Organize the summary using Markdown headings (##) for each main topic.
- Under each heading, use short bullet points for key facts.
- Keep it clear and easy to revise from.
- Do not add information that isn't in the material.
- Aim for roughly 300-500 words total.

STUDY MATERIAL:
{trimmed_text}
"""
    return ask_claude(prompt, system=SUMMARY_SYSTEM_PROMPT, max_tokens=1500)


QA_SYSTEM_PROMPT = (
    "You are a study assistant that answers questions using ONLY the provided "
    "excerpts from the student's material. "
    "If the answer is not contained in the excerpts, say clearly: "
    "\"I couldn't find this in your uploaded material.\" "
    "Do not use outside knowledge. Keep answers concise and clear."
)


def answer_question(question: str, context_chunks: list[str]) -> str:
    """
    Ask the AI to answer a question using only the given context chunks
    (the most relevant parts of the PDF, found via semantic search).
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""Answer the student's question using ONLY the excerpts below.
If the excerpts don't contain the answer, say so clearly instead of guessing.

EXCERPTS FROM THE MATERIAL:
{context}

QUESTION:
{question}
"""
    return ask_claude(prompt, system=QA_SYSTEM_PROMPT, max_tokens=800)

import json
import re


QUIZ_SYSTEM_PROMPT = (
    "You are a quiz-generation engine for a study app. "
    "You ALWAYS respond with valid JSON only — no markdown, no code fences, "
    "no explanation text outside the JSON."
)


def generate_quiz(text: str, num_questions: int = 6) -> list[dict]:
    """
    Ask the AI to create multiple-choice questions from the study material.
    Returns a list of dicts, each shaped like:
    {
        "topic": "Short topic name",
        "question": "...",
        "options": ["A", "B", "C", "D"],
        "correct_index": 0,
        "explanation": "..."
    }
    """
    trimmed_text = text[:12000]

    prompt = f"""Create exactly {num_questions} multiple-choice questions based on
the study material below.

Rules:
- Each question must have exactly 4 options.
- Only one option is correct.
- Cover different parts/topics of the material, not just the start.
- Include a short "topic" label (2-4 words) for each question, describing
  what concept it tests (used later to suggest revision topics).
- Include a one-sentence "explanation" of why the correct answer is right.
- Respond with ONLY a JSON array, formatted exactly like this example:

[
  {{
    "topic": "Cell Division",
    "question": "What is the first phase of mitosis called?",
    "options": ["Prophase", "Metaphase", "Anaphase", "Telophase"],
    "correct_index": 0,
    "explanation": "Prophase is the first phase, where chromatin condenses into chromosomes."
  }}
]

STUDY MATERIAL:
{trimmed_text}
"""
    raw_response = ask_claude(prompt, system=QUIZ_SYSTEM_PROMPT, max_tokens=2500)
    return _parse_quiz_json(raw_response)


def _parse_quiz_json(raw_response: str) -> list[dict]:
    """
    Gemini sometimes wraps JSON in ```json fences even when told not to.
    This strips those before parsing, and validates the shape of each question.
    """
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        questions = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The AI didn't return a valid quiz format. Please try generating again."
        ) from error

    valid_questions = []
    for q in questions:
        if (
            isinstance(q, dict)
            and "question" in q
            and "options" in q
            and len(q.get("options", [])) == 4
            and "correct_index" in q
        ):
            valid_questions.append(q)

    if not valid_questions:
        raise ValueError("No valid questions were generated. Please try again.")

    return valid_questions