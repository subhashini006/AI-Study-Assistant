"""Handles all communication with the Google Gemini API."""

import os
import time

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")

_configured = False


def _ensure_configured():
    """Set up the Gemini client once, using the key from .env."""
    global _configured
    if not _configured:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "paste_your_key_here":
            raise ValueError(
                "GOOGLE_API_KEY is missing. Add it to your .env file."
            )
        genai.configure(api_key=api_key)
        _configured = True


def ask_claude(prompt: str, system: str = "", max_tokens: int = 1500) -> str:
    """
    Send a prompt to Gemini and return its text reply.
    Automatically retries a couple of times if we hit a rate limit (429).

    prompt: the actual question/instruction.
    system: optional instructions that set the AI's role/behaviour.
    """
    _ensure_configured()

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system if system else "You are a helpful study assistant.",
    )

    generation_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)

    max_retries = 3
    wait_seconds = 15

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except ResourceExhausted as error:
            if attempt == max_retries:
                raise RuntimeError(
                    "You've hit the free-tier rate limit for this model. "
                    "Wait about a minute and try again, or check your quota at "
                    "https://ai.dev/rate-limit."
                ) from error
            time.sleep(wait_seconds)

    return ""  # should never reach here