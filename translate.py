from dataclasses import replace
from typing import Callable, List, Optional

import deepl
import requests

from models import Question

# Given (texts, source_lang), returns the translated texts in the same order.
BatchTranslateFn = Callable[[List[str], Optional[str]], List[str]]

_TARGET_LANG = "EN-US"


class Translator:
  
    def __init__(self, translate_batch: BatchTranslateFn, cache_key_prefix: str = ""):

        self._translate_batch = translate_batch
        self._cache_key_prefix = cache_key_prefix
        self._cache: dict[str, str] = {}

    def translate_many(self, texts: List[str], source_lang: Optional[str] = None) -> List[str]:

        results: List[Optional[str]] = [None] * len(texts)
        pending_indices: List[int] = []
        pending_texts: List[str] = []

        for i, text in enumerate(texts):
            if not text:
                results[i] = text
                continue
            key = self._cache_key(text, source_lang)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                pending_indices.append(i)
                pending_texts.append(text)

        if pending_texts:
            translated = self._translate_batch(pending_texts, source_lang)
            for idx, original_text, translated_text in zip(
                pending_indices, pending_texts, translated
            ):
                results[idx] = translated_text
                self._cache[self._cache_key(original_text, source_lang)] = translated_text

        return results

    def translate_questions(
        self, questions: List[Question], source_lang: Optional[str] = None
    ) -> List[Question]:
        """Return new `Question`s with translated fields filled in, keeping
        every other attribute intact."""
        translated_sections = self.translate_many(
            [q.section or "" for q in questions], source_lang
        )
        translated_questions = self.translate_many(
            [q.question for q in questions], source_lang
        )
        translated_definitions = self.translate_many(
            [q.definition or "" for q in questions], source_lang
        )
        translated_options = self.translate_many(
                    [q.options or "" for q in questions], source_lang
                )

        return [
            replace(
                q,
                translated_section=section,
                translated_question=question,
                translated_definition=definition,
                translated_options=options,
            )
            for q, section, question, definition, options in zip(
                questions, translated_sections, translated_questions, translated_definitions, translated_options
            )
        ]

    def _cache_key(self, text: str, source_lang: Optional[str]) -> str:
        return f"{self._cache_key_prefix}::{source_lang or 'auto'}::{text}"


# --------------------------------------------------------------------------- #
# Provider-specific factories — each only defines `translate_batch`.
# --------------------------------------------------------------------------- #


def deepl_translator(api_key: str) -> Translator:
    """Build a `Translator` backed by the DeepL API."""
    if not api_key:
        raise ValueError("A DeepL API key is required.")
    client = deepl.Translator(api_key)

    def translate_batch(texts: List[str], source_lang: Optional[str]) -> List[str]:
        translated = client.translate_text(
            texts, source_lang=source_lang, target_lang=_TARGET_LANG
        )
        return [t.text for t in translated]

    return Translator(translate_batch, cache_key_prefix=f"deepl::{_TARGET_LANG}")


def ollama_translator(model: str, base_url: str = "http://localhost:11434") -> Translator:

    def translate_batch(texts: List[str], source_lang: Optional[str]) -> List[str]:
        return _ollama_translate_batch(texts, source_lang, model, base_url)

    return Translator(
        translate_batch, cache_key_prefix=f"ollama::{model}::{_TARGET_LANG}"
    )


def _ollama_translate_batch(
    texts: List[str], source_lang: Optional[str], model: str, base_url: str
) -> List[str]:
    """Translate each text to English via Ollama, one request at a time."""
    return [_ollama_translate_one(text, source_lang, model, base_url) for text in texts]


def _ollama_translate_one(
    text: str, source_lang: Optional[str], model: str, base_url: str
) -> str:
    """Translate a single text to English via Ollama."""
    source_lang_str = source_lang or "auto-detect"
    prompt = (
        f"Translate the following text from {source_lang_str} to English. "
        "Respond with ONLY the translation, no explanation:\n\n"
        f"{text}"
    )
    return _ollama_generate(prompt, model, base_url).strip()


def _ollama_generate(prompt: str, model: str, base_url: str) -> str:
    """Call Ollama's `/api/generate` and return the raw generated text."""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama translation failed: {exc}") from exc
