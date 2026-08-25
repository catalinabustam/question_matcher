"""Translation client for the source CSV questions, using the DeepL API.

Isolated in its own module so the rest of the application doesn't depend
directly on the DeepL SDK (easy to swap for another provider).
"""
from typing import List, Optional, Protocol
from dataclasses import replace
import requests

import deepl

from models import Question


class Translator(Protocol):
    """Protocol for translation providers."""

    def translate_many(self, texts: List[str], source_lang: Optional[str] = None) -> List[str]:
        ...


class DeepLTranslator:
    """Simple wrapper around the DeepL API with an in-memory cache per text."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("A DeepL API key is required.")
        self._client = deepl.Translator(api_key)
        self._target_lang = 'EN-US'
        self._cache: dict = {}

    def translate_many(self, texts: List[str], source_lang: Optional[str] = None) -> List[str]:
        """Translate a list of texts while preserving order.

        Reuses translations already computed (cache) and batches the rest
        into a single API call to minimize the number of requests.
        """
        results: List[Optional[str]] = [None] * len(texts)
        pending_indices, pending_texts = [], []

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
            translations = self._client.translate_text(
                pending_texts, source_lang=source_lang, target_lang=self._target_lang)
            for idx, original_text, translated in zip(pending_indices, pending_texts, translations):
                results[idx] = translated.text
                self._cache[self._cache_key(original_text, source_lang)] = translated.text

        return results

    def _cache_key(self, text: str, source_lang: Optional[str]) -> str:
        return f"{source_lang or 'auto'}::{self._target_lang}::{text}"


class OllamaTranslator:
    """Wrapper around the Ollama API for local translation."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip('/')
        self._target_lang = 'EN-US'
        self._cache: dict = {}

    @staticmethod
    def get_available_models(base_url: str = "http://localhost:11434") -> List[str]:
        """Fetch available models from Ollama."""
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception:
            return []

    def translate_many(self, texts: List[str], source_lang: Optional[str] = None) -> List[str]:
        """Translate a list of texts using Ollama."""
        results: List[Optional[str]] = [None] * len(texts)
        pending_indices, pending_texts = [], []

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
            for idx, original_text, translated_text in zip(pending_indices, pending_texts, translated):
                results[idx] = translated_text
                self._cache[self._cache_key(original_text, source_lang)] = translated_text

        return results

    def _translate_batch(self, texts: List[str], source_lang: Optional[str]) -> List[str]:
        """Translate a batch of texts using Ollama."""
        # Build prompt for translation
        source_lang_str = source_lang or "auto-detect"
        prompt = (
            f"Translate the following texts from {source_lang_str} to English. "
            f"Return only the translations, one per line, preserving the order:\n\n"
        )
        for i, text in enumerate(texts):
            prompt += f"{i+1}. {text}\n"

        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "").strip()

            # Parse the numbered response
            translations = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and line[0].isdigit() and '. ' in line:
                    translations.append(line.split('. ', 1)[1])
                elif line:
                    translations.append(line)

            # Ensure we have the right number of translations
            while len(translations) < len(texts):
                translations.append("")

            return translations[:len(texts)]
        except Exception as exc:
            raise RuntimeError(f"Ollama translation failed: {exc}")

    def _cache_key(self, text: str, source_lang: Optional[str]) -> str:
        return f"{source_lang or 'auto'}::{self._target_lang}::{self._model}::{text}"


def translate_questions(translator: Translator, questions: List[Question],
                        source_lang: Optional[str] = None) -> List[Question]:
    """Return a new list of `Question` with translated fields updated, keeping all original attributes intact."""

    translated_sections = translator.translate_many([q.section for q in questions], source_lang=source_lang)
    translated_questions = translator.translate_many([q.question for q in questions], source_lang=source_lang)
    translated_definitions = translator.translate_many([q.definition for q in questions], source_lang=source_lang)

    return [
        replace(
            q,
            translated_section=translated_section,
            translated_question=translated_question,
            translated_definition=translated_definition,
        )
        for q, translated_section, translated_question, translated_definition in zip(
            questions, translated_sections, translated_questions, translated_definitions
        )
    ]
