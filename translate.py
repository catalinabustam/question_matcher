"""Translation client for the source CSV questions, using the DeepL API.

Isolated in its own module so the rest of the application doesn't depend
directly on the DeepL SDK (easy to swap for another provider).
"""
from typing import List, Optional
from dataclasses import replace

import deepl

from models import Question


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


def translate_questions(translator: DeepLTranslator, questions: List[Question],
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
