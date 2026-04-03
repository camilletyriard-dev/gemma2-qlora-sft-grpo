"""
Multi-tier quality filter for QA pairs used in synthetic Yoda dataset generation.

Pipeline:
  Tier 1 — Fast regex checks (word count, character set, question-in-answer, etc.)
  Tier 2 — Structural checks (duplicate sentences, list/enumeration patterns)
  Tier 3 — Optional NLP checks (passive voice rate via spaCy)

Usage:
    from src.data.qa_filter import QAFilter
    f = QAFilter(use_passive_filter=True)
    clean = [(q, a) for q, a in pairs if f.is_clean(q, a)]
    print(f.stats())
"""

import re
from typing import Dict, Optional


class QAFilter:
    """Stateful multi-tier filter for question-answer pair quality.

    Tracks rejection reasons across calls for diagnostic reporting.

    Args:
        use_passive_filter: If True, reject answers with >50% passive sentences
                            (requires spaCy 'en_core_web_sm' model).
        min_words: Minimum word count per sentence.
        max_words: Maximum word count per sentence.
    """

    def __init__(
        self,
        use_passive_filter: bool = True,
        min_words: int = 4,
        max_words: int = 15,
    ) -> None:
        self.use_passive_filter = use_passive_filter
        self.min_words = min_words
        self.max_words = max_words
        self._stats: Dict[str, int] = {"total": 0, "passed": 0}
        self._reasons: Dict[str, int] = {}
        self._nlp = None  # lazy-loaded

    def _reject(self, reason: str) -> bool:
        self._reasons[reason] = self._reasons.get(reason, 0) + 1
        return False

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["ner"])
        return self._nlp

    def is_clean(self, question: str, answer: str) -> bool:
        """Return True if the QA pair passes all quality checks.

        Args:
            question: The question string.
            answer: The answer string to evaluate.

        Returns:
            True if the pair passes; False otherwise (reason recorded internally).
        """
        self._stats["total"] += 1

        if not answer or not question:
            return self._reject("empty")

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip())
                     if s.strip()]
        if not sentences:
            return self._reject("no_sentences")

        # ── Tier 1: Fast regex checks ──────────────────────────────────────
        for sent in sentences:
            wc = len(sent.split())
            if wc < self.min_words:
                return self._reject("too_short")
            if wc > self.max_words:
                return self._reject("too_long")

        if re.search(r"[^a-zA-Z .,!?\'()\-]", answer):
            return self._reject("bad_chars")

        if "?" in answer:
            return self._reject("question_in_answer")

        if re.search(r"\btranslat[a-z]*\b", question, re.IGNORECASE):
            return self._reject("translation_question")

        digits = sum(c.isdigit() for c in answer)
        if len(answer) > 0 and digits / len(answer) > 0.20:
            return self._reject("number_density")

        ordinal_markers = len(re.findall(
            r"\b(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|\d+[).]|[a-z][).]|;)",
            answer, re.IGNORECASE,
        ))
        if ordinal_markers >= 2:
            return self._reject("list_enumeration")

        # ── Tier 2: Structural checks ──────────────────────────────────────
        sent_lower = [s.lower().strip().rstrip(".!?") for s in sentences]
        if len(sent_lower) != len(set(sent_lower)):
            return self._reject("duplicate_sentences")

        # ── Tier 3: Optional NLP check (passive voice) ────────────────────
        if self.use_passive_filter:
            nlp = self._get_nlp()
            passive_count = 0
            for sent in sentences:
                doc = nlp(sent)
                if any(tok.dep_ == "nsubjpass" for tok in doc):
                    passive_count += 1
            if len(sentences) > 0 and passive_count / len(sentences) > 0.5:
                return self._reject("passive_voice")

        self._stats["passed"] += 1
        return True

    def stats(self) -> str:
        """Return a formatted summary of filter statistics."""
        total  = self._stats["total"]
        passed = self._stats["passed"]
        rate   = passed / total * 100 if total > 0 else 0.0
        lines  = [
            f"Filter stats: {passed}/{total} passed ({rate:.1f}%)",
            "Rejection reasons:",
        ]
        for reason, count in sorted(self._reasons.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason:30s}: {count}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats = {"total": 0, "passed": 0}
        self._reasons = {}