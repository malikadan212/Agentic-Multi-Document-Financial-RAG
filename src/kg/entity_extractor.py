"""
Financial entity extractor.

Extracts entities at three layers, each independently optional and resilient:

1. Metadata layer (free, instant)
   The corpus's chunk metadata already tags each chunk with bank_name,
   document_type, and product_type. We treat these as pre-extracted
   high-confidence entities.

2. Regex layer (~5ms / chunk)
   Curated patterns for MONEY, PERCENT, and FEE phrases — domain-specific
   patterns that spaCy's generic NER misses (e.g. "Rs. 1,500" or
   "late payment fee").

3. spaCy layer (optional, ~30ms / chunk)
   en_core_web_sm for additional ORG entities (Visa, Mastercard, etc.).
   Falls back gracefully if spaCy or the model is unavailable.

The extractor is intentionally idempotent: feeding the same chunk twice
produces the same entity set.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPACY_AVAILABLE = False

from .financial_entity import EntityType, FinancialEntity

logger = logging.getLogger(__name__)


# ----- Regex patterns ---------------------------------------------------------
# Money: Rs. 5,000 / PKR 50000 / $100 / USD 250 / 1,500 rupees
_MONEY_RE = re.compile(
    r"""
    (?<![\w.])                            # left boundary
    (?:
        (?:Rs\.?|PKR|USD|\$|€|£|AED|SAR)  # currency prefix
        \s?\d{1,3}(?:,\d{3})*(?:\.\d+)?
      |
        \d{1,3}(?:,\d{3})*(?:\.\d+)?      # number first
        \s?(?:rupees?|dollars?|paisa)     # currency suffix
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Percent: 25.99% / 15.5 % / 24% APR
_PERCENT_RE = re.compile(
    r"\b\d{1,3}(?:\.\d+)?\s?%(?:\s?(?:APR|p\.a\.|per\s+annum))?",
    re.IGNORECASE,
)

# Fees: known multi-word phrases that name a charge. We match the whole phrase
# so the canonical form is meaningful (not just the word "fee").
_FEE_PHRASES = [
    r"annual\s+fee",
    r"joining\s+fee",
    r"processing\s+fee",
    r"late\s+payment\s+fee",
    r"late\s+fee",
    r"over[\s-]?limit\s+fee",
    r"cash\s+advance\s+fee",
    r"foreign\s+(?:currency|transaction)\s+fee",
    r"balance\s+transfer\s+fee",
    r"prepayment\s+(?:fee|charge|penalty)",
    r"renewal\s+fee",
    r"replacement\s+(?:card\s+)?fee",
    r"sms\s+(?:alert\s+)?(?:fee|charge)",
    r"service\s+charge",
    r"finance\s+charge",
    r"markup\s+(?:rate|charge)",
]
_FEE_RE = re.compile(
    r"\b(?:" + "|".join(_FEE_PHRASES) + r")\b",
    re.IGNORECASE,
)

# Generic financial product/category mentions in body text — useful when the
# chunk's product_type metadata is "other" but the body clearly names one.
_PRODUCT_PHRASES = [
    r"credit\s+card",
    r"debit\s+card",
    r"prepaid\s+card",
    r"(?:car|auto)\s+loan",
    r"home\s+loan",
    r"personal\s+loan",
    r"student\s+loan",
    r"savings\s+account",
    r"current\s+account",
    r"fixed\s+deposit",
    r"term\s+deposit",
    r"running\s+finance",
    r"overdraft",
    r"cheque\s+book",
    r"locker(?:\s+facility)?",
]
_PRODUCT_RE = re.compile(
    r"\b(?:" + "|".join(_PRODUCT_PHRASES) + r")\b",
    re.IGNORECASE,
)

# Document-type abbreviations that appear in body text (the metadata also
# carries this; we capture body mentions for cross-document linking).
_DOC_TYPE_RE = re.compile(
    r"\b(?:KFS|SOBC|T&Cs?|FAQ|MITC|Schedule\s+of\s+(?:Bank\s+)?Charges)\b",
    re.IGNORECASE,
)


def _spacy_load(model: str = "en_core_web_sm"):
    """Load spaCy model; return (nlp, ok)."""
    if not _SPACY_AVAILABLE:
        return None, False
    try:
        return spacy.load(model, disable=["lemmatizer", "tagger"]), True
    except OSError:
        logger.warning(
            "spaCy model '%s' not installed. KG ORG extraction disabled. "
            "Install with: python -m spacy download %s",
            model,
            model,
        )
        return None, False


class FinancialEntityExtractor:
    """
    Extracts financial entities from a chunk's text + metadata.

    Designed for both offline graph building (called per chunk over 27k chunks)
    and online query-time use (called once per user query).
    """

    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self._nlp = None
        self._spacy_ok = False
        if use_spacy:
            self._nlp, self._spacy_ok = _spacy_load(spacy_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[FinancialEntity]:
        """
        Extract entities from a chunk.

        Args:
            text: The chunk's text content.
            metadata: Optional metadata dict (chunk's `metadata` field). Pre-tagged
                fields (bank_name / doc_name, document_type, product_type, section)
                are converted to high-confidence entities for free.

        Returns:
            Deduplicated list of FinancialEntity objects.
        """
        seen: dict[tuple[str, str], FinancialEntity] = {}

        if metadata:
            for ent in self._from_metadata(metadata):
                seen.setdefault(ent.key, ent)

        if text:
            for ent in self._from_regex(text):
                seen.setdefault(ent.key, ent)
            if self._spacy_ok:
                for ent in self._from_spacy(text):
                    seen.setdefault(ent.key, ent)

        return list(seen.values())

    def extract_from_query(self, query: str) -> list[FinancialEntity]:
        """Extract entities from a user query (no metadata available)."""
        return self.extract(query, metadata=None)

    # ------------------------------------------------------------------
    # Internal extractors
    # ------------------------------------------------------------------
    @staticmethod
    def _from_metadata(meta: dict) -> Iterable[FinancialEntity]:
        # `doc_name` is mapped from `bank_name` upstream (see chunk_loader.py).
        bank = meta.get("doc_name") or meta.get("bank_name")
        if bank and bank.lower() not in ("unknown", ""):
            yield FinancialEntity.make(str(bank), EntityType.BANK)

        doc_type = meta.get("document_type")
        if doc_type and doc_type.lower() not in ("other", "unknown", ""):
            yield FinancialEntity.make(str(doc_type), EntityType.DOC_TYPE)

        product = meta.get("product_type")
        if product and product.lower() not in ("other", "unknown", ""):
            yield FinancialEntity.make(str(product), EntityType.PRODUCT)

        section = meta.get("section")
        if section:
            yield FinancialEntity.make(str(section), EntityType.SECTION)

    @staticmethod
    def _from_regex(text: str) -> Iterable[FinancialEntity]:
        for m in _MONEY_RE.finditer(text):
            yield FinancialEntity.make(m.group(0), EntityType.MONEY)
        for m in _PERCENT_RE.finditer(text):
            yield FinancialEntity.make(m.group(0), EntityType.PERCENT)
        for m in _FEE_RE.finditer(text):
            yield FinancialEntity.make(m.group(0), EntityType.FEE)
        for m in _PRODUCT_RE.finditer(text):
            yield FinancialEntity.make(m.group(0), EntityType.PRODUCT)
        for m in _DOC_TYPE_RE.finditer(text):
            yield FinancialEntity.make(m.group(0), EntityType.DOC_TYPE)

    def _from_spacy(self, text: str) -> Iterable[FinancialEntity]:
        # spaCy's NER is reliable for ORG / proper-noun mentions.
        try:
            doc = self._nlp(text[:4000])  # cap length per chunk for speed
        except Exception as exc:  # noqa: BLE001
            logger.debug("spaCy extraction failed for chunk: %s", exc)
            return
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Filter overly short or junk matches
                surface = ent.text.strip()
                if len(surface) < 3 or surface.isdigit():
                    continue
                yield FinancialEntity.make(surface, EntityType.ORG)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def status(self) -> dict:
        """Report which layers are active. Useful for the build script."""
        return {
            "metadata_layer": True,
            "regex_layer": True,
            "spacy_available": _SPACY_AVAILABLE,
            "spacy_model_loaded": self._spacy_ok,
            "spacy_model_name": self.spacy_model if self._spacy_ok else None,
        }
