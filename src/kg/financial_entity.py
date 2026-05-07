"""
Data model for financial entities extracted from chunks.

The entity model is deliberately minimal: text, canonical form, type, and
optional source. Char offsets are NOT tracked here because we don't need them
for graph construction (chunk-level co-occurrence is enough) and skipping them
keeps memory usage low at 27k+ chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    """Financial entity types tracked by the KG."""

    BANK = "BANK"          # HBL, MCB, UBL, Standard Chartered, ...
    PRODUCT = "PRODUCT"    # credit card, car loan, savings account, ...
    DOC_TYPE = "DOC_TYPE"  # KFS, SOBC, T&Cs, FAQ, ...
    SECTION = "SECTION"    # specific section name within a document
    MONEY = "MONEY"        # Rs. 5,000, PKR 50,000, $100, ...
    PERCENT = "PERCENT"    # 25.99%, 15-24% APR, ...
    FEE = "FEE"            # late payment fee, annual fee, processing fee, ...
    ORG = "ORG"            # Visa, Mastercard, State Bank, partner orgs (spaCy)


@dataclass(frozen=True)
class FinancialEntity:
    """
    A single entity occurrence in a chunk.

    Equality / hashing is on (canonical, entity_type) so duplicate occurrences
    of the same entity in different chunks compare equal — this is what makes
    Sets of entities behave correctly for co-occurrence accounting.
    """

    text: str
    canonical: str
    entity_type: EntityType

    @staticmethod
    def normalize(text: str) -> str:
        """Lowercase + collapse whitespace + strip punctuation we don't care about."""
        out = text.lower().strip()
        # collapse internal whitespace
        out = " ".join(out.split())
        # strip surrounding punctuation that doesn't carry meaning
        out = out.strip(".,;:()[]{}\"'`")
        return out

    @classmethod
    def make(cls, text: str, entity_type: EntityType) -> "FinancialEntity":
        return cls(text=text, canonical=cls.normalize(text), entity_type=entity_type)

    @property
    def key(self) -> tuple[str, str]:
        """Stable graph node key — (canonical, type.value)."""
        return (self.canonical, self.entity_type.value)
