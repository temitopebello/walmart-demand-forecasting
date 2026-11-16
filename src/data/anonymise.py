"""PII anonymisation utility.

The Kaggle Walmart Store Sales dataset contains no PII. This module is a
defensive design artefact for the production path: if operational data
(associate rosters, customer-level transactions, loyalty records) is ever
brought in, this is the function that redacts it before it reaches
data/interim/.

A small fixture of Faker-generated rows is used to demonstrate the function
runs. Each anonymisation is logged via src.utils.audit_log.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from faker import Faker

from src.utils.audit_log import log_event


PII_PATTERNS = {
    "email":       re.compile(r"[\w\.-]+@[\w\.-]+"),
    "phone":       re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def hash_token(value: str, salt: str = "walmart-pipeline") -> str:
    """One-way hash so joins survive without exposing the raw identifier."""
    return hashlib.sha256((salt + "::" + value).encode("utf-8")).hexdigest()[:16]


def redact_pii_series(series: pd.Series) -> pd.Series:
    """Replace any email, phone, or SSN-like substrings with [REDACTED]."""
    def _scrub(s):
        if not isinstance(s, str):
            return s
        for pat in PII_PATTERNS.values():
            s = pat.sub("[REDACTED]", s)
        return s
    return series.map(_scrub)


def anonymise_frame(df: pd.DataFrame, id_cols: list[str], text_cols: list[str]) -> pd.DataFrame:
    """Hash the given id columns and redact PII patterns from the given text columns."""
    out = df.copy()
    for col in id_cols:
        if col in out.columns:
            out[col] = out[col].astype(str).map(hash_token)
    for col in text_cols:
        if col in out.columns:
            out[col] = redact_pii_series(out[col])
    log_event("anonymise", "demo_fixture", rows=len(out),
              extra={"id_cols": id_cols, "text_cols": text_cols})
    return out


def _make_demo_fixture(n: int = 10) -> pd.DataFrame:
    """Generate a tiny PII fixture so the script has something to run against."""
    fake = Faker()
    Faker.seed(42)
    return pd.DataFrame({
        "associate_id":   [fake.uuid4() for _ in range(n)],
        "name":           [fake.name() for _ in range(n)],
        "email":          [fake.email() for _ in range(n)],
        "phone":          [fake.phone_number() for _ in range(n)],
        "notes":          ["Contact at " + fake.email() + " or " + fake.phone_number() for _ in range(n)],
        "store_id":       [1 + i % 45 for i in range(n)],
    })


def main() -> None:
    fixture = _make_demo_fixture()
    print("Fixture BEFORE anonymisation (first 3 rows):")
    print(fixture.head(3).to_string(index=False))
    anon = anonymise_frame(fixture, id_cols=["associate_id", "name", "email", "phone"],
                           text_cols=["notes"])
    print()
    print("Fixture AFTER anonymisation (first 3 rows):")
    print(anon.head(3).to_string(index=False))
    out_path = Path("reports/ethics/anonymisation_demo.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anon.to_csv(out_path, index=False)
    print(f"\nSaved anonymised fixture to {out_path}")


if __name__ == "__main__":
    main()
