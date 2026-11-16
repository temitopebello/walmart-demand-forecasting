"""Privacy audit logging utility.

Writes a structured JSON line for every data access and transformation so the
Working Group (Module 1) can review who touched what, when. Satisfies the
Module 3 Privacy Audit Log requirement.
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path("reports/ethics/audit_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"


_logger = logging.getLogger("walmart.privacy_audit")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)


def log_event(
    event: str,
    artefact: str,
    rows: int | None = None,
    extra: dict | None = None,
) -> None:
    """Append one JSON line to the day's audit log.

    event   : short verb such as 'read', 'write', 'anonymise', 'validate'
    artefact: path or table name that was touched
    rows    : optional row count of the artefact at the time of the event
    extra   : any additional context for the Working Group
    """
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "actor": os.environ.get("USER", getpass.getuser()),
        "host": platform.node(),
        "event": event,
        "artefact": str(artefact),
        "rows": rows,
    }
    if extra:
        payload["extra"] = extra
    _logger.info(json.dumps(payload))


if __name__ == "__main__":
    # Demo entries
    log_event("read", "data/raw/train.csv", rows=478192)
    log_event("transform", "data/interim/train_clean.parquet", rows=477692,
              extra={"stage": "cleaning", "duplicates_dropped": 500})
    log_event("validate", "data/processed/walmart_master.parquet", rows=477692,
              extra={"expectations_met": 27})
    print(f"Wrote demo entries to {LOG_PATH}")
