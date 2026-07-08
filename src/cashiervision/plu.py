from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PLURecord:
    produce_class: str
    plu_code: str
    plu_type: str
    display_name: str
    notes: str
    source_status: str


def normalize_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def load_plu_mapping(path: str | Path) -> dict[str, list[PLURecord]]:
    mapping_path = Path(path)
    records: dict[str, list[PLURecord]] = {}

    with mapping_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            produce_class = normalize_label(row["produce_class"])
            record = PLURecord(
                produce_class=produce_class,
                plu_code=row["plu_code"].strip(),
                plu_type=row["plu_type"].strip(),
                display_name=row["display_name"].strip(),
                notes=row["notes"].strip(),
                source_status=row["source_status"].strip(),
            )
            records.setdefault(produce_class, []).append(record)

    return records


def suggest_plu_codes(produce_class: str, mapping: dict[str, list[PLURecord]]) -> list[PLURecord]:
    return mapping.get(normalize_label(produce_class), [])


def format_plu_suggestions(records: list[PLURecord]) -> list[str]:
    if not records:
        return ["No curated PLU suggestion available. Ask cashier to verify manually."]

    return [
        f"{record.plu_code} ({record.plu_type}) - {record.notes}"
        for record in records
    ]

