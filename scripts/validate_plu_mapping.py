from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = PROJECT_ROOT / "data" / "plu_mapping.csv"
REQUIRED_COLUMNS = {
    "produce_class",
    "plu_code",
    "plu_type",
    "display_name",
    "notes",
    "source_status",
}
ALLOWED_TYPES = {"conventional", "organic", "varies"}


def validate_mapping(path: Path) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
            return errors

        for line_number, row in enumerate(reader, start=2):
            produce_class = row["produce_class"].strip()
            plu_code = row["plu_code"].strip()
            plu_type = row["plu_type"].strip()
            key = (produce_class, plu_code)

            if not produce_class:
                errors.append(f"Line {line_number}: produce_class is blank")
            if not plu_code.isdigit() or len(plu_code) not in {4, 5}:
                errors.append(f"Line {line_number}: plu_code must be 4 or 5 digits")
            if plu_type not in ALLOWED_TYPES:
                errors.append(f"Line {line_number}: plu_type must be one of {sorted(ALLOWED_TYPES)}")
            if key in seen:
                errors.append(f"Line {line_number}: duplicate produce_class/plu_code pair {key}")
            seen.add(key)

    return errors


def main() -> int:
    errors = validate_mapping(MAPPING_PATH)
    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"PLU mapping is valid: {MAPPING_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

