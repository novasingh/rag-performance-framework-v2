from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from main import COLLECTIONS, RQ1_CONDITIONS

REQUIRED_FIELDS = [
    "id",
    "title",
    "text",
    "source_name",
    "source_type",
    "domain",
    "publication_date",
    "freshness_score",
    "freshness_label",
    "url",
]

CONDITION_REQUIRED_FIELDS = REQUIRED_FIELDS + [
    "condition_id",
    "condition_domain",
    "condition_freshness_window",
    "condition_source_configuration",
]


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_records(records: List[Dict]) -> Dict:
    missing_fields = 0
    empty_text = 0
    domain_counts = Counter()
    source_type_counts = Counter()
    freshness_counts = Counter()

    for rec in records:
        domain_counts[rec.get("domain", "unknown")] += 1
        source_type_counts[rec.get("source_type", "unknown")] += 1
        freshness_counts[rec.get("freshness_label", "unknown")] += 1

        for field in REQUIRED_FIELDS:
            if field not in rec:
                missing_fields += 1
                break

        if not (rec.get("text") or "").strip():
            empty_text += 1

    return {
        "total_records": len(records),
        "missing_field_records": missing_fields,
        "empty_text_records": empty_text,
        "domain_counts": dict(domain_counts),
        "source_type_counts": dict(source_type_counts),
        "freshness_counts": dict(freshness_counts),
    }


def validate_condition_records(records: List[Dict]) -> Dict:
    missing_fields = 0
    empty_text = 0
    condition_counts = Counter()

    for rec in records:
        condition_counts[rec.get("condition_id", "unknown")] += 1
        for field in CONDITION_REQUIRED_FIELDS:
            if field not in rec:
                missing_fields += 1
                break
        if not (rec.get("text") or "").strip():
            empty_text += 1

    return {
        "total_records": len(records),
        "missing_field_records": missing_fields,
        "empty_text_records": empty_text,
        "condition_counts": dict(condition_counts),
    }


def check_collection_outputs(base_dir: Path) -> Dict:
    final_dir = base_dir / "final"
    result: Dict[str, Dict[str, int | bool]] = {}

    for cfg in COLLECTIONS:
        json_file = final_dir / cfg.name / "final_documents.json"
        csv_file = final_dir / cfg.name / "final_documents.csv"
        count = 0
        if json_file.exists():
            try:
                count = len(load_json(json_file))
            except Exception:
                count = -1

        result[cfg.name] = {
            "json_exists": json_file.exists(),
            "csv_exists": csv_file.exists(),
            "final_count": count,
        }

    return result


def check_summary(base_dir: Path) -> Dict:
    summary_path = base_dir / "summary.json"
    if not summary_path.exists():
        return {"exists": False}

    try:
        payload = load_json(summary_path)
        summary = payload[0] if isinstance(payload, list) and payload else {}
        return {
            "exists": True,
            "target_min": summary.get("target_min"),
            "target_max": summary.get("target_max"),
            "news_provider": summary.get("news_provider"),
            "rq1_query_alignment": summary.get("rq1_query_alignment"),
            "total_documents": summary.get("total_documents"),
            "collections": summary.get("collections", {}),
            "conditions": summary.get("conditions", {}),
            "expected_conditions": summary.get("expected_conditions"),
        }
    except Exception as err:
        return {"exists": True, "error": str(err)}


def check_condition_outputs(base_dir: Path) -> Dict:
    conditions_dir = base_dir / "final" / "conditions"
    result: Dict[str, Dict[str, int | bool]] = {}

    for condition in RQ1_CONDITIONS:
        condition_id = condition.condition_id
        json_file = conditions_dir / condition_id / "condition_documents.json"
        csv_file = conditions_dir / condition_id / "condition_documents.csv"
        count = 0
        if json_file.exists():
            try:
                count = len(load_json(json_file))
            except Exception:
                count = -1

        result[condition_id] = {
            "json_exists": json_file.exists(),
            "csv_exists": csv_file.exists(),
            "final_count": count,
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify build_rag_dataset outputs for RQ1.")
    parser.add_argument("--output-dir", default="rag_dataset", help="Base output folder where dataset is stored.")
    parser.add_argument("--condition-min", type=int, default=200, help="Minimum required records per condition.")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    combined_path = base_dir / "final" / "combined_dataset.json"
    records = load_json(combined_path)
    combined_conditions_path = base_dir / "final" / "conditions" / "combined_conditions_dataset.json"
    condition_records = load_json(combined_conditions_path) if combined_conditions_path.exists() else []

    summary = validate_records(records)
    summary["condition_dataset"] = validate_condition_records(condition_records)
    summary["collections"] = check_collection_outputs(base_dir)
    summary["conditions"] = check_condition_outputs(base_dir)
    summary["summary_file"] = check_summary(base_dir)

    condition_min = max(1, args.condition_min)
    condition_shortfalls = {
        condition_id: max(0, condition_min - int(payload.get("final_count") or 0))
        for condition_id, payload in summary["conditions"].items()
    }
    conditions_below_min = {
        condition_id: shortfall
        for condition_id, shortfall in condition_shortfalls.items()
        if shortfall > 0
    }
    summary["condition_min_docs"] = condition_min
    summary["condition_shortfalls"] = condition_shortfalls
    summary["conditions_meet_minimum"] = not conditions_below_min

    report_path = base_dir / "final" / "verification_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if conditions_below_min:
        print(f"Conditions below minimum={condition_min}: {json.dumps(conditions_below_min, indent=2)}")
    else:
        print(f"All conditions meet minimum={condition_min}")
    print(f"Verification report saved to: {report_path}")


if __name__ == "__main__":
    main()
