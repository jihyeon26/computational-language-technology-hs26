from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class ContractError(ValueError):
    """Raised when an upstream WP2 delivery cannot satisfy the configured contract."""


@dataclass
class LoadedTable:
    name: str
    path: Path
    frame: pd.DataFrame
    resolved_columns: dict[str, str]


def read_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if "contract_version" not in contract or "tables" not in contract:
        raise ContractError("Contract must define 'contract_version' and 'tables'.")
    return contract


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ContractError(f"Unsupported table format: {path}")


def _resolve_file(input_dir: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = input_dir / candidate
        if path.exists():
            return path
    return None


def _canonicalize_columns(
    frame: pd.DataFrame,
    column_contract: dict[str, list[str]],
    required_fields: list[str],
    derived_fields: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    result = frame.copy()
    resolved: dict[str, str] = {}
    warnings: list[str] = []

    for canonical, aliases in column_contract.items():
        ordered = list(dict.fromkeys([canonical, *aliases]))
        matches = [column for column in ordered if column in result.columns]
        if not matches:
            continue
        selected = matches[0]
        resolved[canonical] = selected
        if len(matches) > 1:
            warnings.append(
                f"Field '{canonical}' matched multiple columns {matches}; using '{selected}'."
            )
        if selected != canonical:
            result = result.rename(columns={selected: canonical})

    for field, derivation in (derived_fields or {}).items():
        if field in result.columns and not derivation.get("overwrite", False):
            continue
        operation = derivation.get("operation")
        if operation != "concat":
            raise ContractError(
                f"Unsupported derived-field operation for '{field}': {operation!r}"
            )
        source_fields = derivation.get("fields", [])
        missing_sources = [source for source in source_fields if source not in result.columns]
        if missing_sources:
            raise ContractError(
                f"Cannot derive '{field}'; missing source fields: {missing_sources}"
            )
        separator = str(derivation.get("separator", ":"))

        def combine(row: pd.Series) -> Any:
            parts: list[str] = []
            for source in source_fields:
                value = row[source]
                if pd.isna(value) or str(value).strip() == "":
                    return pd.NA
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                parts.append(str(value).strip())
            return separator.join(parts)

        result[field] = result.apply(combine, axis=1)
        resolved[field] = f"derived:{operation}({', '.join(source_fields)})"

    missing = [field for field in required_fields if field not in result.columns]
    if missing:
        raise ContractError(f"Missing required canonical fields: {missing}")
    return result, resolved, warnings


def load_contract_tables(
    input_dir: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    tables: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {
        "contract_version": contract["contract_version"],
        "input_dir": str(input_dir.resolve()),
        "tables": {},
        "warnings": [],
        "errors": [],
    }

    for name, specification in contract["tables"].items():
        required_table = bool(specification.get("required", False))
        candidates = specification.get("files", [])
        path = _resolve_file(input_dir, candidates)
        if path is None:
            message = f"Table '{name}' not found. Tried: {candidates}"
            if required_table:
                report["errors"].append(message)
            else:
                report["warnings"].append(message)
            continue

        try:
            raw = _read_table(path)
            canonical, resolved, warnings = _canonicalize_columns(
                raw,
                specification.get("columns", {}),
                specification.get("required_fields", []),
                specification.get("derived_fields", {}),
            )
        except Exception as exc:
            report["errors"].append(f"Table '{name}' failed contract adaptation: {exc}")
            continue

        tables[name] = canonical
        report["tables"][name] = {
            "path": str(path.resolve()),
            "rows": int(len(canonical)),
            "source_columns": list(raw.columns),
            "canonical_columns": list(canonical.columns),
            "resolved_columns": resolved,
        }
        report["warnings"].extend(f"{name}: {warning}" for warning in warnings)

    report["passed"] = not report["errors"]
    if report["errors"]:
        raise ContractError("; ".join(report["errors"]))
    return tables, report


def find_optional_json(input_dir: Path, candidates: list[str]) -> tuple[Path | None, dict[str, Any] | None]:
    path = _resolve_file(input_dir, candidates)
    if path is None:
        return None, None
    return path, json.loads(path.read_text(encoding="utf-8"))
