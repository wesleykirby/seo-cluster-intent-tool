import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from cluster.pipeline import DEFAULT_LEXICON_PATH, load_lexicon, merge_lexicon


def _normalize_brand_updates(raw) -> Dict[str, str]:
    updates: Dict[str, str] = {}
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, list):
        iterable = []
        for entry in raw:
            if isinstance(entry, dict):
                alias = entry.get("alias")
                canonical = entry.get("canonical", alias)
                if alias:
                    iterable.append((alias, canonical))
            else:
                iterable.append((entry, entry))
    elif raw is None:
        iterable = []
    else:
        iterable = [(raw, raw)]

    for alias, canonical in iterable:
        if alias is None:
            continue
        alias_str = str(alias).strip()
        if not alias_str:
            continue
        canonical_str = str(canonical if canonical is not None else alias).strip()
        if not canonical_str:
            canonical_str = alias_str
        updates[alias_str] = canonical_str
    return updates


def _normalize_list_updates(raw: Iterable) -> List[str]:
    values: List[str] = []
    if raw is None:
        return values
    if isinstance(raw, dict):
        raw = raw.values()
    for item in raw:
        if item is None:
            continue
        val = str(item).strip()
        if val:
            values.append(val)
    return values


def _write_lexicon(path: Path, brands: Dict[str, str], modifiers: List[str], regions: List[str]) -> None:
    ordered_brands = {k.lower(): brands[k].lower() for k in sorted(brands)}
    ordered_modifiers = sorted({m.lower() for m in modifiers})
    ordered_regions = sorted({r.lower() for r in regions})

    data = {
        "BRANDS": ordered_brands,
        "MODIFIERS": ordered_modifiers,
        "REGIONS": ordered_regions,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def update_lexicon(lexicon_path: Path, tokens_path: Path, dry_run: bool = False) -> Dict:
    if not tokens_path.exists():
        raise FileNotFoundError(f"Token file not found: {tokens_path}")

    tokens = json.loads(tokens_path.read_text())
    brand_updates = _normalize_brand_updates(tokens.get("BRANDS"))
    modifier_updates = _normalize_list_updates(tokens.get("MODIFIERS"))
    region_updates = _normalize_list_updates(tokens.get("REGIONS"))

    updates = {
        "BRANDS": brand_updates,
        "MODIFIERS": modifier_updates,
        "REGIONS": region_updates,
    }

    if not any(updates.values()):
        return {
            "updated": False,
            "brands_added": [],
            "modifiers_added": [],
            "regions_added": [],
        }

    base_brands, base_modifiers, base_regions = load_lexicon(lexicon_path)

    new_brands, new_modifiers, new_regions = merge_lexicon(
        base_brands,
        base_modifiers,
        base_regions,
        updates,
    )

    brand_deltas = []
    for alias, canonical in brand_updates.items():
        alias_norm = alias.lower()
        canonical_norm = canonical.lower()
        if base_brands.get(alias_norm) != canonical_norm:
            brand_deltas.append({"alias": alias_norm, "canonical": canonical_norm})

    existing_modifiers = {m.lower() for m in base_modifiers}
    modifier_deltas = [m.lower() for m in modifier_updates if m.lower() not in existing_modifiers]

    existing_regions = {r.lower() for r in base_regions}
    region_deltas = [r.lower() for r in region_updates if r.lower() not in existing_regions]

    if not dry_run:
        _write_lexicon(lexicon_path, new_brands, new_modifiers, new_regions)

    return {
        "updated": True,
        "brands_added": brand_deltas,
        "modifiers_added": modifier_deltas,
        "regions_added": region_deltas,
    }


def main():
    parser = argparse.ArgumentParser(description="Append reviewed tokens to the shared lexicon")
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=DEFAULT_LEXICON_PATH,
        help="Path to lexicon JSON (defaults to cluster/lexicon.json)",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("reports/novel_tokens.json"),
        help="Reviewed token file produced by cluster.evaluation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    summary = update_lexicon(args.lexicon, args.tokens, dry_run=args.dry_run)
    if not summary["updated"]:
        print("No updates applied; please populate BRANDS/MODIFIERS/REGIONS in the token file.")
        return

    print("Lexicon updated" if not args.dry_run else "Lexicon diff (dry run)")
    if summary["brands_added"]:
        print("  Brands:")
        for entry in summary["brands_added"]:
            print(f"    {entry['alias']} -> {entry['canonical']}")
    if summary["modifiers_added"]:
        print("  Modifiers:")
        for modifier in summary["modifiers_added"]:
            print(f"    {modifier}")
    if summary["regions_added"]:
        print("  Regions:")
        for region in summary["regions_added"]:
            print(f"    {region}")


if __name__ == "__main__":
    main()
