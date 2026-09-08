#!/usr/bin/env python3
"""validate_labels.py — check the label files against labels/schema.json.

Confirms that every labels_<month>_<class>.json conforms to the published
schema, and that the summary counts inside each file match its own contents.

Usage
-----
    pip install jsonschema
    ./validate_labels.py                    # validates labels/
    ./validate_labels.py --labels some/dir

Exits non-zero if anything fails, so it can be used as a release gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("labels"),
                    help="directory holding schema.json and labels_*.json")
    args = ap.parse_args()

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("error: jsonschema is not installed.  pip install jsonschema")
        return 2

    schema_path = args.labels / "schema.json"
    if not schema_path.exists():
        print(f"error: no schema.json in {args.labels}")
        return 2

    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    files = sorted(glob.glob(str(args.labels / "labels_*.json")))
    if not files:
        print(f"error: no labels_*.json in {args.labels}")
        return 2

    failures = 0
    total = 0
    for path in files:
        doc = json.loads(Path(path).read_text())
        problems = []

        for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
            where = "/".join(str(p) for p in err.path) or "(root)"
            problems.append(f"{where}: {err.message}")

        # Cross-checks the schema cannot express: the summary must match the body.
        anns = doc.get("annotations", [])
        pos = sum(1 for a in anns if a.get("label") == 1)
        neg = sum(1 for a in anns if a.get("label") == 0)
        recs = len({a.get("recording_32khz") for a in anns})
        for field, actual in (("count", len(anns)), ("n_positive", pos),
                              ("n_negative", neg), ("n_recordings", recs)):
            if doc.get(field) != actual:
                problems.append(f"{field}: says {doc.get(field)}, contents give {actual}")
        for a in anns:
            if a.get("species") != doc.get("species"):
                problems.append(f"annotation species {a.get('species')!r} "
                                f"!= file species {doc.get('species')!r}")
                break
            if a.get("month") != doc.get("month"):
                problems.append(f"annotation month {a.get('month')!r} "
                                f"!= file month {doc.get('month')!r}")
                break
            if a.get("frame_index") != int(a.get("annotation_offset_s", 0) / 5):
                problems.append(f"frame_index {a.get('frame_index')} does not match "
                                f"offset {a.get('annotation_offset_s')}")
                break

        total += len(anns)
        name = Path(path).name
        if problems:
            failures += 1
            print(f"FAIL  {name}")
            for p in problems[:5]:
                print(f"        {p}")
            if len(problems) > 5:
                print(f"        ... and {len(problems) - 5} more")
        else:
            print(f"ok    {name}  ({len(anns)})")

    print(f"\n{len(files)} files, {total} annotations, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
