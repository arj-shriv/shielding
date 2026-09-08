"""
Sanity-check new/changed PHITS reference files before they're merged.

Checks, per file under data/raw/phits/{high_energy,multilayer,3layer}/:
  1. Filename follows the convention the app's PHITS matcher expects:
       single-layer : <Material>_<thickness>cm.out   (an optional _1/_2 suffix is OK)
       multilayer   : <Mat1>_<t1>cm_<Mat2>_<t2>cm.out
       3layer       : <Mat1>_<t1>cm_<Mat2>_<t2>cm_<Mat3>_<t3>cm.out
  2. The file parses via shielding_ml.data.loaders.load_phits and yields
     exactly 250 flux bins (the fixed energy grid every model expects).

Usage
-----
    # Check specific files (what the GitHub Action passes):
    python scripts/utilities/check_phits_upload.py data/raw/phits/high_energy/Concrete_45cm.out

    # Check every PHITS file in the repo:
    python scripts/utilities/check_phits_upload.py --all

Exit code 0 = all checked files pass, 1 = at least one failure.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Only needs numpy (see loaders.py) — insert src/ directly so this runs in the
# lightweight CI job without a full `pip install -e .` (which pulls tensorflow).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from shielding_ml.data.loaders import load_phits
from shielding_ml.pipelines.paths import PHITS_HE, PHITS_ML, PHITS_3L

SINGLE_RE      = re.compile(r"^[A-Za-z]+_\d+cm(_\d+)?$")
MULTILAYER_RE  = re.compile(r"^[A-Za-z]+_\d+cm_[A-Za-z]+_\d+cm$")
THREE_LAYER_RE = re.compile(r"^[A-Za-z]+_\d+cm_[A-Za-z]+_\d+cm_[A-Za-z]+_\d+cm$")
EXPECTED_BINS = 250


def check_one(path: Path) -> list[str]:
    """Return a list of problems with this file (empty = OK)."""
    problems: list[str] = []

    if path.suffix != ".out":
        problems.append(f"unexpected extension '{path.suffix}' (expected .out)")

    stem = path.stem
    is_single = PHITS_HE in path.parents
    is_multi  = PHITS_ML in path.parents
    is_3layer = PHITS_3L in path.parents

    if is_single and not SINGLE_RE.match(stem):
        problems.append(
            f"name '{path.name}' doesn't match single-layer convention "
            f"<Material>_<thickness>cm.out (e.g. Concrete_45cm.out)"
        )
    elif is_multi and not MULTILAYER_RE.match(stem):
        problems.append(
            f"name '{path.name}' doesn't match multilayer convention "
            f"<Mat1>_<t1>cm_<Mat2>_<t2>cm.out (e.g. Concrete_45cm_Steel_33cm.out)"
        )
    elif is_3layer and not THREE_LAYER_RE.match(stem):
        problems.append(
            f"name '{path.name}' doesn't match 3-layer convention "
            f"<Mat1>_<t1>cm_<Mat2>_<t2>cm_<Mat3>_<t3>cm.out "
            f"(e.g. Steel_40cm_Concrete_85cm_BPE_20cm.out)"
        )
    elif not is_single and not is_multi and not is_3layer:
        problems.append(
            f"file is not under {PHITS_HE}, {PHITS_ML}, or {PHITS_3L} — "
            "PHITS files must live in one of those three folders to be picked up"
        )

    try:
        flux, _ = load_phits(str(path))
    except Exception as e:
        problems.append(f"failed to parse: {e}")
        return problems

    if flux is None or len(flux) != EXPECTED_BINS:
        n = 0 if flux is None else len(flux)
        problems.append(f"parsed {n} bins, expected {EXPECTED_BINS}")

    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="PHITS .out files to check")
    p.add_argument("--all", action="store_true",
                   help="Check every .out file under data/raw/phits/{high_energy,multilayer,3layer}/")
    args = p.parse_args()

    if args.all:
        paths = sorted(PHITS_HE.glob("*.out")) + sorted(PHITS_ML.glob("*.out"))
        if PHITS_3L.exists():
            paths += sorted(PHITS_3L.glob("*.out"))
    else:
        paths = [Path(f) for f in args.files]

    if not paths:
        print("No PHITS files to check.")
        return 0

    n_fail = 0
    for path in paths:
        if not path.exists():
            print(f"✗ {path}  —  file does not exist")
            n_fail += 1
            continue
        problems = check_one(path)
        if problems:
            print(f"✗ {path}")
            for pr in problems:
                print(f"    - {pr}")
            n_fail += 1
        else:
            print(f"✓ {path}")

    print(f"\n{len(paths) - n_fail}/{len(paths)} passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
