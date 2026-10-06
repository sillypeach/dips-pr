#!/usr/bin/env python3
"""Entry point for the 2026-10-06 reproducibility release.

Use `python reproduce.py <command> --help` for command-specific arguments.
No command without an explicit experiment run request starts production fits.
"""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
os.environ.update({name: "1" for name in THREADS})


def call(relative, *args):
    subprocess.run([sys.executable, str(ROOT / relative), *map(str, args)], check=True)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("Commands: verify, tables, annotations, test, synthetic, hiv, pruning, ablation")
        print("  verify    Check release file hashes and all 561 manuscript summary cells")
        print("  tables    Rebuild CSV/LaTeX summaries from archived per-seed results")
        print("  test      Run small numerical/adapter tests; requires requirements-dev.txt")
        print("  others    Delegate to the corresponding experiment CLI")
        return 0
    command, *rest = args
    if command in ("synthetic", "hiv", "pruning", "ablation"):
        call(f"experiments/{command}/run.py", *rest)
    elif command == "tables":
        call("scripts/rebuild_tables.py", *rest)
    elif command == "annotations":
        call("scripts/rebuild_annotations.py", *rest)
    elif command == "verify":
        call("scripts/check_release.py", *rest)
        call("scripts/rebuild_tables.py", "--check-only")
        call("experiments/synthetic/run.py", "verify-inputs")
    elif command == "test":
        subprocess.run([sys.executable, "-m", "pytest", "--import-mode=importlib",
                        str(ROOT / "tests"), str(ROOT / "experiments"), *rest], check=True)
    else:
        raise ValueError(f"Unknown command: {command}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
