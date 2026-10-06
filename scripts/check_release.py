"""Check the committed release manifest without fitting or network access."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    errors = []
    for name, expected in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append(f"Missing or invalid path: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Changed file: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({"release": manifest["release"], "checked_files": len(manifest["files"]), "status": "verified"}))


if __name__ == "__main__":
    main()
