"""Fast, read-only smoke test for a SiliconGraph .armpack archive.

Example:
  python scripts/test_armpack.py output/model.armpack --serial 88e07cd0
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


REQUIRED_FILES = {
    "manifest.json",
    "device_profile.json",
    "runtime_config.json",
    "selector_index.json",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an .armpack archive")
    parser.add_argument("package", type=Path, help="Path to the .armpack file")
    parser.add_argument("--serial", help="Expected connected-device serial")
    args = parser.parse_args()

    if not args.package.is_file():
        fail(f"package not found: {args.package}")

    try:
        with zipfile.ZipFile(args.package) as archive:
            names = set(archive.namelist())
            missing = REQUIRED_FILES - names
            if missing:
                fail(f"missing required files: {', '.join(sorted(missing))}")

            manifest = json.loads(archive.read("manifest.json"))
            profile = json.loads(archive.read("device_profile.json"))
            json.loads(archive.read("runtime_config.json"))

            graphs = manifest.get("graphs", [])
            weights = manifest.get("weights", [])
            if not graphs:
                fail("manifest has no compiled graphs")
            if not weights:
                fail("manifest has no model weights")

            for graph in graphs:
                path = graph.get("path")
                if not path or path not in names:
                    fail(f"graph entry is missing: {path or '<no path>'}")

            for weight in weights:
                path = weight.get("path")
                if not path or path not in names:
                    fail(f"weight entry is missing: {path or '<no path>'}")
                actual_size = archive.getinfo(path).file_size
                expected_size = weight.get("size_bytes")
                if expected_size is not None and actual_size != expected_size:
                    fail(f"weight size mismatch for {path}: {actual_size} != {expected_size}")

            profile_id = str(profile.get("profile_id", ""))
            serial = (
                profile.get("serial")
                or profile.get("device", {}).get("serial")
                or profile_id.removeprefix("detected_")
            )
            if args.serial and serial != args.serial:
                fail(f"device serial mismatch: expected {args.serial}, got {serial}")

    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        fail(f"invalid package: {error}")

    print("PASS: .armpack structure is valid")
    print(f"  Model:   {manifest.get('model_id', 'unknown')}")
    print(f"  Graphs:  {len(graphs)}")
    print(f"  Weights: {sum(weight.get('size_bytes', 0) for weight in weights):,} bytes")
    print(f"  Device:  {profile.get('name', profile.get('device_name', 'unknown'))}")
    print(f"  Serial:  {serial or 'not recorded'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
