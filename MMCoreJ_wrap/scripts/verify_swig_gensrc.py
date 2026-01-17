import argparse
import json
import subprocess
import sys
from pathlib import Path

TARGET_NAME = "swig-mmcorej"
JAVA_SUBDIR = "generated-java/mmcorej"

parser = argparse.ArgumentParser(
    description="Verify SWIG Java outputs match meson.build declarations"
)
parser.add_argument("builddir", type=Path, help="Meson build directory")

args = parser.parse_args()

builddir = args.builddir.resolve()
if not builddir.is_dir():
    print(f"Error: Build directory does not exist: {builddir}", file=sys.stderr)
    sys.exit(1)

try:
    result = subprocess.run(
        ["meson", "introspect", "--targets", str(builddir)],
        capture_output=True,
        text=True,
    )
except FileNotFoundError:
    print("Error: 'meson' command not found", file=sys.stderr)
    sys.exit(1)
if result.returncode != 0:
    print(f"Error running meson introspect: {result.stderr}", file=sys.stderr)
    sys.exit(1)

try:
    targets = json.loads(result.stdout)
except json.JSONDecodeError as e:
    print(f"Error parsing meson introspect output: {e}", file=sys.stderr)
    sys.exit(1)
swig_target = next((t for t in targets if t["name"] == TARGET_NAME), None)
if swig_target is None:
    print(f"Error: Target '{TARGET_NAME}' not found in build", file=sys.stderr)
    sys.exit(1)

declared = set()
for filepath in swig_target["filename"]:
    name = Path(filepath).name
    if name.endswith(".java"):
        declared.add(name)

java_dir = builddir / JAVA_SUBDIR
if not java_dir.is_dir():
    print(f"Error: SWIG output directory does not exist: {java_dir}", file=sys.stderr)
    print("Has the build been run?", file=sys.stderr)
    sys.exit(1)

actual = {f.name for f in java_dir.iterdir() if f.is_file() and f.suffix == ".java"}

missing = declared - actual
extra = actual - declared

if missing or extra:
    if missing:
        print("Declared in meson.build but not produced by SWIG:")
        for name in sorted(missing):
            print(f"  {name}")
    if extra:
        print("Produced by SWIG but not declared in meson.build:")
        for name in sorted(extra):
            print(f"  {name}")
    sys.exit(1)

print(f"OK: {len(actual)} Java files match")
