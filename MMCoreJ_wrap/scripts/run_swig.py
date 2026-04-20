import argparse
import subprocess
import sys
from pathlib import Path

DOC_PAIRS = [
    ("class_c_m_m_core.html", "CMMCore.java"),
]

parser = argparse.ArgumentParser(description="Create directory and run SWIG")
parser.add_argument("--outdir", type=str, help="Directory to create")
parser.add_argument(
    "--doxygen-html-dir",
    type=str,
    help="Path to Doxygen HTML directory for doc conversion",
)
parser.add_argument("swig_cmd", nargs="*", help="SWIG command and arguments")

args = parser.parse_args()

swig_command = args.swig_cmd
if swig_command and swig_command[0] == "--":
    swig_command = swig_command[1:]

if not swig_command:
    print("Error: No SWIG command provided", file=sys.stderr)
    sys.exit(1)

if args.outdir:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

result = subprocess.run(swig_command)
if result.returncode != 0:
    sys.exit(result.returncode)

if args.doxygen_html_dir:
    script_dir = Path(__file__).parent
    converter_script = script_dir / "swig_doc_converter.py"
    html_dir = Path(args.doxygen_html_dir)

    for html_file, java_file in DOC_PAIRS:
        result = subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                str(converter_script),
                str(html_dir / html_file),
                "--java-file",
                str(Path(args.outdir) / java_file),
            ]
        )
        if result.returncode != 0:
            sys.exit(result.returncode)
