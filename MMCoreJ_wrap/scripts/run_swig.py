import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Create directory and run SWIG")
parser.add_argument("--outdir", type=str, help="Directory to create")
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
sys.exit(result.returncode)
