"""Enter the verified Gadi OS image before importing scorer dependencies."""
import argparse
import os
from pathlib import Path
from runtime_entry import enter

p = argparse.ArgumentParser()
p.add_argument('--work', type=Path, required=True)
p.add_argument('--python', required=True)
p.add_argument('--script', required=True)
a, rest = p.parse_known_args()
enter(a.work)
os.execv(a.python, [a.python, a.script, *rest])
