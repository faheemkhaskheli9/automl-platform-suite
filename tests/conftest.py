import sys
from pathlib import Path

# Tests import `automl_core` / `config` directly, same as manage.py does.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
