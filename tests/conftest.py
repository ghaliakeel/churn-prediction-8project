"""Pytest configuration."""
import sys
from pathlib import Path

# add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
