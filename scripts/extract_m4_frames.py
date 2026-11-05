"""
scripts/extract_m4_frames.py
============================
Standalone script to extract M4 keyframes using the default or specified config.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stumpvision.cli import run_m4_extract


if __name__ == "__main__":
    sys.exit(run_m4_extract())

