"""Episode controller entrypoint:
thin shim over `python -m harness.mcp` so the controller reads as one
command. Everything lives in harness/mcp.py.

    python scripts/prepare_episode.py --task daily_reddit_digest --tm A ...
        == python -m harness.mcp prepare --task daily_reddit_digest --tm A ...
"""

import sys
from pathlib import Path

# run as a plain script: put the repo root (not scripts/) on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.mcp import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["prepare", *sys.argv[1:]]))
