"""Entry point for ``python -m tools.hetzner_dns``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
