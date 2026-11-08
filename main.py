"""Compatibility entry point. Prefer python -m grnn_bio <command>."""
from grnn_bio.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
