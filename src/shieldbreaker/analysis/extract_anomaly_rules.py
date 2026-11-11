"""Paper-aligned replacement for legacy per-API/GID heuristic rule extraction."""

from shieldbreaker.cli import main

if __name__ == "__main__":
    import sys

    main(["analyze", *sys.argv[1:]])
