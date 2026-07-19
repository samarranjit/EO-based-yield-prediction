#!/usr/bin/env python
"""Thin wrapper → `farm-us inspect-batch`. Prefer the CLI directly; this exists for the
script layout in the project spec. Usage: python scripts/inspect_batch.py --config <yaml> [key=value ...]"""
import sys

from farm_us.cli import main

if __name__ == "__main__":
    main(["inspect-batch", *sys.argv[1:]])
