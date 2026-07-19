#!/usr/bin/env python
"""Thin wrapper → `farm-us run-loyo`. Prefer the CLI directly; this exists for the
script layout in the project spec. Usage: python scripts/run_loyo.py --config <yaml> [key=value ...]"""
import sys

from farm_us.cli import main

if __name__ == "__main__":
    main(["run-loyo", *sys.argv[1:]])
