#!/usr/bin/env python
"""Thin wrapper → `farm-us leakage-audit`. Prefer the CLI directly; this exists for the
script layout in the project spec. Usage: python scripts/verify_leakage.py --config <yaml> [key=value ...]"""
import sys

from farm_us.cli import main

if __name__ == "__main__":
    main(["leakage-audit", *sys.argv[1:]])
