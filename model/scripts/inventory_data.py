#!/usr/bin/env python
"""Thin wrapper → `farm-us inventory`. Prefer the CLI directly; this exists for the
script layout in the project spec. Usage: python scripts/inventory_data.py --config <yaml> [key=value ...]"""
import sys

from farm_us.cli import main

if __name__ == "__main__":
    main(["inventory", *sys.argv[1:]])
