#!/usr/bin/env python
"""Thin wrapper → `farm-us build-manifest`. Prefer the CLI directly; this exists for the
script layout in the project spec. Usage: python scripts/build_manifest.py --config <yaml> [key=value ...]"""
import sys

from farm_us.cli import main

if __name__ == "__main__":
    main(["build-manifest", *sys.argv[1:]])
