#!/usr/bin/env python
"""Thin wrapper → `farm-us smoke-test-model`. Prefer the CLI directly; this exists for the
script layout in the project spec. Usage: python scripts/smoke_test_model.py --config <yaml> [key=value ...]"""
import sys

from farm_us.cli import main

if __name__ == "__main__":
    main(["smoke-test-model", *sys.argv[1:]])
