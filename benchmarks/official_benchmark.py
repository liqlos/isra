#!/usr/bin/env python3
"""Compatibility entry point for the trustworthy paired benchmark runner.

The former implementation claimed to compare Direct and ISRA while only
running ISRA and conflated infrastructure errors with failed answers. Keeping
this path as a thin wrapper prevents old commands from silently producing the
invalid comparison.
"""

from paired_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
