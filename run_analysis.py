#!/usr/bin/env python3
"""Run the physiological/cognitive complexity analysis MVP.

The command-line options live in complexity_engine/cli.py, so this script and the installed
`complexity-analysis` command behave identically.
"""

from complexity_engine.cli import timeseries_main as main

if __name__ == "__main__":
    main()
