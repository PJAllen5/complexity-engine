#!/usr/bin/env python3
"""Run network-science analysis on graph-shaped (edge list) data.

The command-line options live in complexity_engine/cli.py, so this script and the installed
`complexity-network` command behave identically.
"""

from complexity_engine.cli import network_main as main

if __name__ == "__main__":
    main()
