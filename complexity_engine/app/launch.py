"""`complexity-app`: start the page in the browser (needs `pip install -e ".[app]"`)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        raise SystemExit('The app needs Streamlit: pip install -e ".[app]"') from None
    page = Path(__file__).with_name("streamlit_app.py")
    sys.argv = ["streamlit", "run", str(page), *sys.argv[1:]]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
