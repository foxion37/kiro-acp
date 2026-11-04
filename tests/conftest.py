"""Test bootstrap: put the hermes-agent repo (and its venv's site-packages) on sys.path.

The plugin imports ``providers`` / ``agent`` from a hermes-agent checkout. Point
HERMES_REPO at one (defaults to the live checkout). Run with any Python that has
pytest; hermes third-party deps come from the checkout's venv via sys.path tail.
"""
import os
import sys

HERMES_REPO = os.environ.get("HERMES_REPO", os.path.expanduser("~/.hermes/hermes-agent"))
if HERMES_REPO not in sys.path:
    sys.path.insert(0, HERMES_REPO)

_site = os.path.join(HERMES_REPO, "venv", "lib", "python3.11", "site-packages")
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.append(_site)
