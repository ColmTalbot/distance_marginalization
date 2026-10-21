#!/usr/bin/env python
"""Check that every version number in the repository agrees.

The release publishes three artifacts from one tag:

- the Rust crate ``distmarg`` (version in ``rust/distmarg/Cargo.toml``);
- the Python bindings ``distmarg-rs`` (version taken from the same ``Cargo.toml``);
- the Python package ``distmarg`` (``__version__`` in ``python/distmarg/__init__.py``),
  whose ``rust`` extra pins ``distmarg-rs`` to the same version.

``CITATION.cff`` carries the same version.

    python tools/check_versions.py            # versions agree with each other
    python tools/check_versions.py v0.1.0     # ... and with the release tag
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def versions() -> dict[str, str]:
    """Collect the version strings.

    Returns
    -------
    dict
        Maps a description of each location to the version found there.

    Raises
    ------
    RuntimeError
        If a version cannot be found.
    """
    cargo = tomllib.loads((ROOT / "rust/distmarg/Cargo.toml").read_text())["package"]["version"]
    init = (ROOT / "python/distmarg/__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.MULTILINE)
    if match is None:
        raise RuntimeError("no __version__ in python/distmarg/__init__.py")
    extras = tomllib.loads((ROOT / "python/pyproject.toml").read_text())["project"]["optional-dependencies"]
    pins = [r.split("==", 1)[1] for r in extras.get("rust", []) if r.startswith("distmarg-rs==")]
    if len(pins) != 1:
        raise RuntimeError('python/pyproject.toml must pin the rust extra as "distmarg-rs==<version>"')
    cff = re.search(r"^version:\s*['\"]?([^'\"\s]+)", (ROOT / "CITATION.cff").read_text(), re.MULTILINE)
    if cff is None:
        raise RuntimeError("no version in CITATION.cff")
    return {
        "rust/distmarg/Cargo.toml (crate and distmarg-rs)": cargo,
        "python/distmarg/__init__.py (distmarg)": match.group(1),
        "python/pyproject.toml rust extra pin": pins[0],
        "CITATION.cff": cff.group(1),
    }


def main(argv=None) -> int:
    """Compare the versions, and optionally a ``vX.Y.Z`` tag.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; an optional tag.

    Returns
    -------
    int
        0 if everything agrees, 1 otherwise.
    """
    argv = sys.argv[1:] if argv is None else argv
    found = versions()
    if argv and argv[0]:
        tag = argv[0]
        if not tag.startswith("v"):
            print(f"tag {tag!r} must look like vX.Y.Z")
            return 1
        found[f"git tag {tag}"] = tag[1:]
    for where, version in found.items():
        print(f"{version:>12}  {where}")
    if len(set(found.values())) != 1:
        print("version mismatch")
        return 1
    print("all versions agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
