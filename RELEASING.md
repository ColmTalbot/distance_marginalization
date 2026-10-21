# Releasing

One tag, `vX.Y.Z`, releases three packages through `.github/workflows/release.yml`:

| package | registry | built from |
|---|---|---|
| `distmarg` (Python) | PyPI | `python/` |
| `distmarg-rs` (Python bindings) | PyPI | `rust/distmarg` with maturin |
| `distmarg` (Rust crate) | crates.io | `rust/distmarg` |

`distmarg-rs` ships the following wheels:
- **stable ABI:** one `cp310-abi3` wheel per platform, which installs on CPython 3.10 and every later version.
- **platforms:** Linux glibc (x86_64, aarch64, i686, armv7, s390x, ppc64le), Linux musl (x86_64, aarch64, armv7), macOS (x86_64, arm64) and Windows (x64, x86, arm64).
- **free-threaded:** `cp313t` and `cp314t` wheels on Linux x86_64 and aarch64 (glibc and musl), macOS and Windows x64. The stable ABI does not cover free-threaded builds.

Before anything is published, the workflow installs the built wheels on Linux (x86_64 and aarch64), macOS (Intel and Apple silicon) and Windows. It runs the test suite against them on Python 3.10, 3.14 and 3.14t.

## One-time setup

1. **PyPI trusted publishing** (no API tokens). On PyPI, under *Your account → Publishing → Add a new pending publisher*, add one pending publisher for each of `distmarg` and `distmarg-rs`, both with:
   - owner `ColmTalbot`
   - repository `distance_marginalization`
   - workflow `release.yml`
   - environment `pypi`
2. **crates.io token.** crates.io only allows trusted publishing for crates that already exist, so the first release uses a token.
   - Create an API token with the `publish-new` and `publish-update` scopes.
   - Add it as the secret `CARGO_REGISTRY_TOKEN` of a GitHub environment named `crates-io`.
   - After the first release you can switch to crates.io trusted publishing (`rust-lang/crates-io-auth-action`) and delete the token.
3. **GitHub environments.** Create `pypi` and `crates-io` under *Settings → Environments*. Adding yourself as a required reviewer on both makes every publish wait for your approval.

## Making a release

1. Set the new version in all four places:
   - `rust/distmarg/Cargo.toml`
   - `__version__` in `python/distmarg/__init__.py`
   - the `distmarg-rs==` pin in the `rust` extra of `python/pyproject.toml`
   - `CITATION.cff`

   Then check that they agree:
   ```bash
   python tools/check_versions.py
   ```
2. Commit, then do a dry run: *Actions → Release → Run workflow* on `main`. This builds and tests every wheel and publishes nothing.
3. Tag and push:
   ```bash
   git tag v0.1.0
   ```
   ```bash
   git push origin v0.1.0
   ```
   The workflow checks the tag against the versions, rebuilds and retests everything, then publishes:
   - the crate to crates.io;
   - `distmarg` and `distmarg-rs` to PyPI, with attestations;
   - a GitHub release with every artifact attached.

   The crate and PyPI uploads are independent jobs. If one fails, rerun only that job.
