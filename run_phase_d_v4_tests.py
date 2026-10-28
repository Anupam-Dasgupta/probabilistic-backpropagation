"""Run the Phase D v4 suite with CUDA determinism configured before imports."""

from __future__ import annotations

import os


def main() -> int:
    """Discover and run all local tests after setting the cuBLAS workspace.

    Inputs are standard unittest command-line defaults; no explicit arguments are
    consumed. The process exit code is zero only when every discovered test
    passes. This sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` before importing tests
    or PyTorch, preserves strict determinism, and performs no downloads or runs.
    """

    configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if configured not in {":4096:8", ":16:8"}:
        raise RuntimeError("Unsupported CUBLAS_WORKSPACE_CONFIG for strict tests")
    import unittest

    suite = unittest.defaultTestLoader.discover(
        "tests", pattern="test_phase_d_v4*.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
