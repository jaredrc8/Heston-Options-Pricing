"""
Errors with the codes in SPEC.md section 14.1.

These two exceptions are the only classes in the package, because Python
exceptions have to be classes.
"""

import math

# Exit code of each error (SPEC.md section 11 and 14.1)
EXIT_CODES = {
    "E_CONFIG": 2,
    "E_INPUT": 2,
    "E_SCHEMA": 2,
    "E_SPARSE_GRID": 3,
    "E_CALIB_FAILED": 3,
    "E_CORR_NOT_PSD": 3,
    "E_NONFINITE": 3,
    "E_CHECK_FAILED": 3,
    "E_RUN_EXISTS": 3,
    "E_DIRTY_TREE": 4,
    "E_GOV_UNAUTHORIZED": 4,
    "E_GOV_RESTRICTED_LEAK": 4,
}


class HestonError(Exception):
    """Any expected failure. `code` is one of the keys of EXIT_CODES."""

    def __init__(self, code, message=""):
        super().__init__(f"{code}: {message}")
        self.code = code


class InputError(HestonError):
    """A precondition in SPEC.md section 10.3 was violated."""

    def __init__(self, message=""):
        super().__init__("E_INPUT", message)


def require(condition, message):
    """Raise InputError unless condition is true."""
    if not condition:
        raise InputError(message)


def require_finite(**values):
    """Raise InputError if any named number is NaN or infinite."""
    for name, value in values.items():
        if not math.isfinite(value):
            raise InputError(f"{name} must be finite")
