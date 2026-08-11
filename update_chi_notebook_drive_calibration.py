"""Compatibility entry point for rebuilding the configuration-only notebook.

The drive-calibration cells are now part of the Python workflow modules.  This
legacy script name is retained so older instructions do not restore the former
function-heavy notebook layout.
"""

from refactor_chi_error_notebook import main


if __name__ == "__main__":
    main()
