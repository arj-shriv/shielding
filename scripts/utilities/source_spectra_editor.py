"""
Pure-text editor for src/shielding_ml/data/source_spectra.py — used by the
Model Explorer's "Add source spectrum" panel to generate the PR diff.

Kept dependency-free (no Streamlit, no GitHub API, no shielding_ml import) so
it's trivially unit-testable: feed it a string, get a string back.
"""
from __future__ import annotations

import keyword
import re

_READY_MARKER = "print('[source_spectra] All sources ready.')"
_DICT_OPEN    = "SOURCE_FUNCTIONS: dict[str, object] = {"

_VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SourceEditError(ValueError):
    pass


def validate_var_name(name: str, existing_names: set[str]) -> None:
    if not _VALID_NAME_RE.match(name):
        raise SourceEditError(
            f"'{name}' isn't a valid source name — lowercase letters, digits, "
            "underscores only, must start with a letter (e.g. 'concrete100')."
        )
    if keyword.iskeyword(name):
        raise SourceEditError(f"'{name}' is a Python reserved word — pick another name.")
    if name in existing_names:
        raise SourceEditError(f"'{name}' is already a registered source spectrum.")


def insert_new_source(original_text: str, var_name: str, phits_filename: str,
                       e_cut: float, fit_label: str) -> str:
    """
    Return new source_spectra.py content with one new source spectrum added:
      - a `{var_name} = _make_noisy_source('{phits_filename}', E_cut={e_cut})`
        line (with its "Fitting ..." print), inserted just before the
        "All sources ready." print
      - a `'{var_name}': {var_name},` entry in the SOURCE_FUNCTIONS dict

    Raises SourceEditError if the anchor lines this depends on aren't found
    (the file was restructured since this was written) rather than silently
    producing a broken file.
    """
    if _READY_MARKER not in original_text:
        raise SourceEditError(
            "Could not find the source-fitting block in source_spectra.py "
            "(expected the \"All sources ready.\" print) — file structure may "
            "have changed; edit it by hand instead."
        )
    if _DICT_OPEN not in original_text:
        raise SourceEditError(
            "Could not find 'SOURCE_FUNCTIONS: dict[str, object] = {' in "
            "source_spectra.py — file structure may have changed; edit it by "
            "hand instead."
        )

    new_def = (
        f"print('[source_spectra] Fitting {fit_label} ...')\n"
        f"{var_name} = _make_noisy_source('{phits_filename}', E_cut={e_cut})\n"
    )
    text = original_text.replace(_READY_MARKER, new_def + _READY_MARKER, 1)

    dict_start = text.index(_DICT_OPEN) + len(_DICT_OPEN)
    dict_close = text.index("}", dict_start)

    # Insert right after the last existing entry (which already ends in a
    # trailing comma), before the closing brace.
    body = text[dict_start:dict_close].rstrip()
    if not body.endswith(","):
        body += ","
    new_body = body + f"\n    '{var_name}': {var_name},\n"

    return text[:dict_start] + new_body + text[dict_close:]
