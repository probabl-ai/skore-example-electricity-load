"""Shared canonical model ordering and colours for the comparison scripts.

Each script that builds plots from ``*_cv_predictions.parquet`` files should use
``canonical_order`` to order models and ``model_colour`` to colour them, so that
models appear in the same order and with the same colour across every plot.

The order groups models by family (HGB, TabICL, Kumo, foundation zero-shot
models), which is more readable than the alphabetical glob order the scripts
would otherwise fall back to. Unknown models (not in the list) go last, in
alphabetical order, so new models still appear deterministically.
"""

# Colourblind-friendly categorical palette with enough entries that every model
# in a full comparison gets a distinct colour (the known models are listed in
# ``MODEL_ORDER`` before ``PALETTE`` wraps around).
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
    "#9932CC",  # dark orchid
    "#00CED1",  # dark turquoise
    "#DC143C",  # crimson
    "#8B6914",  # dark goldenrod
]

MODEL_ORDER = [
    "hgb_mean",
    "hgb_mean_fast",
    "hgb_quantile_q7",
    "hgb_quantile_recal_q7",
    "hgb_quantile_conformal_q7",
    "tabicl_n8_full",
    "tabicl_recal_n8_full",
    "tabicl_conformal_n8",
    "tabicl_forecast_ctx1024_n4",
    "kumo_tabular",
    "kumo_forecast",
    "chronos2_ctx4000",
    "t0_gamma",
]


def canonical_order(names):
    """Return ``names`` in the canonical order (unknown names last, sorted)."""
    known = [n for n in MODEL_ORDER if n in names]
    known_set = set(known)
    extra = sorted(set(names) - known_set)
    return known + extra


def model_colour(name, index=None):
    """A stable colour for ``name``.

    Uses its position in the canonical order when available, otherwise falls
    back to ``index`` (position in the current list). Colours cycle through the
    palette so any number of models can be distinguished.
    """
    try:
        position = MODEL_ORDER.index(name)
    except ValueError:
        position = index if index is not None else 0
    return PALETTE[position % len(PALETTE)]