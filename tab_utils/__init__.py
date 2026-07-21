"""Utilities for tabular data."""

from . import data, simple_pfn, vis
from .data import make_regression_data
from .simple_pfn import (
    SimplePFN,
    FeatureNormalization,
    RepeatedFeatureGrouping,
    TargetAwareCellEmbedding,
    TabularTransformer,
    TabularTransformerBlock,
)
from .vis import plot_data_2d, plot_function_2d
