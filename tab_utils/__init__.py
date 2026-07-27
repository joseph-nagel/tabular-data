"""Utilities for tabular data."""

from . import data, prior, simple_pfn, vis
from .data import make_regression_data
from .prior import PriorDataModule
from .simple_pfn import (
    SimplePFN,
    SimplePFNClassifier,
    FeatureNormalization,
    RepeatedFeatureGrouping,
    CellEmbedding,
    TabularTransformer,
    TabularTransformerBlock,
)
from .vis import plot_data_2d, plot_function_2d
