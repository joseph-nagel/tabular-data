"""Simple tabular PFN classifier witth sklearn-like API."""

from typing import Self

import numpy as np
import torch
import torch.nn as nn

from .model import SimplePFN


class SimplePFNClassifier:
    """
    Simple PFN classifier with a fit/predict-interface scikit-learn.

    Parameters
    ----------
    model : SimplePFN
        PFN model to be wrapped.

    """

    def __init__(self, model: SimplePFN):
        self.model = model

    # TODO: impute NaNs, encode categoricals, remove constants
    @staticmethod
    def _preprocess_features(x: np.ndarray) -> np.ndarray:
        """Prepare features."""
        x = np.asarray(x)
        # ensure features of shape (num_samples, num_features)
        if x.ndim != 2:
            raise ValueError(f"Invalid features shape: {x.shape}")
        return x

    # TODO: convert dtype
    @staticmethod
    def _preprocess_targets(y: np.ndarray) -> np.ndarray:
        """Prepare targets."""
        y = np.asarray(y)
        # ensure targets of shape (num_samples,) or (num_samples, 1)
        if y.ndim != 1 and not (y.ndim == 2 and y.shape[2] == 1):
            raise ValueError(f"Invalid targets shape: {y.shape}")
        return y

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, num_labels: int | None = None) -> Self:
        """Prepare and store train data."""
        # set training data
        self.x_train = self._preprocess_features(x_train)  # (num_train, num_features)
        self.y_train = self._preprocess_targets(y_train)  # (num_train,) or (num_train, 1)

        if len(x_train) != len(y_train):
            raise ValueError("Training data size does not match")

        # set number of classes
        if num_labels is None:
            unique_labels = np.unique(y_train)
            num_labels = len(unique_labels)
            max_label = unique_labels.max()

            if num_labels != max_label + 1:
                raise ValueError("Not all labels in training set")

        self.num_labels = num_labels

        return self

    # TODO: generalize device handling
    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        # concatenate train and test features
        x_test = self._preprocess_features(x_test)  # (num_test, num_features)
        x = np.concatenate((self.x_train, x_test), axis=0)  # (num_train + num_test, num_features)

        # prepare tensors
        x = torch.from_numpy(x).unsqueeze(0)  # (1, num_train + num_test, num_features)
        y_train = torch.from_numpy(self.y_train).unsqueeze(0)  # (1, num_train)

        if hasattr(self.model, "device"):
            x = x.to(self.model.device)
            y_train = y_train.to(self.model.device)

        # predict
        with torch.no_grad():
            y_logits = self.model(x, y_train)  # (1, num_test, num_classes)
        y_probas = nn.functional.softmax(y_logits, dim=-1)  # (1, num_test, num_classes)

        return y_probas.squeeze(0).cpu().numpy()  # (num_test, num_classes)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        y_probas = self.predict_proba(x_test)  # (num_test, num_classes)
        y_topclass = y_probas.argmax(axis=1)  # (num_test,)
        return y_topclass
