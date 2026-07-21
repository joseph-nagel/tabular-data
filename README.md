# Foundation models for tabular data

This repository allows for an exploration of foundation models for tabular data. Prior-data fitted networks such as the TabPFN architecture are discussed in the [introduction](notebooks/intro.ipynb). Following this, a [classification](notebooks/tabicl_classif.ipynb) and a [regression](notebooks/tabicl_reg.ipynb) toy problem are solved for demonstration purposes. The TabICL model is used in those examples.

A [simplified PFN model](tab_utils/simple_pfn/) is provided for educational purposes.
The implementation combines ideas from TabPFN as well as TabICL and focuses on clarity and comprehensibility.

<p>
  <img src="assets/tabicl_classif.svg" alt="The half-moons test set is shown together with the TabICL predictions" title="Test data and TabICL predictions for binary classification" height="230" style="padding-right: 1em;">
  <img src="assets/tabicl_reg.svg" alt="A 1D regression analysis with heteroscedastic noise is carried out by TabICL" title="Train data and TabICL predictions for 1D regression" height="230">
</p>


## Notebooks

- [Introduction](notebooks/intro.ipynb)
- [TabICL for classification](notebooks/tabicl_classif.ipynb)
- [TabICL for regression](notebooks/tabicl_reg.ipynb)


## Installation

```bash
pip install -e .
```
