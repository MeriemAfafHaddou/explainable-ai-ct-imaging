# Exploring Explainability and Reliability in Brain CT Classification

This project explores **explainable and reliable AI for brain CT imaging** through intracranial hemorrhage (ICH) classification. The work combines model evaluation, uncertainty estimation, explainability, latent-space perturbation, and an initial exploration of diffusion-based generative models.

## Project Overview

The main workflow uses a **patient-level split** of brain CT data and an ImageNet-pretrained **ResNet-18** classifier to distinguish:

* **ICH**
* **Non-ICH**

The analysis focuses not only on classification performance, but also on understanding model reliability and behavior on difficult cases.

### Main Components

* Patient-level train/test split to avoid data leakage
* ResNet-18 transfer learning
* 3-fold cross-validation
* Held-out test-set evaluation
* Probability calibration using Brier score and ECE
* Uncertainty estimation with **Monte Carlo Dropout**
* **Grad-CAM** for visual explanation
* Latent-space perturbation using a convolutional autoencoder
* Exploratory **diffusion-based generative modeling**

The latent-space experiments provide an initial exploration of counterfactual analysis. The diffusion notebook extends this exploration toward more expressive generative approaches that could support future counterfactual explanation methods.


## Notebooks

### `brain_ct_xai.ipynb`

The main notebook covers:

1. Dataset exploration and patient-level splitting
2. ResNet-18 classifier development
3. Cross-validation and final test evaluation
4. Probability calibration
5. MC Dropout uncertainty estimation
6. Error analysis and Grad-CAM
7. Latent-space counterfactual exploration
8. Limitations and future research directions

### `diffusion_exploration.ipynb`

This notebook explores **diffusion models** and their potential use for generative counterfactual explanations in medical imaging.

It focuses on understanding and implementing the diffusion process and serves as an initial technical exploration rather than a completed diffusion-based counterfactual system.

## Dataset

The project uses the **Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation** dataset installed from [Kaggle](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images/data).

The `data/` directory contains the dataset metadata and CT images. See the accompanying dataset documentation and license files for information about the original dataset and its use.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the main notebook:

```bash
jupyter notebook brain_ct_xai.ipynb
```

Or explore the diffusion experiments:

```bash
jupyter notebook diffusion_exploration.ipynb
```

## Future Work

Future work includes investigating stronger generative approaches, particularly **diffusion models**, for counterfactual explanations. Other directions include multi-class ICH classification, different fine-tuning strategies, larger and more diverse datasets, and evaluation across different CT acquisition conditions.

Further validation could include **external datasets and medical domain-expert feedback** to assess the robustness and clinical relevance of model predictions and explanations.
