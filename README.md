# Exploring Reliable, Explainable, and Generative AI for Brain CT Imaging

This project explores **reliable, explainable, and generative AI for brain CT imaging** through intracranial hemorrhage (ICH) classification. The work combines classification, reliability analysis, explainability, and **latent-space exploration using an autoencoder**, with an initial investigation of diffusion-based generative models.

## Project Overview

The main workflow uses a **patient-level split** of brain CT data and an ImageNet-pretrained **ResNet-18** classifier to distinguish:

* **ICH**
* **Non-ICH**

The project goes beyond classification performance to investigate **model reliability, explainability, and how latent representations can be used to explore changes in model predictions**.

### Main Components

* Patient-level train/test splitting to avoid data leakage
* ResNet-18 transfer learning
* Sensitivity improvement experiments
* 4-fold cross-validation
* Held-out test-set evaluation
* Probability calibration using Brier score and ECE
* Uncertainty estimation with **Monte Carlo Dropout**
* **Grad-CAM** for visual explanation
* Latent-space exploration using a convolutional autoencoder
* Exploratory **diffusion-based generative modeling**

The **latent-space experiments form the main exploratory component** of the project. They investigate how modifications to learned representations affect the classifier's predictions and provide an initial foundation for counterfactual analysis.

## Notebooks

### `patients_splitting.ipynb`

Documents the construction of the **patient-level development and held-out test sets**.

It covers:

1. Dataset and patient-level label preparation
2. Patient-level stratification
3. Development/test split construction
4. Split validation and class distribution checks
5. Saving the final patient ID lists for reproducibility

### `brain_ct_xai.ipynb`

The main notebook covers:

1. Dataset exploration
2. ResNet-18 classifier development
3. Cross-validation and final test evaluation
4. Probability calibration
5. MC Dropout uncertainty estimation
6. Error analysis and Grad-CAM
7. Autoencoder reconstruction
8. Latent-space perturbation and counterfactual exploration
9. Limitations and future research directions

### `sensitivity_improvement_experiments.ipynb`

This notebook investigates training strategies intended to improve **ICH sensitivity** under the existing patient-level evaluation setup.

It includes experiments with:

* Stochastic intensity augmentation
* ICH oversampling using `WeightedRandomSampler`
* Class-weighted cross-entropy loss
* Weight-decay selection using cross-validation

The selected configuration is subsequently used for final training and held-out test evaluation.

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

Run the notebooks with:

```bash
jupyter notebook patients_splitting.ipynb
jupyter notebook sensitivity_improvement_experiments.ipynb
jupyter notebook brain_ct_xai.ipynb
jupyter notebook diffusion_exploration.ipynb
```

## Challenges

Several challenges were encountered during the project:

* **Test-set construction:** selecting a representative patient-level test set while preserving diversity in ICH subtypes and slice distributions.
* **Class imbalance:** improving ICH sensitivity without substantially reducing specificity.
* **Model reliability:** investigating high-confidence errors and imperfect calibration.
* **Latent-space exploration:** generating meaningful modifications was limited by the autoencoder's ability to preserve subtle pathological features.


## Presentation

The project presentation provides a concise overview of the methodology, findings, reliability analysis, explainability experiments, and latent-space exploration.

The presentation is available in the [`presentation/`](presentation/) directory.

## Future Work

Future work includes investigating **larger and multi-center datasets** to assess robustness under domain shift, as well as **better fine-tuning strategies** to improve ICH sensitivity. **Quantitative XAI evaluation** could complement visual inspection, while **higher-fidelity generative models**, particularly diffusion-based approaches, could support more faithful latent-space exploration.

Further validation could include **independent datasets and clinician feedback** to assess generalization and the practical relevance of model explanations.
