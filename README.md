# Application of Statistical Pattern Recognition in a Generating-Element Space for Human Activity Classification

This repository contains the official source code and experimental setup for the research paper: **"Application of Statistical Pattern Recognition in a Generating-Element Space for Human Activity Classification Using Smartphone Sensors"**.

The study introduces a novel feature engineering technique based on the mathematical framework of decomposition in a space with a generating element (DSGE), also known as Kunchenko space. This method is applied to the classic UCI HAR dataset to demonstrate its effectiveness in enhancing classification performance, particularly for non-Gaussian sensor data.

## Abstract

Human Activity Recognition (HAR) is a cornerstone of modern context-aware computing. While traditional machine learning methods often assume a Gaussian distribution of sensor data, real-world signals are frequently non-Gaussian. This paper adapts the DSGE framework, originally from statistical pattern recognition, to generate powerful polynomial features from raw sensor signals. These features capture higher-order statistical dependencies that are often missed by standard time- and frequency-domain methods. Our experiments show that augmenting a traditional feature set with these DSGE features significantly improves classification accuracy, especially in distinguishing between challenging static activities like sitting and standing.

## Methodology Overview

The core of our approach is to generate a new, compact set of features for each activity class based on signal reconstruction error. The process can be summarized in three steps:

1.  **Train Class-Specific Models:** For each of the six activities (e.g., WALKING, SITTING), a unique DSGE reconstruction model is trained using only the signal data corresponding to that activity. Each model learns the "typical" structure of its class.
2.  **Generate Error Features:** For a new, unseen signal segment, we pass it through all six trained models. Each model attempts to reconstruct the signal, and we calculate the mean squared error of this reconstruction. This results in a 6-dimensional feature vector, where each component represents the "distance" or "dissimilarity" of the signal from the archetypal pattern of each class.
3.  **Classify Using New Features:** This low-dimensional feature vector is then used as input for a standard classifier, such as a Support Vector Machine (SVM), to perform the final activity classification.


## Repository Structure

The project is organized into a series of numbered Python scripts, each corresponding to a specific experiment described in the paper. This structure allows for a clear and sequential reproduction of our findings.

-   `1_UCI_HAR_DSGE_basic.py`: Implements **Experiment 1**. This script runs the most basic version of the DSGE method using a default set of basis functions and evaluates its performance.
-   `2_UCI_HAR_basis_comparison.py`: Implements **Experiment 2**. It systematically compares four different types of basis functions (Polynomial, Trigonometric, Robust, Fractional) to analyze their impact on classification accuracy. The results from this script populate **Table 1** in the paper.
-   `3_UCI_HAR_basis_optimization.py`: Implements **Experiment 3**. This script performs a grid search to find the optimal number of basis functions for the best-performing basis type (Fractional), demonstrating that a compact set of features can achieve high accuracy.
-   `4_UCI_HAR_hybrid_model.py`: Implements **Experiment 4 & 5**. This is the final and most comprehensive experiment.
    -   It develops a **hybrid model** by combining the 6 DSGE features with the most informative traditional features.
    -   It performs an **incremental validation** by adding the 6 DSGE features to the full set of 561 traditional features to prove their unique informational value.
    -   The results from this script populate **Table 2** and are discussed in the final sections of the paper.

## Getting Started

### Prerequisites

-   Python 3.9 or higher
-   A virtual environment is recommended.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/SZabolotnii/SKPR_HAR.git
    cd SKPR_HAR
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### How to Run the Experiments

The scripts are designed to be run sequentially, but they can also be executed independently as they are self-contained. The UCI HAR dataset will be automatically handled by the script logic (it is typically included with `scikit-learn` or downloaded).

To run a specific experiment, simply execute the corresponding Python file. For example, to run the final hybrid model experiment:

```bash
python 4_UCI_HAR_hybrid_model.py
```

The script will print the classification results and metrics directly to the console.

## How to Cite

If you use this code or the methodology in your research, please cite our paper:

```bibtex
@inproceedings{zabolotnii202Xhar,
  title={Application of Statistical Pattern Recognition in a Generating-Element Space for Human Activity Classification Using Smartphone Sensors},
  author={Zabolotnii, Serhii and Khotunov, Vladislav and Chepynoha, Anatolii and Klopotovskyi, Pavlo},
  booktitle={Proceedings of the ... Conference ...},
  year={2025},
  pages={...},
  publisher={...}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.