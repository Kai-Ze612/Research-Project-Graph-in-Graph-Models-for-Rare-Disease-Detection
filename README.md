# Graph-in-Graph (GiG) Framework for Rare Disease Detection

This repository contains the implementation, documentation, and results of the research project **"Graph-in-Graph Models for Rare Disease Detection"**. This project applies the **Graph-in-Graph (GiG)** framework to the **Shepherd Dataset** to detect rare genetic disorders.

The work was conducted at the **Technical University of Munich (TUM)** under the supervision of the **Chair for Computer Aided Medical Procedures & Augmented Reality (CAMP)**.

## Project Overview

**Rare disease detection** faces significant challenges due to the scarcity of patient data and the complexity of phenotypic patterns. Standard approaches often treat patients as isolated data points, ignoring the latent structural relationships between patients with similar genetic profiles.

We propose an implementation of the **Graph-in-Graph (GiG)** architecture, a novel deep learning framework that:
1.  **Preserves Structural Information:** Models individual patients as graphs (phenotypes as nodes) rather than flattening them into vectors.
2.  **Learns Population Dynamics:** Simultaneously learns a latent **population-level graph** that connects similar patients, uncovering hidden relationships.
3.  **Regularizes with NDDL:** Incorporates a **Node Degree Distribution Loss (NDDL)** to enforce realistic sparsity and connectivity in the learned patient network.

## Team

| Role | Name | Affiliation |
| :--- | :--- | :--- |
| **Teammate** | **Kai-Ze Deng** | Technical University of Munich (TUM) |
| **Teammate** | **Büşra Nur Zeybek** | Technical University of Munich (TUM) |
| **Teammate** | **Tuna Karacan** | Technical University of Munich (TUM) |
| **Supervisor** | **Kamilia Zaripova** | Chair for Computer Aided Medical Procedures (CAMP), TUM |

## Dataset

The project utilizes the **Shepherd Dataset** from the Harvard Dataverse.
> **Source:** [Harvard Dataverse - Shepherd Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TZTPFL)

### Usage
Process from Scratch
1.  Download the raw dataset from the Harvard Dataverse link above.
2.  Run the preprocessing script to generate the graph structures:
    ```bash
    jupyter notebook processing.ipynb
    ```

## Repository Structure
```text
Research-Project-Graph-in-Graph-Models-for-Rare-Disease-Detection/
├── main                # Main implementation (Model training & Evaluation)
├── processing.ipynb    # Data preprocessing script for Shepherd Dataset
├── kg.py               # Knowledge Graph (KG) utilities
├── Data/               # Input Data directory (Place OneDrive data here)
└── Presentation/       # Documentation of project processes and milestones
