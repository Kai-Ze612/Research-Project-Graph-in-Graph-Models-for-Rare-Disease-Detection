# True Gene Classification Based on the GiG Framework
This repository contains the implementation and results of our research project on Graph-in-Graph Models for Rare Disease Detection, conducted in collaboration with two teammates under the supervision of faculty CAMP at the Technical University of Munich.

## Team

**Teammates:**
* Kai-Ze Deng
* Büşra Nur Zeybek
* Tuna Karacan

**Supervisor:**
* Kamilia Zaripova

## Dataset
The dataset comes from [Harvard Dataverse Shepherd Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TZTPFL).

The preprocessed dataset is too large to be stored in this GitHub repository. It is available via [my personal OneDrive](https://1drv.ms/f/s!ApjXPmewijhustFXDzvZkjdoujq47A?e=9AUmkl).

## Repository Structure
```
Research-Project-Graph-in-Graph-Models-for-Rare-Disease-Detection/
├── main.ipynb      # Main implementation notebook
├── processing.ipynb # data preprocessing for Shepherd Dataset
├── kg.py           # Knowledge graph utilities
├── Data/           # Data directory (placeholder - see OneDrive link for actual data)
└── Presentation    # Document the project processes
```

## Project Overview
Our project proposes a novel Graph-in-Graph (GiG) framework for rare disease detection. It addresses the challenge of limited data and complex symptom patterns by modeling individual patients and their phenotypic relationships using graph structures. We leverage both node-level feature learning and cross-patient connections to enhance diagnostic accuracy and uncover hidden disease patterns.

## Results:
The research results are documented in detail in our paper: Research-Project-Graph-in-Graph-Models-for-Rare-Disease-Detection.pdf

## Installation and Usage

### Install required dependencies
```bash
pip install -r requirements.txt
```
