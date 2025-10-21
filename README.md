# Improving Soundscape Retrieval for Bioacoustic Monitoring

This repository contains the official implementation of the paper **"Improving Soundscape Retrieval for Bioacoustic Monitoring: An  Analysis of Fusion Techniques with Pre-trained Embeddings"**, accepted at **WebMedia 2025**.  
The project focuses on developing efficient and explainable methods for retrieving similar ecoacoustic recordings using deep-learning embeddings and vector databases.

---

##  Overview

The retrieval of similar soundscapes is essential for **bioacoustic** and **ecoacoustic monitoring**, yet it remains challenging due to the large volume of unlabeled data, environmental noise, and the complexity of acoustic scenes.  To overcome the limitations of traditional feature-based methods, this study  proposes an **efficient system** that integrates **embeddings extracted from a pretrained deep learning model**, combined with a **noise reduction technique** and **feature fusion strategies** within a **vector database** to enable similarity-based retrieval.  We evaluated the system using **bird**, **amphibian**, and **mammal** recordings across four experimental methodologies, including a use case focused on **endangered species**. Results show that **embedding vectors consistently outperform traditional MFCC (Mel-frequency cepstral coefficients) features** in capturing acoustic similarity, and that **approximate search algorithms (HNSW)** significantly improve both retrieval precision and query efficiency.   Additionally, the system effectively retrieves recordings of the **critically endangered species *Crax alberti*** and maps their **geographic distribution**, highlighting its potential for **conservation planning** and **early-warning monitoring**.

---

##  Project Structure

```bash
acoustic-soundscape/
│
├── configs/                     # YAML experiment configuration files
│   ├── experiment_1.yaml
│   ├── experiment_2.yaml
│   └── experiment_3.yaml
│
├── scripts/                     # Main experiment scripts
│   ├── check_dependencies.py
│   ├── run_experiments_embeddings.py
│   ├── run_experiments_mfcc.py
│   ├── filter_species.py
│   ├── experiment_1.py
│   ├── experiment_2.py
│   ├── experiment_3.py
│   └── geolocation_map.py
│
├── src/                         # Core signal-processing modules
│   ├── extract_embeddings.py
│   ├── extract_mfccs.py
│   ├── generate_fusions.py
│   ├── resampling_noisereduce.py
│   └── rms_energy.py
│
│
├── main.py                      # Orchestrator script 
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment definition
├── Makefile                     # Linux/macOS automation
├── Makefile.win                 # Windows automation
└── README.md                    # Project documentation
```
## Dataset

The experiments were conducted using public bioacoustic datasets, including BirdCLEF 2025, available at: https://www.kaggle.com/competitions/birdclef-2025/data

This dataset contains thousands of birds, amphibians, mammals and insects vocalizations from different species and recording conditions.

---


## Installation

Clone the repository and create a virtual environment: git clone https://github.com/AndresDavidPeralta/acustic-soundscape.git

cd acustic-soundscape

### Option 1: using Conda

conda env create -f environment.yml

conda activate acoustic-soundscape

### Option 2: using pip

python -m venv .venv
source .venv/bin/activate  # Linux/macOS

.venv\Scripts\activate     # Windows

pip install -r requirements.txt

---

## Running the Experiments

#### Each experiment can be executed via the orchestrator script: 

python main.py --config configs/experiment_1.yaml


#### Alternatively, use the provided Makefile targets:


make experiment1      # Run Experiment 1 

make experiment2      # Run Experiment 2 

make experiment3      # Run Experiment 3


#### For Windows users:

make -f Makefile.win experiment1

--- 




