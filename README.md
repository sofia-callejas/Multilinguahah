# Multilinguahah 

A New Unsupervised Multilingual Acoustic Laughter Segmentation Method

## Repository Structure

```text
.
├── evaluation/              # Scripts for F1-score and performance metrics
├── ext/                     # External dependencies
├── laughter_detection/core/ # Core processing logic
│   ├── audio_embedding.py   # Audio feature extraction (e.g., BYOL-A)
│   ├── embedding.py         # Embedding utilities
│   ├── voice_remover.py     # Source separation and voice removal logic
│   └── utils.py             # General helper functions
├── elbow_method.py          # Clustering optimization (K-means/Elbow)
├── get_embeddings.py        # Script to generate embeddings from raw audio
├── remove_voice_music.py    # Preprocessing: removing background music
├── remove_voice.py          # Preprocessing: isolating non-vocal components
├── main.py                  # Unified training and testing script
├── .gitignore               # Files to exclude from Git
├── requirements.txt         # Project dependencies
├── setup.sh                 # Environment setup script
└── README.md                # Project documentation
```

## Getting Started

### 1. Prerequisites

Before running the scripts, ensure you have the following installed.

#### **Python & Environment Management**
We recommend using **Miniconda** or **Anaconda** to manage your environments.
* **Python:** 3.10+
* **Conda:** [Download Miniconda here](https://docs.anaconda.com/miniconda/)

#### **System Dependencies (FFmpeg)**
`FFmpeg` is required for `voice_remover.py` and all audio manipulation tasks. You can install it via your system package manager:

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**On macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**On Windows (using Winget):**
```bash
winget install ffmpeg
```

### Installation & Setup

The setup.sh script automates the installation of TimeSformer and BYOL-A, and downloads the required pretrained weights.

```bash
# 1. Create and activate your conda env (recommended)
conda create --name laughter_env python=3.10 -y
conda activate laughter_env

# 2. Install core requirements
pip install -r requirements.txt

# Run the setup script
./setup.sh
```

### Workflow

### Step 1: Preprocessing

Clean audio files by removing vocal or music components:

```bash
python remove_voice.py --root_dir path/to/audio
```

### Step 2: Feature Extraction

Generate embeddings from the processed audio:

```bash
python get_embeddings.py --root_dir data/processed/
```

### Step 3: Training

Select the training script based on your storage and dataset needs:

* **Save all files:** 
```bash
  python main.py --root_dir path/to/audio --labels_dir path/to/labels --task train_isolation_all
  ```

* **Save only test files:** 
```bash
  python main.py --root_dir path/to/audio --labels_dir path/to/labels --task train_isolation
  ```

* **Dataset specific (AudioSet/Friends/Kuznetsova):** 
```bash
python main.py --root_dir path/to/audio --labels_dir path/to/labels --task train_isolation_audioset
```

### Dataset

Repository with dataset Standup4AI https://github.com/sofia-callejas/seq-Standup4AI

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC 4.0).

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

You are free to:
* **Share** — copy and redistribute the material in any medium or format
* **Adapt** — remix, transform, and build upon the material

Under the following terms:
* **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* **NonCommercial** — You may not use the material for commercial purposes.

For more details, please see the [full license text](https://creativecommons.org/licenses/by-nc/4.0/).
