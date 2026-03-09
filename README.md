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
├── test_isolation.py        # Only test for Isolation Forest
├── train_clusters.py        # Training clustering models
├── train_isolation_all.py   # Training: Saves all files
├── train_isolation_audioset.py # Training: Specifically on AudioSet/Friends/Kuznetsova
├── train_isolation.py       # Training: Saves only test files
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
  python train_isolation_all.py
  ```

* **Save only test files:** 
```bash
  python train_isolation.py
  ```

* **Dataset specific (AudioSet/Friends/Kuznetsova):** 
```bash
python train_isolation_audioset.py
```
