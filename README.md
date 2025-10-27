## [**UPDATE**]: FunnyNet-W: Multimodal Learning of Funny Moments in Videos in the Wild
By Zhi-Song Liu, Robin Courant and Vicky Kalogeiton

### [Project Page](https://www.lix.polytechnique.fr/vista/projects/2024_ijcv_liu/) | [Paper](https://arxiv.org/pdf/2401.04210.pdf) | [Data](https://drive.google.com/drive/folders/1ZM6agmEnheiyP0IIrD3Fc7DOubjyu5eO?usp=share_link)

---

# Funnynet: Audiovisual Learning of Funny Moments in Videos

By Zhi-Song Liu*, Robin Courant* and Vicky Kalogeiton

ACCV 2022 (Oral, **Best Student Paper Honorable mention**)

### [Project Page](http://www.lix.polytechnique.fr/vista/projects/2022_accv_liu) | [Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Liu_FunnyNet_Audiovisual_Learning_of_Funny_Moments_in_Videos_ACCV_2022_paper.pdf) | [Data](https://drive.google.com/drive/folders/1ZM6agmEnheiyP0IIrD3Fc7DOubjyu5eO?usp=share_link)

## Dependencies

Python 3.8
OpenCV library
Pytorch 1.12.0
CUDA 11.3

## Environment setup

1. Clone code to your local computer.
```sh
git clone https://github.com/robincourant/FunnyNet.git
cd FunnyNet
```

2. Create working environment.
```sh
conda create --name funnynet -y python=3.8
conda activate funnynet
```

1. Install the dependencies.
```sh
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

1. Run the setup script to intsall all the dependencies.
```
./setup.sh
```

1. Modify in `ext/TimeSformer/timesformer/models/vit_utils.py`
```
from torch._six import container_abcs --> import collections.abc as container_abcs
```

1. Comment `ext/TimeSformer/timesformer/models/resnet_helper.py`
```
from torch.nn.modules.linear import _LinearWithBias
```

## Laughter detection

### Remove voice

```sh
python remove_voice.py DATA_DIR/audio/raw 
```

### Create audio embedding
```sh
python get_embedding.py DATA_DIR/audio/raw -e embedding_name
```

### byol-a (hydra based code)

1. byola/config.yaml to change the config of the finetuning
```sh
python byola/finetuning.py
```


