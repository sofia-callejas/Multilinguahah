import os.path as osp
from typing import List

from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
import wav2clip

from ext.byol_a.byol_a.common import load_yaml_config
from ext.byol_a.byol_a.augmentations import PrecomputedNorm
from ext.byol_a.byol_a.models import AudioNTT2020
import sys
import os
sys.path.append(os.path.abspath("ext/unilm/beats"))

from ext.unilm.beats.BEATs import BEATs as BEATsModel, BEATsConfig
EPS = torch.finfo(torch.float).eps


class BYOLa(LightningModule):
    def __init__(self, root_dir: str = "ext/byol_a"):
        super(BYOLa, self).__init__()
        self._root_dir = root_dir
        config_path = osp.join(self._root_dir, "config.yaml")
        self._config = load_yaml_config(config_path)
        self.model = AudioNTT2020(d=self._config.feature_d)

    def load_weights(self):
        self.pretrained_path = osp.join(
            self._root_dir,
            "pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth",
        )
        self.model.load_weight(self.pretrained_path, "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.model(x)
        return audio_embedding

class BYOLav2(LightningModule):
    def __init__(self, root_dir: str = "ext/byol_a"):
        super(BYOLav2, self).__init__()
        self._root_dir = root_dir
        config_path = osp.join(self._root_dir, "config.yaml")
        self._config = load_yaml_config(config_path)
        self.model = AudioNTT2020(d=self._config.feature_d)

    def load_weights(self):
        self.pretrained_path = osp.join(
            self._root_dir,
            "pretrained_weights/BYOLA-NTT2020d2048s64x96-2508171106-e100-bs256-lr0005-rs42.pth",
        )
        self.model.load_weight(self.pretrained_path, "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.model(x)
        return audio_embedding

class BYOLaFinetuning(LightningModule):
    def __init__(self, root_dir: str = "ext/byol_a"):
        super(BYOLaFinetuning, self).__init__()
        self._root_dir = root_dir
        config_path = osp.join(self._root_dir, "config.yaml")
        self._config = load_yaml_config(config_path)
        self.model = AudioNTT2020(d=self._config.feature_d)

    def load_weights(self):
        self.pretrained_path = osp.join(
            self._root_dir,
            "pretrained_weights/encoder_ssl_finetuned.pth",
        )
        self.model.load_weight(self.pretrained_path, "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.model(x)
        return audio_embedding


class Wav2CLIP(LightningModule):
    def __init__(self, root_dir: str = "ext/byol_a"):
        super(Wav2CLIP, self).__init__()
        self.model = wav2clip.get_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio_embedding = wav2clip.embed_audio(x.squeeze().cpu(), self.model)
        audio_embedding = torch.from_numpy(audio_embedding)
        return audio_embedding


class BEATs(LightningModule):
    def __init__(self, root_dir: str = "ext/unilm/beats"):
        super(BEATs,self).__init__()
        self._root_dir = root_dir
        self.pretrained_path = osp.join(self._root_dir,"BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt")
        checkpoint = torch.load(self.pretrained_path)
        self.cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATsModel(self.cfg)
        self.model.load_state_dict(checkpoint["model"])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] <= 400:
            raise ValueError(f"Audio segment too short! Expected > 400 samples, got {x.shape[-1]}")

        padding_mask = torch.zeros(x.shape[0], x.shape[1], device=x.device).bool()
    
        feats, _ = self.model.extract_features(x, padding_mask=padding_mask)
        return feats.mean(dim=1)

class AudioEmbedder:
    """Compute pre-trained audio embeddings.

    :param model_name: embedding model to use
        ("byola", "wav2clip" or "b+w": both).
    :param byol_dir: directory containing BYOL-a weights.
    :param batch_size: size of batches.
    :param num_workers: number of workers.
    :param num_gpus: number of GPUs.
    :param verbose: wether to display verbose or not.
    """

    def __init__(
        self,
        model_name: str = "byola",
        byol_dir: str = "ext/byol_a",
        model_dir: str = "models",
        batch_size: int = 20,
        num_workers: int = 1,
        num_gpus: int = 1,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trainer = Trainer(
            devices=num_gpus,
            accelerator="gpu",
            logger=False,
        )
        
        self.wav2clip_trainer = self.trainer 

        if self.model_name == "byola":
            self.byola_model = BYOLa(byol_dir)
            self.byola_model.load_weights()
            self.to_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.byola_model._config.sample_rate,
                n_fft=self.byola_model._config.n_fft,
                win_length=self.byola_model._config.win_length,
                hop_length=self.byola_model._config.hop_length,
                n_mels=self.byola_model._config.n_mels,
                f_min=self.byola_model._config.f_min,
                f_max=self.byola_model._config.f_max,
            )
            self.normalizer = PrecomputedNorm([-5.4919195, 5.0389895])
        
        elif self.model_name == "byola-v2":
            self.byola_model = BYOLav2(byol_dir)
            self.byola_model.load_weights()

            self.to_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.byola_model._config.sample_rate,
                n_fft=self.byola_model._config.n_fft,
                win_length=self.byola_model._config.win_length,
                hop_length=self.byola_model._config.hop_length,
                n_mels=self.byola_model._config.n_mels,
                f_min=self.byola_model._config.f_min,
                f_max=self.byola_model._config.f_max,
            )
            self.normalizer = PrecomputedNorm([-5.4919195, 5.0389895])
        
        elif self.model_name == "byola-f":
            self.byola_model = BYOLaFinetuning(byol_dir)
            self.byola_model.load_weights()

            self.to_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.byola_model._config.sample_rate,
                n_fft=self.byola_model._config.n_fft,
                win_length=self.byola_model._config.win_length,
                hop_length=self.byola_model._config.hop_length,
                n_mels=self.byola_model._config.n_mels,
                f_min=self.byola_model._config.f_min,
                f_max=self.byola_model._config.f_max,
            )
            self.normalizer = PrecomputedNorm([-5.4919195, 5.0389895])
        
        elif self.model_name.startswith("wav2clip"):
            self.wav2clip_model = Wav2CLIP()
            self.wav2clip_trainer = Trainer(
                devices=num_gpus,
                accelerator="gpu",
                logger=False,
            )
        
        elif self.model_name.startswith("beats"):
            self.beats_model = BEATs()
            self.beats_trainer = Trainer(
                devices=num_gpus,
                accelerator="gpu",
                logger=False,
            )

        elif self.model_name.startswith("b+w"):
            self.wav2clip_model = Wav2CLIP()
            self.byola_model = BYOLa(byol_dir)
            self.byola_model.load_weights()
            self.to_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.byola_model._config.sample_rate,
                n_fft=self.byola_model._config.n_fft,
                win_length=self.byola_model._config.win_length,
                hop_length=self.byola_model._config.hop_length,
                n_mels=self.byola_model._config.n_mels,
                f_min=self.byola_model._config.f_min,
                f_max=self.byola_model._config.f_max,
            )
            self.normalizer = PrecomputedNorm([-5.4919195, 5.0389895])
            
        else:
            raise NameError(f"No model named: {self.model_name}")

    def _preprocess_audio_byola(
        self, audio_segments: List[torch.Tensor], current_samplerate: float
    ) -> List[torch.Tensor]:
        """
        [BYOL-a] Pre-process audio tracks by resampling them and convert them
        to log-mel spectrograms.

        :param audio_segments: (n_samples, n_frames).
        :param current_samplerate: current sample rates of audio tracks.
        :return: processed audio segments.
        """
        resampler = T.Resample(current_samplerate, self.byola_model._config.sample_rate)

        log_spectrograms = [
            self.normalizer((self.to_spectrogram(resampler(segment)) + EPS).log())
            for segment in audio_segments
        ]

        return log_spectrograms

    def _get_byola(
        self, audio_segments: List[torch.Tensor], current_samplerate: float
    ) -> torch.Tensor:
        """Compute BYOL-a audio embedding given a list of audio tracks."""

        if not audio_segments:
            return torch.empty((0, 1280))

        processed_segments = self._preprocess_audio_byola(
            audio_segments, current_samplerate
        )
        
        spec_loader = DataLoader(
            processed_segments,
            batch_size=1,
            num_workers=self.num_workers,
        )
        byola_embeddings = self.trainer.predict(self.byola_model, spec_loader)
        byola_embeddings = torch.stack(byola_embeddings).squeeze().cpu()

        return byola_embeddings

    def _get_wav2clip(self, audio_segments: List[torch.Tensor]) -> torch.Tensor:
        """Compute Wav2CLIP audio embedding given a list of audio tracks."""
        
        if not audio_segments:
            return torch.empty((0, 1280))
    
        raw_loader = DataLoader(
            audio_segments,
            batch_size=1,
            num_workers=self.num_workers,
        )
        
        wav2clip_embeddings = self.wav2clip_trainer.predict(
            self.wav2clip_model, raw_loader
        )
        wav2clip_embeddings = torch.stack(wav2clip_embeddings).squeeze().cpu()
        
        return wav2clip_embeddings

    def _get_beats(self, audio_segments: List[torch.Tensor]) -> torch.Tensor:
        """Compute Wav2CLIP audio embedding given a list of audio tracks."""
        
        if not audio_segments:
            return torch.empty((0, 1280))
    
        raw_loader = DataLoader(
            audio_segments,
            batch_size=1,
            num_workers=self.num_workers,
        )
        
        beats_embeddings = self.wav2clip_trainer.predict(
            self.beats_model, raw_loader
        )
        beats_embeddings = torch.stack(beats_embeddings).squeeze().cpu()
        
        return beats_embeddings

    def get_audioembeddings(
        self,
        audio_segments: List[torch.Tensor],
        current_samplerate: float,
        save_filename: str = None,
    ) -> List[torch.Tensor]:
        """Compute audio embedding given a list of audio tracks."""
        if self.model_name.startswith("byola"):
            audio_embeddings = self._get_byola(audio_segments, current_samplerate)

        elif self.model_name.startswith("byola-v2"):
            audio_embeddings = self._get_byola(audio_segments, current_samplerate)
        
        elif self.model_name.startswith("byola-f"):
            audio_embeddings = self._get_byola(audio_segments, current_samplerate)
        
        elif self.model_name.startswith("wav2clip"):
            audio_embeddings = self._get_wav2clip(audio_segments)
        
        elif self.model_name.startswith("beats"):
            audio_embeddings = self._get_beats(audio_segments)

        elif self.model_name.startswith("b+w"):
            byola_embeddings = self._get_byola(audio_segments, current_samplerate)
            wav2clip_embeddings = self._get_wav2clip(audio_segments)
            audio_embeddings = torch.hstack([byola_embeddings, wav2clip_embeddings])


        if save_filename:
            torch.save(audio_embeddings, save_filename)

        return audio_embeddings
