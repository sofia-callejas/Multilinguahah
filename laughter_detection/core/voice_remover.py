from collections import Counter, defaultdict
import os
import os.path as osp
from typing import Dict, List, Tuple

import auditok
import numpy as np
from sklearn.cluster import KMeans
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
import sklearn_crfsuite
from audio_separator.separator import Separator
from demucs.pretrained import get_model
from demucs.apply import apply_model

import librosa
import soundfile as sf
from laughter_detection.core.audio_embedding import AudioEmbedder


class VoiceRemover:
    """Detect all laugthers within an audio track."""

    def __init__(
        self,
        root_dir: str,
    ):
        self.root_dir = root_dir

        # Directory with stereo audio tracks
        self.raw_dir = root_dir

        # Directory with difference of stereo audio tracks
        parent_dir = os.path.dirname(root_dir)  
        self.diff_dir = os.path.join(parent_dir, "diff")
        self.diff_no_music_dir = os.path.join(parent_dir, "diff_no_music")

        if not osp.exists(self.diff_dir):
            os.makedirs(self.diff_dir)
        if not osp.exists(self.diff_no_music_dir):
            os.makedirs(self.diff_no_music_dir)


    

    def _save_stereodiff(
        self, raw_track: torch.Tensor, sample_rate: float, diff_path: str, name:str
    ):
        """Save the difference between stereo channels."""
        #diff_track = (raw_track[1] + raw_track[0])/2
        
        self.sample_rate = sample_rate
        raw_track = np.mean(raw_track.numpy(), axis=0)
        separator = Separator()
        separator.load_model()
        output_names = {
            "Vocals": "vocals_output",
            "Instrumental": name,
        }
        file= osp.join(self.raw_dir, name + ".wav")
        outputs = separator.separate(file,output_names)
        import shutil

        shutil.move(name + ".wav", diff_path)
        if os.path.exists("vocals_output.wav"):
            os.remove("vocals_output.wav")
        
        self.sample_rate = sample_rate

    def _save_stereodiff_no_music(
        self, raw_track: torch.Tensor, sample_rate: float, diff_path: str, name:str
    ):
        """Save the difference between stereo channels."""
        #diff_track = (raw_track[1] + raw_track[0])/2
        
        self.sample_rate = sample_rate
        #energy_threshold = 0.2
        if isinstance(raw_track, np.ndarray):
            wav = torch.tensor(raw_track, dtype=torch.float32)
        else:
            wav = raw_track

        if wav.ndim == 1:
            # Mono -> stereo
            wav = wav.unsqueeze(0).repeat(2, 1)
        elif wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
        

        wav = wav.unsqueeze(0) 

        model = get_model("htdemucs") 
        model.to("cpu")
        model.eval()


        with torch.no_grad():
            sources = apply_model(model, wav, split=True, progress=True)[0]

        drums, bass, other, vocals = sources

        other_stem = other

        other_stem = other_stem.squeeze().cpu().numpy().T  # remove batch dim, convert to numpy
        sf.write(diff_path, other_stem, sample_rate)

        exit()


        shutil.move(name + ".wav", diff_path)
        if os.path.exists("vocals_output.wav"):
            os.remove("vocals_output.wav")
        
        self.sample_rate = sample_rate

    def _get_diff_no_music(self, audio_filename: str):
        """Get non-silent segment tiomecodes of the given audio file."""
        # Load raw audio tracks
        raw_path = osp.join(self.raw_dir, audio_filename)
        raw_track, sample_rate = torchaudio.load(raw_path)
        n_channels = raw_track.shape[0]

        diff_path = osp.join(self.diff_no_music_dir, audio_filename[:-4] + ".wav")

        self._save_stereodiff_no_music(raw_track, sample_rate, diff_path,audio_filename[:-4])


    def _get_diff(self, audio_filename: str):
        """Get non-silent segment tiomecodes of the given audio file."""
        # Load raw audio tracks
        raw_path = osp.join(self.raw_dir, audio_filename)
        raw_track, sample_rate = torchaudio.load(raw_path)
        n_channels = raw_track.shape[0]

        diff_path = osp.join(self.diff_dir, audio_filename[:-4] + ".wav")

        self._save_stereodiff(raw_track, sample_rate, diff_path,audio_filename[:-4])






