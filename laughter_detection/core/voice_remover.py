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
import torchaudio.functional as F
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
        #center = (raw_track[0] + raw_track[1]) / 2
        #diff_track = raw_track[0] - center 
        #sf.write(diff_path, diff_track.numpy(), sample_rate)


        #background = diff_track

        #background = F.equalizer_biquad(
        #    background.unsqueeze(0),
        #    sample_rate,
        #    center_freq=400.0,
        #    gain=-6.0,   # softer cut
        #    Q=1.0
        #).squeeze(0)

        #background = F.equalizer_biquad(
        #    background.unsqueeze(0),
        #    sample_rate,
        #    center_freq=2500.0,
        #    gain=-6.0,   # softer cut
        #    Q=1.0
        #).squeeze(0)

        #background = background / (background.abs().max() + 1e-8)
        #sf.write(diff_path, background.numpy(), sample_rate)

        #self.sample_rate = sample_rate

        
        
        self.sample_rate = sample_rate
        #TARGET_SECONDS = 10
        #target_len = TARGET_SECONDS * self.sample_rate
        raw_track = np.mean(raw_track.numpy(), axis=0)
        current_len = raw_track.shape[0]
        #if current_len < target_len:
        #    pad_len = target_len - current_len
        #    raw_track = np.pad(raw_track, (0, pad_len), mode="constant")
        #def pad_to_multiple(x, multiple):
        #    remainder = x.shape[0] % multiple
        #    if remainder == 0:
        #        return x
        #    pad_len = multiple - remainder
        #    return np.pad(x, (0, pad_len), mode="constant")

        #raw_track = pad_to_multiple(raw_track, multiple=1024) 
        temp_file = os.path.join(self.raw_dir, name + ".wav")

        sf.write(temp_file, raw_track, self.sample_rate)
        separator = Separator()
        separator.load_model()
        output_names = {
            "Vocals": "vocals_output",
            "Instrumental": name,
        }
        file= osp.join(self.raw_dir, name + ".wav")
        outputs = separator.separate(temp_file,output_names)
        import shutil
        new_name = name.replace("__", "_")
        if name.startswith("_"):
            name = name[1:]
            shutil.move(name + ".wav", diff_path)
        else:
            shutil.move(new_name + ".wav", diff_path)
        if os.path.exists("vocals_output.wav"):
            os.remove("vocals_output.wav")
        
        #self.sample_rate = sample_rate

    def _save_stereodiff_no_music(
        self, raw_track: torch.Tensor, sample_rate: float, diff_path: str, name:str
    ):
        """Save the difference between stereo channels."""
        #diff_track = (raw_track[1] + raw_track[0])/2
        
        self.sample_rate = sample_rate
        raw_track = np.mean(raw_track.numpy(), axis=0)
        separator = Separator()
        separator.load_model()
        output_names = {
            "Vocals": name,
            "Instrumental": "vocals_output",
        }
        file= osp.join(self.raw_dir, name + ".wav")
        outputs = separator.separate(file,output_names)
        import shutil

        #shutil.move(name + ".wav", diff_path)
        if os.path.exists("vocals_output.wav"):
            os.remove("vocals_output.wav")

        new_name = name.replace("__", "_")
        if name.startswith("_"):
            name = name[1:]
            shutil.move(name + ".wav", diff_path)
        else:
            shutil.move(new_name + ".wav", diff_path)
        
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