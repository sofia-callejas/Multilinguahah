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

import librosa
import soundfile as sf
from laughter_detection.core.audio_embedding import AudioEmbedder


class Embedding:
    """Detect all laugthers within an audio track."""

    def __init__(
        self,
        embedding_name: str,
        root_dir: str,
        byol_dir: str = "ext/byol_a",
        batch_size: int = 20,
        num_workers: int = 7,
        num_gpus: int = 1,
        verbose: bool = False,
    ):
        self.root_dir = root_dir

        # Directory with difference of stereo audio tracks
        self.diff_dir = osp.join(root_dir,"diff")
        if not osp.exists(self.diff_dir):
            os.makedirs(self.diff_dir)
        # Directory with audi embedding vectors
        self.embedding_dir = osp.join(root_dir, "embedding", embedding_name)
        if not osp.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)

        # Minimum duration of a valid audio event in seconds
        self.min_dur = 0.8
        # Maximum duration of an event
        self.max_dur = 11
        # Maximum duration of continuous silence in an event
        self.max_silence = 0.1
        # Time offset to add before and after the detected segment
        self.offset = 0.6
        # Detection threshold for stereo audio tracks
        #self.stereo_detection_threshold = 57
        self.stereo_detection_threshold = 57
        #self.stereo_detection_threshold = 35
        # Detection threshold for surround audio tracks
        self.surround_detection_threshold = 45

        # Initilaize the audio embedder
        self.audio_embedder = AudioEmbedder(
            model_name=embedding_name,
            byol_dir=byol_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            num_gpus=num_gpus,
            verbose=verbose,
        )
        

    

    def _detect_nonsilent(
        self, diff_path: str, detection_threshold: int
    ) -> List[Tuple[float, float]]:
        """Detect non-silent audio segments within an audio track."""
        nonsilent_segments = auditok.split(
            input=diff_path,
            min_dur=self.min_dur,
            max_dur=self.max_dur,
            max_silence=self.max_silence,
            energy_threshold=detection_threshold,
        )

        # Get each segment's timecodes and enlarge it with an offset
        nonsilent_timecodes = sorted(
            [
                [r.meta.start - self.offset, r.meta.end + self.offset]
                for r in nonsilent_segments
            ]
        )

        return nonsilent_timecodes

    def _get_nonsilent(self, audio_filename: str):
        """Get non-silent segment tiomecodes of the given audio file."""
        # Load raw audio tracks
        raw_path = osp.join(self.root_dir, audio_filename)
        
        raw_track, sample_rate = torchaudio.load(raw_path)
        n_channels = raw_track.shape[0]

        diff_path = osp.join(self.diff_dir, audio_filename[:-4] + ".wav")
        nonsilent_timecodes = self._detect_nonsilent(diff_path, self.stereo_detection_threshold)

        return nonsilent_timecodes

    def _load_segments(
        self,
        segment_timecodes: List[Tuple[float, float]],
        audio_filename: str,
    ) -> List[torch.Tensor]:
        """Load and extract audio segments within given timecodes."""
        raw_segments = []
        for start_timecode, end_timecode in segment_timecodes:
            start_index = max(int(self.sample_rate * start_timecode), 0)
            duration = int(self.sample_rate * (end_timecode - start_timecode))

            raw_segment, _ = torchaudio.load(
                osp.join(self.root_dir, audio_filename),
                frame_offset=start_index,
                num_frames=duration,
            )
            raw_segments.append(raw_segment[0][None])

        return raw_segments

    @staticmethod
    def _merge_segments(
        segments: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Merge segments sharing a common part."""
        index, lenght = 0, len(segments)
        while (lenght > 1) and (lenght - index > 1):
            # Check if the two consecutive segment share a part
            if max(segments[index]) >= min(segments[index + 1]):
                new_segment = [min(segments[index]), max(segments[index + 1])]
                # Add the merged segment and revove the originals
                segments.pop(index)
                segments.insert(index, new_segment)
                segments.pop(index + 1)

            else:
                index += 1
            lenght = len(segments)

        return segments


    def _get_embeddings(self, audio_filename: str):
        """[DEPRECATED] Detect laughters within a given stereo audio track.

        :param audio_filename: name of the audio track in the stereo directory.
        :param detection_threshold: threshold of detection.
        :return: detected laughter timecodes.
        """
        # Load raw audio tracks
        raw_path = osp.join(self.root_dir, audio_filename)
        parent_dir = os.path.dirname(self.root_dir)
        diff_path = osp.join(parent_dir,"diff",audio_filename)
        raw_track, sample_rate = torchaudio.load(raw_path)
        embedding_filename = osp.join(
                self.embedding_dir , audio_filename[:-4] + ".pt"
            )
        os.makedirs(self.embedding_dir , exist_ok=True)
        self.sample_rate = sample_rate
        n_channels = raw_track.shape[0]

        nonsilent_timecodes = self._detect_nonsilent(
                diff_path, self.stereo_detection_threshold
            )

        # Load and extract all detected non-silent regions
        raw_segments = self._load_segments(nonsilent_timecodes, audio_filename)

        # Compute audio embedding for all detected segments
        audio_embeddings = self.audio_embedder.get_audioembeddings(
            raw_segments, self.sample_rate
            )

        torch.save(audio_embeddings, embedding_filename)


        
