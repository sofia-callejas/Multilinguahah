import os
import sys
import torch
import torchaudio
import torch.nn.functional as F

from ext.byol_a.byol_a.common import load_yaml_config
from ext.byol_a.byol_a.augmentations import PrecomputedNorm
from ext.byol_a.byol_a.models import AudioNTT2020


class BYOLa(torch.nn.Module):
    def __init__(self, byol_dir: str):
        super().__init__()
        self.byol_dir = byol_dir
        self._config = load_yaml_config(os.path.join(byol_dir, "config.yaml"))

        # Spectrogram frontend
        self.to_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._config.sample_rate,
            n_fft=self._config.n_fft,
            win_length=self._config.win_length,
            hop_length=self._config.hop_length,
            n_mels=self._config.n_mels,
            f_min=self._config.f_min,
            f_max=self._config.f_max,
        )

        # Normalizer (from pretrained stats)
        self.normalizer = PrecomputedNorm([-5.4919195, 5.0389895])

        # Backbone model
        self.model = AudioNTT2020(d=self._config.feature_d)

    def load_weights(self, ckpt_path: str):
        print(f"Loading weights from: {ckpt_path}")
        self.model.load_weight(ckpt_path, "cpu")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            spec = self.to_spectrogram(wav.squeeze(1))  # [B, n_mels, frames]
            spec_db = torchaudio.functional.amplitude_to_DB(
                spec, multiplier=10.0, amin=1e-10, db_multiplier=0
            )
            feat = self.normalizer(spec_db)             # normalize
            feat = feat.unsqueeze(1)                    # ADD CHANNEL DIM -> [B, 1, n_mels, frames]
            emb = self.model(feat)                      # [B, d]
        return emb


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_ckpts.py <pretrained_ckpt.pth> <finetuned_ckpt.pth> <audio.wav>")
        sys.exit(1)

    pretrained_ckpt = sys.argv[1]
    finetuned_ckpt = sys.argv[2]
    wav_path = sys.argv[3]

    byol_dir = "ext/byol_a"  # path to BYOL-A code + config
    wav, sr = torchaudio.load(wav_path)

    # ensure mono + batchify
    if wav.dim() == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif wav.dim() == 2:
        wav = wav.mean(dim=0, keepdim=True).unsqueeze(0)  # stereo → mono
    else:
        wav = wav.unsqueeze(0)

    # Load pretrained model
    model1 = BYOLa(byol_dir)
    model1.load_weights(pretrained_ckpt)
    emb1 = model1(wav)

    # Load finetuned model
    model2 = BYOLa(byol_dir)
    model2.load_weights(finetuned_ckpt)
    emb2 = model2(wav)

    # Compare
    cos_sim = F.cosine_similarity(emb1, emb2).item()
    print("Embedding 1 (pretrained):", emb1[0, :5])  # print first 5 dims
    print("Embedding 2 (finetuned):", emb2[0, :5])
    print(f"Cosine similarity: {cos_sim:.4f}")


if __name__ == "__main__":
    main()
