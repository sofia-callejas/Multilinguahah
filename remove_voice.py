"""
This script detects laughter within all audio files contained in the directory
`root_dir/audio/raw`, and save one pickle file for each audio file with
laughter timecodes in the directory `root_dir/audio/laughter`.
"""


import argparse
import os
import os.path as osp
import pickle

from laughter_detection.core.voice_remover import VoiceRemover


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="Path to the root of FunnyNet dataset"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    root_dir = args.root_dir
    parent_dir = os.path.dirname(root_dir) 
    diff_dir = os.path.join(parent_dir, "diff")
    laughter_detector = VoiceRemover(root_dir)

    for filename in os.listdir(root_dir):
        input_path = os.path.join(root_dir, filename)
        diff_path = os.path.join(diff_dir, filename)

        if os.path.isfile(input_path) and not os.path.exists(diff_path):
            laughter_detector._get_diff(audio_filename=filename)
