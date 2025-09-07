"""
This script detects laughter within all audio files contained in the directory
`root_dir/audio/raw`, and save one pickle file for each audio file with
laughter timecodes in the directory `root_dir/audio/laughter`.
"""


import argparse
import os
import os.path as osp
import pickle

from laughter_detection.core.embedding import Embedding

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="Path to the root of FunnyNet dataset"
    )
    parser.add_argument(
        "--embedding-name",
        "-e",
        type=str,
        help="embedding model to use.",
        default="byola",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    embedding_name = args.embedding_name
    root_dir = args.root_dir

    laughter_detector = Embedding(
        embedding_name, root_dir
    )

    for filename in os.listdir(root_dir):
        input_path = os.path.join(root_dir, filename)
        if os.path.isfile(input_path):
            input_path = os.path.join(root_dir, filename)
            laughter_detector._get_embeddings(audio_filename=filename)