import argparse
import os

from laughter_detection.core.embedding import Embedding

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str,
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
        subdir = os.path.dirname(root_dir)

        embedding_dir = os.path.join(subdir, "embedding",embedding_name)

        embedding_path = os.path.join(embedding_dir, filename.replace(".wav", ".pt"))
        if not os.path.exists(embedding_path):
            if os.path.isfile(input_path):
                input_path = os.path.join(root_dir, filename)
                laughter_detector._get_embeddings(audio_filename=filename)
