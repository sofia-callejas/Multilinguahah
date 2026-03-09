import pickle
from typing import List, Tuple

def load_labels(label_path: str) -> List[Tuple[float, float]]:
    """Load a label file and extract laugther timecodes."""

    labels = pickle.load(open(label_path, "rb"))
    true_timecodes = []
    for segment in labels.values():
        if segment[-1][-10:-2].lower() == "laughter":
            true_timecodes.append(segment[:2])

    return sorted(true_timecodes)


def load_preds(pred_path: str) -> List[Tuple[float, float]]:
    """Load a prediction file with laugther timecodes."""
    preds = pickle.load(open(pred_path, "rb"))
    return sorted(preds)
