"""Human-readable and JSON cards describing the benchmark datasets."""

import json
from pathlib import Path


DATA_CARDS = {
    "congressional_voting_records": """# Congressional Voting Records dataset\n\n- **Source:** UCI Machine Learning Repository, Congressional Voting Records (ID 105), 1984 U.S. House votes.\n- **Rows:** 435 House member voting records.\n- **Columns:** 16 categorical issue votes, each encoded as yea (`y`), nay (`n`), or unknown (`?`).\n- **Label:** party affiliation (`democrat` or `republican`).\n- **Usage:** real-world categorical benchmark for rate-accuracy curves and optional held-out vote-combination generalization. Preserve unknown votes as an explicit category.\n- **Limitations:** small and historically specific; party and voting patterns reflect the 1984 U.S. House and should not be generalized to current political alignments.\n""",
    "car_evaluation": """# Car Evaluation dataset\n\n- **Source:** UCI Machine Learning Repository, Car Evaluation (ID 19), cached from the original archive.\n- **Rows:** 1,728 hypothetical car purchase scenarios.\n- **Columns:** buying price, maintenance price, number of doors, passenger capacity, luggage boot size, and safety. All six are categorical.\n- **Label:** acceptability (`unacc`, `acc`, `good`, or `vgood`).\n- **Usage:** communication grammar and held-out-attribute-combination generalization experiments. The Speaker observes each row; the Listener receives only the message and predicts selected attributes.\n- **Limitations:** synthetic rule-generated data with discrete values and an imbalanced label distribution; held-out-combination results measure this designed domain, not real purchase behavior.\n""",
    "fashion_mnist": """# Fashion-MNIST dataset\n\n- **Source:** Zalando Research Fashion-MNIST (`https://github.com/zalandoresearch/fashion-mnist`).\n- **Rows:** 70,000 grayscale 28x28 clothing images (60,000 train + 10,000 test).\n- **Features:** 784 pixel intensities scaled and projected via PCA fit strictly on training rows.\n- **Label:** 10 clothing categories (`T-shirt/top` through `Ankle boot`).\n- **Usage:** vision-derived rate-accuracy curves at `L = 1` across compact discrete symbol budgets.\n- **Limitations:** low-resolution centered silhouettes; PCA features discard fine spatial texture compared with a full CNN.\n""",
}

DATA_CARD_JSON = {
    "congressional_voting_records": {
        "name": "congressional_voting_records",
        "source": "UCI Machine Learning Repository (ID 105)",
        "rows": 435,
        "features": 16,
        "classes": ["democrat", "republican"],
        "missing_policy": "Preserve '?' as explicit third category",
    },
    "car_evaluation": {
        "name": "car_evaluation",
        "source": "UCI Machine Learning Repository (ID 19)",
        "rows": 1728,
        "features": 6,
        "classes": ["unacc", "acc", "good", "vgood"],
        "note": "Model-generated domain; used for controlled grammar and held-out attribute pairs",
    },
    "fashion_mnist": {
        "name": "fashion_mnist",
        "source": "Zalando Research Fashion-MNIST",
        "rows": 70000,
        "features": 784,
        "classes": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "preprocessing": "StandardScaler + PCA fit strictly on X_train",
    },
}


def write_data_cards(directory: str | Path = "docs/data-cards", *, json_dir: str | Path | None = None) -> tuple[Path, ...]:
    """Write the canonical cards to Markdown files (and optional JSON metadata) and return their paths."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in DATA_CARDS.items():
        path = root / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    if json_dir is not None:
        jroot = Path(json_dir)
        jroot.mkdir(parents=True, exist_ok=True)
        for name, meta in DATA_CARD_JSON.items():
            jpath = jroot / f"{name}_data_card.json"
            jpath.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            paths.append(jpath)
    return tuple(paths)
