"""Human-readable cards describing the Phase 3 benchmark datasets."""

from pathlib import Path


DATA_CARDS = {
    "congressional_voting_records": """# Congressional Voting Records dataset\n\n- **Source:** UCI Machine Learning Repository, Congressional Voting Records (ID 105), 1984 U.S. House votes.\n- **Rows:** 435 House member voting records.\n- **Columns:** 16 categorical issue votes, each encoded as yea (`y`), nay (`n`), or unknown (`?`).\n- **Label:** party affiliation (`democrat` or `republican`).\n- **Usage:** real-world categorical benchmark for rate-accuracy curves and optional held-out vote-combination generalization. Preserve unknown votes as an explicit category.\n- **Limitations:** small and historically specific; party and voting patterns reflect the 1984 U.S. House and should not be generalized to current political alignments.\n""",
    "car_evaluation": """# Car Evaluation dataset\n\n- **Source:** UCI Machine Learning Repository, Car Evaluation (ID 19), cached from the original archive.\n- **Rows:** 1,728 hypothetical car purchase scenarios.\n- **Columns:** buying price, maintenance price, number of doors, passenger capacity, luggage boot size, and safety. All six are categorical.\n- **Label:** acceptability (`unacc`, `acc`, `good`, or `vgood`).\n- **Usage:** communication grammar and held-out-attribute-combination generalization experiments. The Speaker observes each row; the Listener receives only the message and predicts selected attributes.\n- **Limitations:** synthetic rule-generated data with discrete values and an imbalanced label distribution; held-out-combination results measure this designed domain, not real purchase behavior.\n""",
}


def write_data_cards(directory: str | Path = "docs/data-cards") -> tuple[Path, ...]:
    """Write the canonical cards to Markdown files and return their paths."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in DATA_CARDS.items():
        path = root / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return tuple(paths)
