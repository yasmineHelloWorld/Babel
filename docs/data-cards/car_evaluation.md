# Car Evaluation dataset

- **Source:** UCI Machine Learning Repository, Car Evaluation (ID 19), cached from the original archive.
- **Rows:** 1,728 hypothetical car purchase scenarios.
- **Columns:** buying price, maintenance price, number of doors, passenger capacity, luggage boot size, and safety. All six are categorical.
- **Label:** acceptability (`unacc`, `acc`, `good`, or `vgood`).
- **Usage:** communication grammar and held-out-attribute-combination generalization experiments. The Speaker observes each row; the Listener receives only the message and predicts selected attributes.
- **Limitations:** synthetic rule-generated data with discrete values and an imbalanced label distribution; held-out-combination results measure this designed domain, not real purchase behavior.
