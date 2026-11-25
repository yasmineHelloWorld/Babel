# Congressional Voting Records dataset

- **Source:** UCI Machine Learning Repository, [Congressional Voting Records (ID 105)](https://archive.ics.uci.edu/dataset/105/congressional%2Bvoting%2Brecords), DOI [10.24432/C5C01P](https://doi.org/10.24432/C5C01P). License: CC BY 4.0.
- **Rows:** 435 U.S. House member voting records from 1984.
- **Columns:** 16 categorical issue votes: handicapped-infants, water-project-cost-sharing, adoption-of-the-budget-resolution, physician-fee-freeze, el-salvador-aid, religious-groups-in-schools, anti-satellite-test-ban, aid-to-nicaraguan-contras, mx-missile, immigration, synfuels-corporation-cutback, education-spending, superfund-right-to-sue, crime, duty-free-exports, and export-administration-act-south-africa.
- **Values:** `y` (yea), `n` (nay), or `?` (unknown disposition). Unknown values are preserved as their own categorical value.
- **Label:** party affiliation (`democrat` or `republican`).
- **Usage:** real-world categorical benchmark for rate-accuracy curves and held-out vote-combination generalization. The Speaker observes votes; the Listener receives only the message and predicts party affiliation.
- **Why it fits:** a compact, interpretable categorical feature set and binary categorical label make it suitable for testing whether agents can learn a short communication code. Selected vote-value combinations can be held out to measure generalization.
- **Limitations:** small and historically specific; recorded votes and party affiliations reflect the 1984 U.S. House and should not be treated as representative of current political alignments.
