# Babel Experimental Results

All tables and metrics below are generated programmatically from saved `runs/summaries/*.json` artifacts.
Historical 1984 U.S. House voting records (**Congressional Voting Records**), vision PCA features (**Fashion-MNIST**),
and controlled rule-generated data (**Car Evaluation**) are reported separately and never conflated.

## Rate-Accuracy Benchmark: `exp1_congressional_rate_accuracy` (`congressional_voting_records`, $L = 1$)

- **Split SHA-256:** `b1f47013dd7f350f3026bd2ac0a7fc0a24f0511e3868a7ad80b99688de5e8b84` | **Seeds:** `[0, 1, 2, 3, 4]`
- **Class balance:** `{'democrat': 267, 'republican': 168}`
- **Validation-selected hyperparameters:** `{'entropy_bonus': 0.01, 'hidden_dim': 64}`
- **Held-out test class coverage:** `{'democrat': 40, 'republican': 25}`

| $K$ | Bits ($\log_2 K$) | Emergent Acc | Emergent Macro-F1 | ROC AUC (OvR) | $I(O; M)$ (bits) | Active Vocab | k-Means VQ Acc | Size-Limited Tree Acc | Oracle (LogReg / MLP) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.00 | 0.985 ± 0.010 | 0.984 ± 0.010 | 0.984 ± 0.010 | 0.863 ± 0.061 | 2.0 ± 0.0 | 0.923 ± 0.000 | 1.000 ± 0.000 | 0.969 / 0.978 |
| 3 | 1.58 | 0.978 ± 0.012 | 0.977 ± 0.013 | 0.977 ± 0.013 | 0.834 ± 0.070 | 2.0 ± 0.0 | 0.877 ± 0.000 | 1.000 ± 0.000 | 0.969 / 0.978 |
| 4 | 2.00 | 0.985 ± 0.013 | 0.984 ± 0.014 | 0.985 ± 0.013 | 0.869 ± 0.079 | 2.0 ± 0.0 | 0.892 ± 0.000 | 1.000 ± 0.000 | 0.969 / 0.978 |
| 6 | 2.58 | 0.985 ± 0.013 | 0.984 ± 0.014 | 0.983 ± 0.014 | 0.862 ± 0.086 | 2.0 ± 0.0 | 0.923 ± 0.010 | 0.969 ± 0.000 | 0.969 / 0.978 |
| 8 | 3.00 | 0.988 ± 0.011 | 0.987 ± 0.012 | 0.987 ± 0.012 | 0.882 ± 0.072 | 2.0 ± 0.0 | 0.929 ± 0.012 | 0.994 ± 0.007 | 0.969 / 0.978 |
| 16 | 4.00 | 0.972 ± 0.011 | 0.971 ± 0.012 | 0.972 ± 0.013 | 0.797 ± 0.063 | 2.0 ± 0.0 | 0.908 ± 0.017 | 0.957 ± 0.006 | 0.969 / 0.978 |

## Grammar & Held-Out Generalization: `exp2_car_evaluation_grammar` (`car_evaluation`, $L = 2$)

- **Split SHA-256:** `c8697405b9405b75e94a8a2d19bfcc78c8d47797981e8c4cbccf43a492a2ce23` | **Held-out support:** 288 rows
- **Held-out coverage:** `{'col_0': {'0': 144, '3': 144}, 'col_5': {'0': 144, '2': 144}}`

| Listener Mode | $K$ | Seen Exact Acc | Held-Out Exact Acc | Held-Out Attr Acc | PosDis | $\rho_{\text{topsim}}$ | Ambiguity | Synonymy | Active Vocab |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| holistic | 3 | 0.500 ± 0.107 | 0.000 ± 0.000 | 0.238 ± 0.159 | 0.467 ± 0.267 | 0.319 ± 0.028 | 8.592 ± 0.330 | 0.000 ± 0.000 | 3.0 ± 0.0 |
| holistic | 4 | 0.640 ± 0.100 | 0.000 ± 0.000 | 0.350 ± 0.176 | 0.525 ± 0.116 | 0.326 ± 0.029 | 8.261 ± 0.239 | 0.000 ± 0.000 | 3.8 ± 0.4 |
| holistic | 6 | 0.760 ± 0.159 | 0.000 ± 0.000 | 0.350 ± 0.196 | 0.412 ± 0.187 | 0.363 ± 0.023 | 7.986 ± 0.415 | 0.000 ± 0.000 | 5.2 ± 0.7 |
| holistic | 9 | 0.780 ± 0.114 | 0.000 ± 0.000 | 0.297 ± 0.109 | 0.276 ± 0.194 | 0.302 ± 0.033 | 7.902 ± 0.297 | 0.000 ± 0.000 | 7.0 ± 1.1 |
| structured | 3 | 0.600 ± 0.088 | 0.007 ± 0.014 | 0.372 ± 0.110 | 0.502 ± 0.107 | 0.359 ± 0.024 | 8.301 ± 0.226 | 0.000 ± 0.000 | 3.0 ± 0.0 |
| structured | 4 | 0.760 ± 0.133 | 0.000 ± 0.000 | 0.377 ± 0.144 | 0.454 ± 0.178 | 0.342 ± 0.056 | 7.982 ± 0.302 | 0.000 ± 0.000 | 4.0 ± 0.0 |
| structured | 6 | 0.700 ± 0.088 | 0.000 ± 0.000 | 0.251 ± 0.069 | 0.284 ± 0.156 | 0.304 ± 0.010 | 8.036 ± 0.229 | 0.000 ± 0.000 | 4.6 ± 0.5 |
| structured | 9 | 0.900 ± 0.088 | 0.002 ± 0.004 | 0.289 ± 0.118 | 0.446 ± 0.141 | 0.348 ± 0.028 | 7.630 ± 0.236 | 0.000 ± 0.000 | 6.4 ± 0.5 |

## Grammar & Held-Out Generalization: `exp2_synthetic_grammar` (`synthetic`, $L = 2$)

- **Split SHA-256:** `synthetic-3x3` | **Held-out support:** 1 rows
- **Held-out coverage:** `{'pairs': [[2, 0]]}`

| Listener Mode | $K$ | Seen Exact Acc | Held-Out Exact Acc | Held-Out Attr Acc | PosDis | $\rho_{\text{topsim}}$ | Ambiguity | Synonymy | Active Vocab |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tabular | 3 | 0.625 ± 0.134 | 0.000 ± 0.000 | 0.100 ± 0.196 | 0.429 ± 0.166 | 0.427 ± 0.075 | 0.734 ± 0.329 | 0.000 ± 0.000 | 3.0 ± 0.0 |
| tabular | 4 | 0.725 ± 0.143 | 0.000 ± 0.000 | 0.200 ± 0.240 | 0.270 ± 0.062 | 0.364 ± 0.066 | 0.461 ± 0.165 | 0.000 ± 0.000 | 4.0 ± 0.0 |
| tabular | 6 | 0.700 ± 0.125 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.187 ± 0.093 | 0.389 ± 0.081 | 0.222 ± 0.138 | 0.000 ± 0.000 | 5.6 ± 0.5 |
| tabular | 9 | 0.675 ± 0.125 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.111 | 0.297 ± 0.042 | 0.089 ± 0.107 | 0.000 ± 0.000 | 7.8 ± 1.0 |
| holistic | 3 | 0.750 ± 0.110 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.134 | 0.376 ± 0.047 | 0.734 ± 0.251 | 0.000 ± 0.000 | 3.0 ± 0.0 |
| holistic | 4 | 0.750 ± 0.077 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.445 ± 0.131 | 0.454 ± 0.067 | 0.700 ± 0.194 | 0.000 ± 0.000 | 3.8 ± 0.4 |
| holistic | 6 | 0.925 ± 0.098 | 0.000 ± 0.000 | 0.200 ± 0.240 | 0.227 ± 0.086 | 0.492 ± 0.089 | 0.283 ± 0.244 | 0.000 ± 0.000 | 5.4 ± 0.8 |
| holistic | 9 | 0.900 ± 0.092 | 0.000 ± 0.000 | 0.200 ± 0.240 | 0.204 ± 0.079 | 0.333 ± 0.055 | 0.372 ± 0.246 | 0.000 ± 0.000 | 6.6 ± 1.0 |
| structured | 3 | 0.800 ± 0.125 | 0.000 ± 0.000 | 0.300 ± 0.240 | 0.486 ± 0.163 | 0.517 ± 0.111 | 0.628 ± 0.258 | 0.000 ± 0.000 | 3.0 ± 0.0 |
| structured | 4 | 0.850 ± 0.120 | 0.000 ± 0.000 | 0.400 ± 0.196 | 0.351 ± 0.190 | 0.476 ± 0.110 | 0.461 ± 0.215 | 0.000 ± 0.000 | 3.6 ± 0.5 |
| structured | 6 | 0.975 ± 0.049 | 0.000 ± 0.000 | 0.100 ± 0.196 | 0.404 ± 0.214 | 0.533 ± 0.077 | 0.283 ± 0.120 | 0.000 ± 0.000 | 5.2 ± 0.4 |
| structured | 9 | 0.950 ± 0.060 | 0.000 ± 0.000 | 0.300 ± 0.240 | 0.373 ± 0.226 | 0.538 ± 0.079 | 0.283 ± 0.183 | 0.000 ± 0.000 | 6.6 ± 1.0 |

## Dialect Cross-Play Matrix: `exp3_dialect_matrix`

| Dataset | $K$ | $L$ | Seeds | Diagonal Mean | Raw Off-Diagonal Mean | Hungarian-Aligned Off-Diagonal Mean |
|---|---:|---:|---:|---:|---:|---:|
| synthetic | 3 | 2 | 8 | 0.653 | 0.089 | 0.317 |
| congressional_voting_records | 4 | 1 | 8 | 0.983 | 0.511 | 0.983 |

## Training Objective Ablation: `exp_ablation_both_reinforce` (`congressional_voting_records`, $K=4, L=1$)

| Training Mode | Test Accuracy | Macro-F1 | Active Vocab |
|---|---:|---:|---:|
| hybrid_ce | 0.978 ± 0.012 | 0.977 ± 0.013 | 2.0 ± 0.0 |
| both_reinforce | 0.766 ± 0.181 | 0.625 ± 0.293 | 2.2 ± 0.7 |

## Rate-Accuracy Benchmark: `exp_heldout_congressional` (`congressional_voting_records`, $L = 1$)

- **Split SHA-256:** `663f02e8376f8b070d9bcb69821c9e54a27739ee53de935dc1b132d06b9c888d` | **Seeds:** `[0, 1, 2, 3, 4]`
- **Class balance:** `{'democrat': 267, 'republican': 168}`
- **Validation-selected hyperparameters:** `{'entropy_bonus': 0.02, 'hidden_dim': 32}`
- **Held-out test class coverage:** `{'democrat': 225, 'republican': 140}`

| $K$ | Bits ($\log_2 K$) | Emergent Acc | Emergent Macro-F1 | ROC AUC (OvR) | $I(O; M)$ (bits) | Active Vocab | k-Means VQ Acc | Size-Limited Tree Acc | Oracle (LogReg / MLP) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.00 | 0.625 ± 0.020 | 0.549 ± 0.046 | 0.563 ± 0.034 | 0.018 ± 0.015 | 2.0 ± 0.0 | 0.616 ± 0.000 | 0.984 ± 0.000 | 0.616 / 0.626 |
| 4 | 2.00 | 0.585 ± 0.042 | 0.483 ± 0.033 | 0.509 ± 0.036 | 0.007 ± 0.006 | 2.0 ± 0.0 | 0.673 ± 0.009 | 0.827 ± 0.004 | 0.616 / 0.626 |
| 8 | 3.00 | 0.592 ± 0.032 | 0.516 ± 0.019 | 0.529 ± 0.023 | 0.006 ± 0.005 | 2.0 ± 0.0 | 0.706 ± 0.064 | 0.742 ± 0.025 | 0.616 / 0.626 |

**Honest Interpretation:** At $K=2$, emergent code achieved rough parity with k-Means VQ (0.625 vs 0.616). At $K=4$, emergent accuracy (0.585) trailed k-Means VQ (0.673). At $K=8$, emergent accuracy (0.592) trailed k-Means VQ (0.706).
