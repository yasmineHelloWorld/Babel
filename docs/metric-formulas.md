# Phase 2 metric definitions

All entropies and mutual information values use base-2 logarithms and are reported in bits. Empirical quantities are computed from observed object/message samples without smoothing. If the Speaker is stochastic, pass multiple message draws per object; the metrics pool those draws and repeat each object's labels and ID accordingly.

## Predictive accuracy

Shape and color accuracy are the fractions of examples whose corresponding predicted class is correct. Exact accuracy is the fraction for which both predictions are correct.

## Message statistics

- Message entropy is `H(M) = -sum_m p(m) log2 p(m)` over complete message tuples.
- Position entropy is `H(M_j)` for symbol position `j`.
- Active vocabulary is the number of distinct symbol IDs appearing in any position. The metric also reports the IDs themselves.
- Mutual information is `I(X;Y) = sum_(x,y) p(x,y) log2[p(x,y)/(p(x)p(y))]`. Overall values use the complete message as `X`; positional values use one symbol position. Shape and color labels are evaluated separately.
- Ambiguity is `H(O|M)`, where `O` is object identity. Zero means every observed message identifies one object; larger values mean messages are shared across objects.
- Synonymy is `H(M|O)`. Zero means every object uses one message; larger values mean an object emits multiple messages. For deterministic argmax evaluation this is necessarily zero, so stochastic draws are needed to measure policy synonymy.

These are empirical estimates. MI is nonnegative by construction, including small positive values under independent random codes at finite sample sizes; use oracle/null tests and sample counts when interpreting it.

## Positional disentanglement (defined, not currently reported)

For `L` positions and two target properties `A` and `B`, define normalized MI `NMI(M_j; A) = I(M_j; A)/H(A)` (zero when `H(A)=0`), and likewise for `B`. Define the score as the maximum over the two possible one-to-one position/property assignments of the mean of their two assigned normalized MIs, minus the mean of the two cross-assignment normalized MIs. For messages longer than two positions, first select the best distinct pair of positions, then apply the same assignment score. Report the raw score without clipping; perfect two-position disentanglement is 1 and no positional preference is 0 in expectation under a random code. This statistic is not implemented yet.

## Topographic similarity

For every unordered pair of objects, compute Hamming distance over discrete message symbols and Hamming distance over their binary/categorical feature vectors. Topographic similarity is Spearman's rank correlation between these two pairwise distance lists, using average ranks for ties. It is 0 when either distance list has zero variance, and requires at least three objects. The current helper uses this definition and should be interpreted with the chosen feature encoding in mind.
