# TUSZ Meta-TTT v1 frozen protocol

This namespace rebuilds the experiment on TUSZ v2.0.6 from the public
CBraMod checkpoint. It trains a new detector and new SSL/meta heads. Official
Train is used for development and fitting, Dev for operating-point calibration,
and Eval for the final locked evaluation.

The primary estimand is the paired change from a source checkpoint's frozen
predictions to its record-carry online adaptation predictions. The primary
operating point targets 80% event sensitivity on Dev. Advancement requires no
more than two percentage points sensitivity loss, at least 10% lower false
alarms per hour, no increase in false-alarm time, and at most two seconds added
median delay among commonly detected events.

Online inference always predicts before updating. Updates occur every 30 s
from three non-overlapping 10 s support windows and affect CBraMod blocks 10
and 11. State resets at every EDF boundary. Same-window adaptation is trained
and reported as a separate condition.

The four planned objectives are Band classification, Temporal permutation,
masked-patch reconstruction, and a learned scalar objective. Every objective
uses the same source classifier, patient split, adaptation scope, inner-LR
search, training coverage, and scorer.

## Explicit gradient-alignment ablation

The phrase “gradient alignment” in Stage D means an explicit cosine-gradient
regularizer in Meta training. It is not part of the deployment-time update and
it is not the same operation as PCGrad.

For each Meta episode, using the same adapted parameters in blocks 10 and 11,
compute

\[
g_{\rm ssl}=\nabla_{\theta_{\rm fast}}L_{\rm SSL}(x_{\rm support}),\qquad
g_{\rm cls}=\nabla_{\theta_{\rm fast}}L_{\rm BCE}(x_{\rm query},y_{\rm query}).
\]

The support gradient retains its computation graph so the outer optimizer can
update the auxiliary module. The query labels are available only during
Meta-training; the deployed inner update remains label-free. The alignment
term is

\[
L_{\rm outer}=L_{\rm post\text{-}BCE}
 +\lambda\left(1-\cos(g_{\rm ssl},g_{\rm cls})\right).
\]

The pre-registered conditions are \(\lambda\in\{0,0.1,1.0\}\), where
\(\lambda=0\) is the existing post-BCE Meta baseline. Cosine is computed only
over the adapted backbone parameters; the SSL-only head is excluded from the
vector. Zero or near-zero norms are recorded as invalid and excluded from the
alignment term rather than assigned cosine zero. Each run must log cosine,
negative-cosine fraction, both gradient norms, parameter update norm, the
predicted first-order loss change, and the actual finite-step BCE change.

PCGrad is a separate secondary ablation. It is not enabled by this protocol:
PCGrad projects conflicting task gradients when their dot product is negative,
whereas the cosine term directly trains the updater toward the seizure
classification direction. No PCGrad result may be presented as a cosine-
alignment result.

The machine-readable source of truth is `configs/tusz_meta_ttt_v1.yaml`.
