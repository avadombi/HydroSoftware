# DeepDiscover conservative NumPy formulation in HydroSoftware 1.0

HydroSoftware implements a controlled, instantaneous DiscoverUnit variant with
explicit recurrent storage and analytical water allocation. It is informed by
the historical Keras implementation but does not claim numerical equivalence
with its weights or an exact reproduction of every published layer constraint.

## Structure

The default configuration has one DiscoverUnit, one candidate release process,
and eight hidden units. More candidate processes can be requested explicitly.
All configured processes remain active unless process-switch learning is enabled.

## Snow front-end and training-only transformation

The selected snow module partitions precipitation into rainfall and snowfall and
computes melt. Let `R`, `M`, and `PET` denote rainfall, snowmelt, and potential
evapotranspiration. For each input, the mean and standard deviation are computed
from the training prefix only. Constant inputs receive a unit scale.

The transformed forcing vector is

    x_t = ([R_t, M_t, PET_t] - mean_train) / scale_train.

Positive learned input weights are obtained with softplus. Optional input
switches use sigmoid probabilities. When switch learning is disabled, every
switch equals one. With the normalized antecedent storage

    s_t = S_(t-1) / (S_(t-1) + S_ref),

the learned context and hidden state are

    a_t = x_t * softplus(w_input) * p_input,
    h_t = tanh([a_t, s_t] W_h + b_h).

The hidden and output kernels are unconstrained finite parameters. This is the
actual NumPy engine contract and differs deliberately from the non-negative ELU
layers in the historical Keras source.

## Conservative allocation

An auxiliary sigmoid head produces an evapotranspiration fraction:

    f_ET,t = sigmoid(h_t w_ET + b_ET).

Available water and actual evapotranspiration are

    A_t = S_(t-1) + R_t + M_t,
    ET_t = f_ET,t min(PET_t, A_t),
    B_t = A_t - ET_t.

For each candidate process, the raw propensity is

    rho_j,t = softplus(h_t W_q,j + b_q,j).

Optional process switches provide probabilities `p_j`; otherwise every `p_j`
equals one. The analytical allocation is

    D_t = 1 + sum_j(rho_j,t p_j),
    Q_j,t = B_t rho_j,t p_j / D_t,
    S_t = B_t / D_t,
    Q_t = sum_j Q_j,t.

Consequently,

    S_(t-1) + R_t + M_t - ET_t - Q_t - S_t = 0

up to floating-point roundoff. There is no trainable capacity, spill path, or
post-update storage clipping in this variant.

## Objective and training

The selected hydrological objective is evaluated after warm-up. Optional terms
cover process decorrelation, process smoothness, switch binarity, and switch
sparsity. Mass balance is a structural identity and is also monitored. No hidden
kernel L1/L2 penalty is applied by this NumPy engine.

Adam uses exact full-sequence backpropagation through the training prefix,
gradient clipping, validation-based checkpoint selection, and early stopping.
Only the training and validation prefixes are evaluated during epoch selection;
the test suffix is evaluated once after the selected checkpoint is restored.

## Runtime characteristics

The recurrence is implemented as explicit NumPy loops. Runtime therefore grows
approximately linearly with the number of dates, hidden units, candidate
processes, epochs, and snow-coordinate evaluations. It prioritizes transparent
state accounting and deterministic replay over accelerator throughput.
