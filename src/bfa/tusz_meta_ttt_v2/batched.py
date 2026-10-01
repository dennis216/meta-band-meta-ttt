"""Independent patient lanes for a shared source model (experimental runner).

Only patient-independent tensor operations are vectorized. Fast weights and
Armijo decisions retain their own leading lane dimension.
"""
from dataclasses import dataclass

import torch

from .update import InnerStepConfig, parameter_block
from .functional import SplitFunctionalTUSZModel


def enable_second_order_batched_attention():
    # MHA inference fastpath misidentifies BatchedTensor.requires_grad. Its
    # native operator has no backward, so explicitly select the math path.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def lane_features(functional, prefix, parameters):
    return torch.vmap(functional.features_from_prefix, in_dims=(0, 0), randomness='error')(prefix, parameters)


def prepare_mask_lanes(objective, signal, valid, *, prefix_fn, seeds, prefix_unique_fn=None):
    lanes, windows = signal.shape[:2]
    copies=objective.condition_count if getattr(objective,'deduplicate_views',False) else 1
    unique=lanes//copies
    signal=signal[:unique]
    flat = signal.flatten(0, 1)
    mask = objective._mask(flat, seeds[:unique*windows]).reshape(unique, windows, 10)
    indexes = mask.long().argsort(dim=-1, descending=True, stable=True)
    k = objective.masked_positions

    def select(value, positions):
        index = positions[:, :, None, :, None].expand(-1, -1, value.shape[2], -1, value.shape[4])
        return value.gather(3, index).flatten(2)

    spectrum = torch.fft.rfft(signal.float(), dim=-1, norm='forward').abs().square()[..., 1:76]
    scale = select(spectrum, indexes[:, :, k:]).median(-1).values.clamp_min(objective.scale_floor)
    target = select(torch.log1p(spectrum / scale[:, :, None, None, None]), indexes[:, :, :k])
    masked = signal.masked_fill(mask[:, :, None, :, None], 0)
    call=prefix_fn if copies==1 else prefix_unique_fn
    if call is None: raise ValueError('Deduplicated transforms require a unique-prefix callback')
    prefix = call(masked.flatten(0, 1)).reshape(unique, windows, 16, 10, 200)
    if copies>1:
        prefix=prefix.repeat(copies,1,1,1,1)
        target=target.repeat(copies,1,1)
        indexes=indexes.repeat(copies,1,1)
    counts = valid.sum(1).clamp_min(1)

    def loss(feature_fn):
        prediction = select(objective.head(feature_fn(prefix)), indexes[:, :, :k])
        per_window = torch.nn.functional.smooth_l1_loss(prediction, target, reduction='none').mean(2)
        return (per_window * valid).sum(1) / counts

    return loss


def prepare_band_lanes(objective,signal,valid,*,prefix_fn,seeds,prefix_unique_fn=None):
    lanes,windows=signal.shape[:2]
    copies=objective.condition_count if getattr(objective,'deduplicate_views',False) else 1
    unique=lanes//copies
    views,_=objective._batch(signal[:unique].flatten(0,1),seeds[:unique*windows])
    # Original transform order is band -> lane -> window. The tail and shared
    # prefix scheduler need condition-major lane -> band -> window order.
    views=views.reshape(5,unique,windows,16,10,200).permute(1,0,2,3,4,5).reshape(unique*5*windows,16,10,200)
    call=prefix_fn if copies==1 else prefix_unique_fn
    if call is None: raise ValueError('Deduplicated transforms require a unique-prefix callback')
    prefix=call(views).reshape(unique,5*windows,16,10,200)
    if copies>1: prefix=prefix.repeat(copies,1,1,1,1)
    labels=torch.arange(5,device=signal.device).repeat_interleave(windows).repeat(lanes)
    counts=valid.sum(1).clamp_min(1)
    def loss(feature_fn):
        features=feature_fn(prefix).mean(2).flatten(2)
        logits=objective.head(features)
        per_window=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels,reduction='none').reshape(lanes,5,windows).mean(1)
        return (per_window*valid).sum(1)/counts
    return loss


def prepare_ssl_lanes(objective,signal,valid,*,prefix_fn,seeds,prefix_unique_fn=None):
    prepare={'mask':prepare_mask_lanes,'band':prepare_band_lanes}[objective.name]
    return prepare(objective,signal,valid,prefix_fn=prefix_fn,seeds=seeds,prefix_unique_fn=prefix_unique_fn)


@dataclass
class LaneUpdate:
    parameters: dict
    accepted: torch.Tensor
    trial_scale: torch.Tensor
    gradient_norms: torch.Tensor
    nonfinite_batch_rejection: bool = False


def normalized_lane_step(parameters, *, loss_fn, record_initial, config: InnerStepConfig, create_graph=True, learning_rate=None, stop_encoder_through_inner=False):
    names = list(parameters)
    lanes = next(iter(parameters.values())).shape[0]
    gradient_parameters=({n:p.detach().requires_grad_(True) for n,p in parameters.items()} if stop_encoder_through_inner else parameters)
    before = loss_fn(gradient_parameters)
    raw_gradients = torch.autograd.grad(before.sum(), tuple(gradient_parameters.values()), create_graph=create_graph, allow_unused=True)
    if any(g is None for g in raw_gradients):
        raise RuntimeError('Missing lane gradient: batched engine cannot silently skip an encoder parameter')
    gradients = dict(zip(names, raw_gradients, strict=True))
    groups = {}
    for name in names:
        groups.setdefault(parameter_block(name), []).append(name)
    packed, norms, squared_norms, directions = {}, {}, {}, {}
    slope = before.new_zeros(lanes)
    eligible = torch.ones(lanes, device=before.device, dtype=torch.bool)
    for block, ns in groups.items():
        g = torch.cat([gradients[n].reshape(lanes, -1) for n in ns], 1)
        squared_norms[block] = g.float().square().sum(1)
        norm = squared_norms[block].detach().sqrt()
        norms[block] = norm
        eligible = eligible & (norm >= config.degenerate_fraction * config.block_gradient_medians[block])
        packed[block] = g
    norm_matrix = torch.stack(list(norms.values()), 1)
    if not bool(torch.isfinite(norm_matrix).all()):
        # Discard this entire inner graph, including healthy lanes. Selecting a
        # NaN branch with torch.where can still poison double backward (0*NaN).
        # Preserve pre-update weights and expose the conservative batch rejection.
        return LaneUpdate(parameters,torch.zeros_like(eligible),before.new_zeros(lanes),
            torch.nan_to_num(norm_matrix.detach(),nan=0.,posinf=0.,neginf=0.),True)
    for block, ns in groups.items():
        g = packed[block]
        # Inactive padded lanes have zero gradients. sqrt(norm2)**2 has an
        # undefined intermediate derivative at zero; use norm2 directly.
        denominator = (squared_norms[block] + (config.epsilon_fraction * config.block_gradient_medians[block])**2).sqrt()
        direction = (-config.relative_step * config.block_reference_norms[block] * g / denominator[:, None] if learning_rate is None else -learning_rate*g)
        slope = slope + (g * direction).float().sum(1)
        for name, piece in zip(ns, direction.split([parameters[n][0].numel() for n in ns], dim=1), strict=True):
            directions[name] = piece.reshape_as(parameters[name])
    accepted = torch.zeros_like(eligible)
    scales = before.new_zeros(lanes)
    result = parameters
    for scale in config.trial_scales:
        candidate = {n: parameters[n] + scale * directions[n] for n in names}
        valid = eligible & ~accepted
        for block, ns in groups.items():
            drift = torch.cat([(candidate[n] - record_initial[n]).reshape(lanes, -1) for n in ns], 1).float().square().sum(1).sqrt()
            valid = valid & torch.isfinite(drift) & (drift <= config.maximum_record_drift * config.block_reference_norms[block] * (1 + 1e-4))
        with torch.no_grad():
            after = loss_fn(candidate)
        take = valid & torch.isfinite(after) & (after <= before.detach() + config.armijo * scale * slope.detach())
        result = {n: torch.where(take.reshape(lanes, *([1] * (parameters[n].ndim - 1))), candidate[n], result[n]) for n in names}
        accepted = accepted | take
        scales = torch.where(take, scale, scales)
        if bool((accepted | ~eligible).all()):
            break
    return LaneUpdate(result, accepted, scales, norm_matrix.detach())


class EnsembleHead(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads=torch.nn.ModuleList(heads)

    def forward(self, features):
        conditions=len(self.heads)
        if features.shape[0]%conditions:
            raise ValueError('Lane count must be divisible by condition count')
        patients=features.shape[0]//conditions
        dictionaries=[dict(h.named_parameters()) for h in self.heads]
        parameters={n:torch.stack([p[n] for p in dictionaries]).repeat_interleave(patients,0) for n in dictionaries[0]}
        def forward(x,p):
            return torch.func.functional_call(self.heads[0],p,(x,),strict=True)
        return torch.vmap(forward,in_dims=(0,0),randomness='error')(features,parameters)


class ConditionEnsemble:
    """Same S1 frozen prefix, independently optimized suffixes and detectors.

    Caller must supply identical patients/windows/seeds in condition-major order.
    Prefix deduplication relies on this checked scheduling invariant.
    """
    def __init__(self, models):
        self.models=models
        self.functionals=[SplitFunctionalTUSZModel(m,prefix_cuda_graph=True) for m in models]
        self.detector=EnsembleHead([m.detector for m in models])
        self.adaptable_names=self.functionals[0].adaptable_names

    def initial_lane_parameters(self,lanes):
        if lanes%len(self.models): raise ValueError('Unequal patient counts per condition')
        patients=lanes//len(self.models)
        sources=[f.initial_fast_parameters(detach=False) for f in self.functionals]
        return {n:torch.stack([s[n] for s in sources]).repeat_interleave(patients,0) for n in sources[0]}

    def prefix(self,signal):
        if signal.shape[0]%len(self.models): raise ValueError('Unequal prefix windows per condition')
        count=signal.shape[0]//len(self.models)
        prefix=self.functionals[0].prefix(signal[:count])
        return prefix.repeat(len(self.models),1,1,1)

    def features_from_prefix(self,prefix,parameters):
        return self.functionals[0].features_from_prefix(prefix,parameters)

    def prefix_unique(self,signal):
        return self.functionals[0].prefix(signal)

    def detect_lanes(self,features):
        return self.detector(features)


class EnsembleMaskObjective(torch.nn.Module):
    name='mask'
    def __init__(self,objectives,deduplicate_views=False):
        super().__init__()
        self.prototype=objectives[0]
        self.condition_count=len(objectives)
        self.deduplicate_views=deduplicate_views
        self.masked_positions=self.prototype.masked_positions
        self.scale_floor=self.prototype.scale_floor
        self.head=EnsembleHead([o.head for o in objectives])

    def _mask(self,signal,seeds):
        return self.prototype._mask(signal,seeds)


class EnsembleBandObjective(torch.nn.Module):
    name='band'
    def __init__(self,objectives,deduplicate_views=False):
        super().__init__()
        self.prototype=objectives[0]
        self.condition_count=len(objectives)
        self.deduplicate_views=deduplicate_views
        self.head=EnsembleHead([o.head for o in objectives])

    def _batch(self,signal,seeds):
        return self.prototype._batch(signal,seeds)
