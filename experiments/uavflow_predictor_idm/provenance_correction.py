"""Small controller for existing GAM correction; no second geometry decoder."""
import torch
from torch import nn
from robot.modeling.provenance_attention import LocalReliability, shared_residual
from .patch_transport import bilinear_patch_transport, nominal_intrinsics, actions_to_camera_transform
from .support_aware_transport import decode_levels


class ProvenanceCorrection(nn.Module):
    def __init__(self, *, mode, language_dim, visual_dim, config=None):
        super().__init__()
        self.mode, self.config = mode, dict(config or {})
        allowed = {'uncertainty_layers', 'uncertainty_width', 'uncertainty_tau',
                   'uncertainty_weight', 'fov_degrees', 'heading_from_camera',
                   'depth_scale_meters', 'cleanup_kernel', 'min_hole_area',
                   'translation_gate_meters', 'yaw_gate_degrees',
                   'support_detach_action', 'support_detach_depth', 'support_routing', 'residual_merge'}
        if set(self.config)-allowed:
            raise ValueError(f'Unknown provenance settings: {set(self.config)-allowed}')
        required = {'support_detach_action': True, 'support_detach_depth': True,
                    'support_routing': {'action_to_geometry': True, 'geometry_to_action': True, 'cross_action_branch': False},
                    'residual_merge': {'type': 'support_mass'}}
        for key, value in required.items():
            if self.config.get(key, value) != value:
                raise ValueError(f'{key} must be {value} for paired provenance routing')
        self.proposal_projection = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, visual_dim))
        self.reliability = (LocalReliability(visual_dim,
                            layers=self.config.get('uncertainty_layers', [27, 39]),
                            width=int(self.config.get('uncertainty_width', 64)),
                            tau=float(self.config.get('uncertainty_tau', 1.)))
                            if mode == 'uncertain' else None)

    def proposal(self, model, language):
        if language.ndim != 3 or language.shape[1] != model.action_chunk_size:
            raise ValueError('Correction modes require native Qwen five action placeholder hiddens')
        return model.parallel_action_head(self.proposal_projection(language)[:, None, None])

    def support(self, model, observed, proposal, denormalize, *, metric=None, valid=None, intrinsics=None):
        if observed.shape[1:3] != (1, 1) or denormalize is None:
            raise ValueError('Support correction requires H=1/V=1 and ActionNormalizer')
        prefix = 1+int(getattr(model.da3, 'num_register_tokens', 0))
        current = model.da3.propagate_shallow_visual_slots_grad(
            observed, gradient_checkpointing=model.deep_gradient_checkpointing,
            return_layer_patches=model.parallel_current_geometry_read_enabled,
            layer_patch_indices=model.current_geometry_layer_indices,
            layer_patch_mode=model.current_geometry_patch_mode)
        decoded = decode_levels(model.da3, current['deep_levels'], prefix)
        b, _, _, n, _ = observed.shape
        grid = int((n-prefix)**.5)
        if metric is None:
            metric = decoded['depth'].reshape(b, grid*14, grid*14).detach().float()
            metric = metric*float(self.config.get('depth_scale_meters', 100.))
        else:
            metric = metric.detach().float()
        if intrinsics is None:
            intrinsics = nominal_intrinsics(*metric.shape[-2:],
                fov_degrees=float(self.config.get('fov_degrees', 90.)), device=metric.device)
        raw = denormalize(proposal.float()).detach()[:, 0]
        transport = bilinear_patch_transport(metric, intrinsics,
            actions_to_camera_transform(raw, self.config.get('heading_from_camera')),
            valid=valid, cleanup_kernel=int(self.config.get('cleanup_kernel', 5)),
            min_hole_area=int(self.config.get('min_hole_area', 128)))
        return transport, decoded, current

    def residual(self, model, sup, unsup, rho):
        ds, du, alpha = shared_residual(model.parallel_action_correction_head, sup, unsup, rho)
        stats = {'support_residual_norm': ds.detach().float().norm(dim=-1).mean(),
                 'unsupported_residual_norm': du.detach().float().norm(dim=-1).mean()}
        stats['unsupported_residual_ratio'] = stats['unsupported_residual_norm'] / stats['support_residual_norm'].clamp_min(1e-8)
        stats.update(alpha_sup=alpha.mean(), alpha_unsup=(1-alpha).mean(),
                     residual_norm=(ds+du).detach().float().norm(dim=-1).mean())
        return ds+du, ds, du, stats


class JointDeepTeacher:
    """One GT future pass through the existing DA3, with detached targets."""
    def __init__(self, da3, layers):
        self.da3, self.layers = da3, tuple(int(l)-1 for l in layers)

    @torch.no_grad()
    def __call__(self, shallow):
        from .objectives import _base_da3_teacher_without_lora
        with _base_da3_teacher_without_lora(self.da3):
            return self.da3.propagate_shallow_visual_slots_grad(
                shallow.detach(), gradient_checkpointing=False, return_layer_patches=True,
                layer_patch_indices=self.layers, layer_patch_mode='current')


def checkpoint_state(model):
    module = getattr(model, 'provenance_correction', None)
    if module is None:
        return None
    return {'mode': model.correction_mode, 'routing_version': 2, 'config': module.config, 'weights': module.state_dict()}


def load_checkpoint_state(model, saved, *, initialize=False):
    module = getattr(model, 'provenance_correction', None)
    if module is None:
        if saved is not None:
            raise ValueError('Checkpoint uses an explicit correction mode')
        return
    if saved is None:
        if initialize and model.correction_mode in {'pure', 'support_aware', 'uncertain'}:
            print('Legacy GAM warm-start: missing correction weights='
                  f'{list(module.state_dict())}; unexpected=[]; '
                  'VLM proposal projection starts new, shared GAM/DA3/heads are reused. '
                  'This is NOT an exact continuation of the legacy Predictor proposal.', flush=True)
            return
        raise ValueError('Legacy Predictor-proposal checkpoint has no VLM proposal projection; '
                         'not silently interchangeable with pure VLA correction')
    if saved['mode'] != 'pure' and saved.get('routing_version') != 2:
        raise ValueError('Old average-bridge/post-hoc checkpoint is not an exact paired-routing continuation; initialize from pure')
    if not initialize and saved['mode'] != model.correction_mode:
        raise ValueError('Correction mode changed during exact resume')
    if initialize and saved['mode'] not in {'pure', 'support_aware', model.correction_mode}:
        raise ValueError('Unsupported correction-stage migration')
    oldcfg, newcfg = dict(saved['config']), dict(module.config)
    for cfg in (oldcfg, newcfg):
        # Former post-hoc endpoint gates have no role in joint calibration.
        cfg.pop('translation_gate_meters', None); cfg.pop('yaw_gate_degrees', None)
        cfg.setdefault('support_detach_action', True)
        cfg.setdefault('support_detach_depth', True)
        cfg.setdefault('support_routing', {'action_to_geometry': True, 'geometry_to_action': True, 'cross_action_branch': False})
        cfg.setdefault('residual_merge', {'type': 'support_mass'})
    if initialize:
        for key in ('uncertainty_layers', 'uncertainty_width', 'uncertainty_tau', 'uncertainty_weight'):
            oldcfg.pop(key, None); newcfg.pop(key, None)
        if oldcfg != newcfg:
            raise ValueError('Support geometry settings changed during stage initialization')
    elif oldcfg != newcfg:
        raise ValueError('Correction settings changed during exact resume')
    loaded = module.load_state_dict(saved['weights'], strict=False)
    if loaded.unexpected_keys or any(not k.startswith('reliability.') for k in loaded.missing_keys):
        raise ValueError(f'Unexpected correction weights: {loaded}')
    if loaded.missing_keys and not (initialize and model.correction_mode == 'uncertain'):
        raise ValueError(f'Missing correction weights: {loaded}')
    print(f'Correction load: missing={list(loaded.missing_keys)}, unexpected={list(loaded.unexpected_keys)}', flush=True)
