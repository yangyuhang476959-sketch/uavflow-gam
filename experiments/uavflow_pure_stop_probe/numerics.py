"""Replay FLA autotuner decisions when reusing frozen features/checkpoints.

Different valid Triton tilings can round BF16 computations differently. Model
weights alone therefore do not identify bitwise frozen features. Keep the
per-shape decisions, rather than depending on a warmup selecting the same ones.
This module changes no kernels in another process or another experiment.
"""
import importlib.metadata
import sys
import torch


def _environment():
    versions={'torch':str(torch.__version__),'cuda':torch.version.cuda}
    for name in ('triton','flash-linear-attention','fla-core'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return versions


def _autotuners():
    # CPU protocol tests need not import FLA, Triton or initialize CUDA.
    module=sys.modules.get('triton.runtime.autotuner')
    if module is None:return {}
    result={}
    for name,loaded in list(sys.modules.items()):
        if not name.startswith('fla.') or loaded is None:continue
        for value in list(vars(loaded).values()):
            for _ in range(6):
                if isinstance(value,module.Autotuner):
                    base=value.base_fn
                    result[base.__module__+'.'+base.__name__]=value
                    break
                value=getattr(value,'fn',None)
                if value is None:break
    return result


def snapshot_fla_kernels():
    kernels={}
    for name,fn in sorted(_autotuners().items()):
        if not fn.cache:continue
        entries=[]
        for key,config in fn.cache.items():
            if config.pre_hook is not None:
                raise ValueError('Cannot persist an FLA configuration with a pre-hook')
            entries.append(dict(key=list(key),config=config.all_kwargs()))
        kernels[name]=entries
    return dict(version=1,environment=_environment(),kernels=kernels)


def restore_fla_kernels(plan):
    """Validate the whole plan before replacing this process's per-shape cache."""
    if plan.get('version')!=1 or plan.get('environment')!=_environment():
        raise ValueError('Frozen feature numerical runtime changed; use matching versions')
    available=_autotuners();updates=[]
    for name,entries in plan['kernels'].items():
        if name not in available:raise ValueError(f'Missing recorded FLA kernel: {name}')
        fn=available[name]
        for entry in entries:
            config=next((c for c in fn.configs if c.pre_hook is None and c.all_kwargs()==entry['config']),None)
            if config is None:raise ValueError(f'Unsupported recorded FLA configuration: {name}')
            key=tuple(entry['key'])
            hash(key)  # Only native, flat Triton shape/dtype keys are supported.
            updates.append((fn,key,config))
    for fn,key,config in updates:fn.cache[key]=config
    return len(updates)
