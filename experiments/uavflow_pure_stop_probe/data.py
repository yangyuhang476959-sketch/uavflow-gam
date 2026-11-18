"""Use native window boundaries, masks, stop labels and pose conversion."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

# Load the existing native adapter directly for the metadata-only CPU audit.
# robot.__init__ eagerly imports all geometry backends, which is unnecessary
# here and exceeds some no-GPU containers' memory quota. No source is copied.
import importlib.util
import inspect
import sys


def native_adapter(metadata_only=False):
    name = '_pure_stop_probe_native_metadata' if metadata_only else '_pure_stop_probe_native_dataset'
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / 'src/robot/data/uavflow_dataset.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        if metadata_only:
            # Execute the SAME native metadata/sampling code without tensor
            # imports. Images/actions are never requested by this audit path.
            # This is needed on the 2GB no-GPU host; training uses the unmodified
            # module via exec_module below. Future annotations stay deferred.
            import ast
            tree = ast.parse(path.read_text(), filename=str(path))
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.Import) and any(a.name == 'torch' for a in node.names)
                or isinstance(node, ast.ImportFrom) and (node.module or '').startswith('torch'))]
            module.Dataset = object
            exec(compile(tree, str(path), 'exec'), module.__dict__)
        else:
            spec.loader.exec_module(module)
    return sys.modules[name]


def build_native(settings, is_eval, metadata_only=False):
    cls = native_adapter(metadata_only).UAVFlowParquetDataset
    allowed = set(inspect.signature(cls).parameters)
    ignored = {'type', 'n_views', 'camera_keys'}
    unknown = set(settings) - allowed - ignored
    if unknown:
        raise ValueError(f'Unsupported native settings: {unknown}')
    if settings.get('type') != 'uavflow' or settings.get('n_views') != 1 or settings.get('camera_keys') != ['frontcamera']:
        raise ValueError('Probe requires native single-front-camera UAVFlow')
    return cls(**{k:v for k,v in settings.items() if k in allowed}, is_eval=is_eval)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def window_kind(dataset, sample):
    if len(sample) > 2:
        if sample[2] != 'self_pair':
            raise ValueError(f'Unknown sample mode: {sample}')
        return 'terminal_self_pair'
    tid, start = sample
    n = min(len(dataset.traj[tid]), len(dataset.logs[tid][0]))
    # EXACT native definition: all visual anchors physical vs clamped future.
    return 'normal' if start <= n - 1 - dataset.future_steps * dataset.visual_anchor_stride else 'extra_absorbing'


def variant_view(source, stage, variant, self_pairs=5):
    if stage not in (1, 2) or variant not in ('A', 'B'):
        raise ValueError('Invalid stage/variant')
    result = copy.copy(source)
    if any(len(s) != 2 for s in source.samples):
        raise ValueError('Stage-1 source must not contain artificial endpoint repetitions')
    normal = list(source.stat_samples)
    extras = [s for s in source.samples if window_kind(source, s) == 'extra_absorbing']
    if Counter(source.samples) != Counter(normal + extras):
        raise ValueError('Native source sampling is not base + extra absorbing; refusing a different protocol')
    # New protocol: identical samples/order in both arms; only supervision differs.
    result.samples = list(source.samples)
    result.mask_padded_actions = stage == 1 and variant == 'B'
    if stage == 2:
        for tid in sorted(source.traj):
            n = min(len(source.traj[tid]), len(source.logs[tid][0]))
            if n:
                result.samples.extend([(tid, n - 1, 'self_pair')] * self_pairs)
    return result


def physical_action_count(ds, sample):
    """Only indices i < N-1 have an observed pose-to-next-pose transition.

    The native terminal zero at i=N-1 also lacks a physical successor. It is
    padding supervision, whereas a measured zero delta at i<N-1 stays valid.
    """
    tid, start = sample[:2]
    n = min(len(ds.traj[tid]), len(ds.logs[tid][0]))
    return min(ds.chunk_size, max(0, n - 1 - start))


def inventory(ds):
    counts = Counter(window_kind(ds, s) for s in ds.samples)
    positives = sum(s[1] >= len(ds.traj[s[0]]) - 1 for s in ds.samples)
    real = sum(physical_action_count(ds, s) for s in ds.samples)
    total = len(ds.samples) * ds.chunk_size
    masked = bool(getattr(ds, 'mask_padded_actions', False))
    return dict(episodes=len(ds.traj), total_windows=len(ds.samples), normal_windows=counts['normal'],
                extra_absorbing_windows=counts['extra_absorbing'], terminal_self_pair_windows=counts['terminal_self_pair'],
                extra_absorbing_fraction=counts['extra_absorbing'] / max(1, len(ds.samples)),
                stop_positive=positives, stop_negative=len(ds.samples)-positives,
                real_transition_action_slots=real, padded_action_slots=total-real,
                supervised_action_slots=real if masked else total,
                masked_padding_slots=total-real if masked else 0,
                zero_action_supervision_windows=sum(physical_action_count(ds,s)==0 for s in ds.samples) if masked else 0,
                sample_sha256=digest(ds.samples), normal_sha256=digest(ds.stat_samples))


def build_sources(cfg, manifest=None, metadata_only=False):
    cache_path = cfg.get('metadata_cache')
    if cache_path and manifest:
        return cached_sources(cfg, manifest, metadata_only=metadata_only)
    dc = dict(cfg['dataset'])
    if (dc.get('future_steps'), dc.get('visual_anchor_stride'), dc.get('chunk_size'), dc.get('terminal_zero_action')) != (1, 5, 5, True):
        raise ValueError('Probe preserves native two-anchor stride5/chunk5 terminal-zero boundaries')
    forbidden = ['endpoint_repeat_count', 'endpoint_self_pair_count', 'endpoint_self_pair_start_count',
                 'endpoint_self_pair_end_count', 'endpoint_terminal_window_repeat_count',
                 'episode_start_repeat_count', 'episode_start_train_fraction', 'endpoint_absorbing_train_fraction',
                 'stop_soft_target_fraction']
    if any(dc.get(k, 0) != 0 for k in forbidden):
        raise ValueError('Probe forbids endpoint repetition, fractional balancing, and soft labels in source')
    if dc.get('endpoint_absorbing_window_count') != 1 or dc.get('endpoint_absorbing_max_starts') != 5:
        raise ValueError('A must retain the native one-copy / five-start absorbing protocol')
    if dc.get('stop_target_mode') != 'current' or not dc.get('stop_absorbing_positive'):
        raise ValueError('Probe requires current-frame hard terminal labels')
    result = []
    for split in ('train', 'val'):
        settings = dict(dc)
        if manifest:
            settings['include_episodes'] = manifest['episodes'][split]
        ds = build_native(settings, is_eval=split == 'val', metadata_only=metadata_only)
        if manifest and inventory(ds) != manifest['native'][split]:
            raise ValueError(f'{split} sample inventory changed since preparation')
        result.append(ds)
    if set(result[0].traj) & set(result[1].traj):
        raise ValueError('Train/Val episodes overlap')
    if manifest and digest({tid: value for ds in result for tid, value in ds.logs.items()}) != manifest['metadata_sha256']:
        raise ValueError('Episode poses/instructions changed since preparation')
    return result


class ProbeWindows:
    def __init__(self, source):
        self.source = source

    def __getstate__(self):
        # The native adapter is loaded dynamically, and its PyArrow module/open
        # row caches cannot be pickled. Send metadata only to fresh workers.
        return {'source_state':{k:v for k,v in self.source.__dict__.items()
                                if k not in ('pq','_cache','_depth_cache')}}

    def __setstate__(self, state):
        from collections import OrderedDict
        import pyarrow.parquet as pq
        cls=native_adapter().UAVFlowParquetDataset
        self.source=cls.__new__(cls)
        self.source.__dict__.update(state['source_state'])
        self.source.pq=pq
        self.source._cache=OrderedDict();self.source._depth_cache=OrderedDict()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source.samples[index]
        # This Pure experiment consumes ONE current image. Reuse the native
        # image decoder, action builder and observed progress cache without
        # decoding unused F0/future images or constructing past-action inputs.
        import torch
        tid, start = sample[:2]
        mode = sample[2] if len(sample) > 2 else 'normal'
        loc = self.source.traj[tid]
        raw, text = self.source.logs[tid]
        actions = (self.source._zero_actions() if mode == 'self_pair'
                   else self.source._actions(raw, start))[0]
        image = self.source._image(loc[start])[None, None]
        native_mask = torch.ones_like(actions, dtype=torch.bool)
        flags = [j < physical_action_count(self.source, sample) for j in range(self.source.chunk_size)]
        physical_mask = native_mask.new_tensor(flags)[:, None].expand_as(native_mask)
        mask = native_mask & physical_mask if getattr(self.source, 'mask_padded_actions', False) else native_mask
        return dict(images=image, instruction=text, actions=actions, mask=mask,
                    native_mask=native_mask, physical_mask=physical_mask,
                    pose=torch.from_numpy(self.source.pose_progress_states[tid][start].copy()),
                    stop=torch.tensor(float(start >= len(loc)-1), dtype=torch.float32),
                    kind=window_kind(self.source, sample), task=native_adapter()._instruction_task_class(text),
                    episode=tid, start=int(start))


def read_config(path, overrides=()):
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(cfg, resolve=True)


def prepared(cfg):
    root = Path(cfg['prepared_dir'])
    manifest = json.loads((root / 'manifest.json').read_text())
    # Output directories/epoch budgets are adjustable; data/model/norm are locked.
    for key in ('dataset', 'qwen_model', 'normalizer_dir', 'seed'):
        if cfg[key] != manifest['config'][key]:
            raise ValueError(f'Prepared {key} differs; use a new prepared_dir')
    if cfg['stage1'].get('action_supervision') != 'shared_windows_padding_mask_ab' or manifest.get('action_supervision') != 'shared_windows_padding_mask_ab':
        raise ValueError('Wrong A/B supervision protocol; prepare a new mask-contrast manifest')
    stats_path = Path(cfg['normalizer_dir']) / (cfg['dataset']['action_stats_key'] + '.json')
    if hashlib.sha256(stats_path.read_bytes()).hexdigest() != manifest['normalizer_sha256']:
        raise ValueError('Normalizer changed since preparation')
    return manifest


def cached_sources(cfg, manifest, *, metadata_only=False):
    import fcntl
    import pickle
    from collections import OrderedDict
    path = Path(cfg['metadata_cache'])
    path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(cfg['dataset']['parquet_root']).glob('*.parquet'))
    stamp = digest(dict(manifest=manifest, source=Path(__file__).resolve().parents[2].joinpath('src/robot/data/uavflow_dataset.py').read_text(),
                        files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]))
    with path.with_suffix(path.suffix+'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            with path.open('rb') as stream:
                payload = pickle.load(stream)
            if payload['stamp'] != stamp:
                raise ValueError('Dataset metadata cache is stale; select a new metadata_cache path')
            result=[]
            import pyarrow.parquet as pq
            for state in payload['states']:
                ds = native_adapter(metadata_only).UAVFlowParquetDataset.__new__(native_adapter(metadata_only).UAVFlowParquetDataset)
                ds.__dict__.update(state); ds.pq=pq
                ds._cache=OrderedDict();ds._depth_cache=OrderedDict()
                result.append(ds)
            return result
        settings=dict(cfg);settings.pop('metadata_cache')
        result=build_sources(settings,manifest,metadata_only=metadata_only)
        states=[]
        for ds in result:
            state={k:v for k,v in ds.__dict__.items() if k not in ('pq','_cache','_depth_cache')}
            states.append(state)
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(dict(stamp=stamp,states=states),stream,protocol=5)
        temporary.replace(path)
        return result
