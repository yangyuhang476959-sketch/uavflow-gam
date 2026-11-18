"""IndoorUAV-VLA: letterboxed RGB/depth and current-anchor local pose targets.

An action chunk contains poses at t+1,...,t+K, ALL expressed relative to t.
It must not be integrated as K successive motion increments.
"""
from collections import OrderedDict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


ACTION_REPRESENTATION = 'chunk_anchor_local'


def read_json(path):
    raw = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'gbk'):
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unsupported JSON encoding: {path}')


def heading_basis(yaw_degrees):
    """IndoorUAV raw [x,y,height] from [forward,right,up].

    Verified against real_action/posture and RGB/depth reprojection. IndoorUAV
    yaw=0 faces raw -Y; increasing yaw turns toward raw +X.
    """
    angle = np.deg2rad(float(yaw_degrees))
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[s, c, 0.], [-c, s, 0.], [0., 0., 1.]], np.float64)


def cumulative_local_targets(poses, anchor, chunk_size=5):
    """Metres/radians; all future targets use the same anchor origin/axes."""
    poses = np.asarray(poses, dtype=np.float64)
    if not 0 <= int(anchor) < len(poses):
        raise IndexError(anchor)
    indices = np.minimum(np.arange(1, chunk_size+1)+int(anchor), len(poses)-1)
    delta = poses[indices, :3]-poses[anchor, :3]
    xyz = delta @ heading_basis(poses[anchor, 3])
    yaw = np.deg2rad((poses[indices, 3]-poses[anchor, 3]+180.) % 360.-180.)
    return np.concatenate((xyz, yaw[:, None]), axis=1).astype(np.float32)


def cumulative_targets_to_world(current_pose, actions):
    """Controller adapter: independently decode each target, without summing."""
    current = np.asarray(current_pose, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    xyz = actions[..., :3] @ heading_basis(current[3]).T + current[:3]
    yaw = (current[3]+np.rad2deg(actions[..., 3])) % 360.
    return np.concatenate((xyz, yaw[..., None]), axis=-1)


def letterbox_layout(source_hw, target_hw=(224,224), hfov_degrees=90.):
    """Same size truncation and symmetric black padding as openpi resize_with_pad."""
    sh, sw = map(int, source_hw)
    th, tw = map(int, target_hw)
    ratio = max(sw/tw, sh/th)
    rh, rw = int(sh/ratio), int(sw/ratio)
    top, left = (th-rh)//2, (tw-rw)//2
    focal = sw/(2*math.tan(math.radians(float(hfov_degrees))/2))
    sx, sy = rw/sw, rh/sh
    # Pixel-centre resize convention, followed by the actual padding offsets.
    k = np.array([[focal*sx,0.,sw*.5*sx-.5+left],
                  [0.,focal*sy,sh*.5*sy-.5+top],[0.,0.,1.]],np.float32)
    valid = np.zeros((th,tw),dtype=bool)
    valid[top:top+rh,left:left+rw] = True
    return (rh,rw,top,left), k, valid


def letterbox_rgb(rgb, target_hw=(224,224)):
    rgb = np.asarray(rgb,dtype=np.uint8)
    layout, _, valid = letterbox_layout(rgb.shape[:2],target_hw)
    rh,rw,top,left = layout
    image = Image.fromarray(rgb).resize((rw,rh),Image.Resampling.BILINEAR)
    padded = np.zeros((*target_hw,3),np.uint8)
    padded[top:top+rh,left:left+rw] = image
    return torch.from_numpy(padded).permute(2,0,1).float()/255., torch.from_numpy(valid)


def letterbox_depth(raw, target_hw, unit_scale_m, depth_kind='z', hfov_degrees=90.):
    if raw.dtype != np.uint16:
        raise ValueError(f'Expected uint16 metric depth PNG, got {raw.dtype}')
    layout,k,content = letterbox_layout(raw.shape,target_hw,hfov_degrees)
    rh,rw,top,left = layout
    meters = raw.astype(np.float32)*float(unit_scale_m)
    valid = (raw>0)&(raw<np.iinfo(np.uint16).max)
    if depth_kind == 'ray':
        h,w = raw.shape
        y,x = np.mgrid[:h,:w]
        f = w/(2*math.tan(math.radians(hfov_degrees)/2))
        meters /= np.sqrt(1+((x-(w-1)/2)/f)**2+((y-(h-1)/2)/f)**2)
    elif depth_kind != 'z':
        raise ValueError('Depth kind must be explicitly verified as z or ray')
    resized = np.asarray(Image.fromarray(meters).resize((rw,rh),Image.Resampling.NEAREST))
    resized_valid = np.asarray(Image.fromarray(valid).resize((rw,rh),Image.Resampling.NEAREST))
    output = np.zeros(target_hw,np.float32)
    mask = np.zeros(target_hw,bool)
    output[top:top+rh,left:left+rw] = resized
    mask[top:top+rh,left:left+rw] = resized_valid
    output[~mask] = 0
    return torch.from_numpy(output),torch.from_numpy(mask),torch.from_numpy(k),torch.from_numpy(content)


class IndoorUAVVLADataset(Dataset):
    """Prepared manifest owns the train/val grouping and official test exclusion."""
    def __init__(self, manifest, data_root, *, is_eval=False, include_episodes=None,
                 image_size=(224,224), chunk_size=5, visual_anchor_stride=5,
                 depth_unit_scale_m=None, depth_kind=None, hfov_degrees=90.,
                 max_trajectories=None, **unused):
        if int(chunk_size)!=5 or int(visual_anchor_stride)!=5:
            raise ValueError('Current IndoorUAV Warp recipe requires chunk5/stride5')
        if depth_unit_scale_m is None or depth_kind not in {'z','ray'}:
            raise ValueError('Set verified depth_unit_scale_m and depth_kind before loading')
        self.root = Path(data_root)
        if unused.get('manifest_sha256'):
            import hashlib
            if hashlib.sha256(Path(manifest).read_bytes()).hexdigest()!=unused['manifest_sha256']:
                raise ValueError('Prepared IndoorUAV manifest changed since configuration was generated')
        self.manifest = read_json(manifest)
        self.dataset_name = self.action_stats_key = 'indooruav_vla'
        self.action_dim = 4
        self.image_size = tuple(image_size)
        self.chunk_size = int(chunk_size)
        self.visual_anchor_stride = int(visual_anchor_stride)
        self.depth_unit_scale_m = float(depth_unit_scale_m)
        self.depth_kind = depth_kind
        self.hfov_degrees = float(hfov_degrees)
        self.is_eval = bool(is_eval)
        split = 'val' if is_eval else 'train'
        ids = list(self.manifest['splits'][split])
        if include_episodes is not None:
            requested = set(map(str,include_episodes))
            if not requested.issubset(ids):
                raise ValueError('Explicit episode selection crosses prepared split')
            ids = [key for key in ids if key in requested]
        stress_key=unused.get('memory_stress_episode_'+split)
        if stress_key:
            if stress_key not in ids:raise ValueError('Memory stress episode outside prepared split')
            ids=[stress_key]
        elif max_trajectories is not None:
            ids = ids[:int(max_trajectories)]
        self.episodes = {key:self.manifest['episodes'][key] for key in ids}
        self.samples = [(key,start) for key in ids
                        for start in range(len(self.episodes[key]['poses']))]
        if stress_key:
            # Deliberately fill every microbatch with the longest instruction.
            # Scratch memory tests only; normal configs never set these keys.
            self.samples=[(stress_key,0)]*96
        self.stat_samples = list(self.samples)
        self._cache = OrderedDict()
        if not self.samples:
            raise ValueError('Empty IndoorUAV dataset')
        print(f'IndoorUAV-VLA {split}: {len(ids)} tasks / {len(self.samples)} windows; '
              '5 current-anchor local targets, terminal-clamped poses',flush=True)

    def __len__(self):
        return len(self.samples)

    def _frame(self, episode, index):
        key = (episode['trajectory'],episode['source'][0]+int(index))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        trajectory,frame = key
        rgb_path = self.root/'rgb'/trajectory/'screenshots'/f'{frame}.png'
        depth_path = self.root/'depth'/trajectory/'depth_screenshots'/f'{frame}.png'
        with Image.open(rgb_path) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        with Image.open(depth_path) as im:
            raw = np.asarray(im).copy()
        if rgb.shape[:2]!=raw.shape:
            raise ValueError(f'RGB/depth resolution mismatch: {key}')
        image,_ = letterbox_rgb(rgb,self.image_size)
        depth,mask,k,content = letterbox_depth(raw,self.image_size,self.depth_unit_scale_m,
                                              self.depth_kind,self.hfov_degrees)
        value = image,depth,mask,k,content
        self._cache[key] = value
        while len(self._cache)>24:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self,index):
        key,start = self.samples[index]
        episode = self.episodes[key]
        poses = np.asarray(episode['poses'],np.float64)
        anchors = [start,min(start+5,len(poses)-1)]
        loaded = [self._frame(episode,a) for a in anchors]
        images = torch.stack([x[0] for x in loaded])[:,None]
        actions = torch.from_numpy(np.stack([cumulative_local_targets(poses,a) for a in anchors]))
        # Preserve the shared 5D Pose MLP: episode-start-local cm, relative yaw
        # in degrees, and is_start appended by pose5_state in the objective.
        xyz = (poses[anchors,:3]-poses[0,:3]) @ heading_basis(poses[0,3])
        yaw = (poses[anchors,3]-poses[0,3]+180.)%360.-180.
        prompt_pose = torch.tensor(np.column_stack((xyz*100,yaw)),dtype=torch.float32)
        episode_pose = torch.tensor(np.column_stack((xyz,np.sin(np.deg2rad(yaw)),
                                                     np.cos(np.deg2rad(yaw)))),dtype=torch.float32)
        mask = torch.ones_like(actions,dtype=torch.bool)
        real_targets=torch.tensor([[a+k<len(poses) for k in range(1,6)] for a in anchors])
        return dict(current_images=images[0],future_images=images[1:],all_view_images=images,
                    # No F0 image is fed to the current-only policy.
                    episode_first_image=images[0],actions=actions,
                    proprioception=torch.zeros(2,6),task_description=episode['instruction'],
                    episode_pose=episode_pose,openvla_prompt_pose=prompt_pose,
                    start_t=torch.tensor(start),frame_indices=torch.tensor(anchors),
                    camera_keys=['frontcamera'],view_valid_mask=torch.ones(2,1,dtype=torch.bool),
                    context_valid_mask=torch.ones(2,dtype=torch.bool),
                    transition_loss_mask=torch.ones(1,dtype=torch.bool),action_loss_mask=mask,
                    action_real_target_mask=real_targets,
                    past_action_history=torch.zeros_like(actions),
                    past_action_history_mask=torch.zeros_like(mask),dataset_name=self.dataset_name,
                    action_stats_key=self.action_stats_key,episode_id=key,has_action=True,
                    is_self_pair=False,stop_target=torch.tensor([float(a==len(poses)-1) for a in anchors]),
                    gt_depth_meters=torch.stack([x[1] for x in loaded])[:,None],
                    gt_depth_mask=torch.stack([x[2] for x in loaded])[:,None],
                    gt_depth_semantic_mask=torch.zeros(2,1,*self.image_size,dtype=torch.bool),
                    camera_intrinsics=loaded[0][3],
                    image_valid_mask=torch.stack([x[4] for x in loaded])[:,None])

    def compute_action_statistics(self,max_samples=None):
        samples=self.samples if not max_samples or max_samples<0 else self.samples[:max_samples]
        actions=np.concatenate([cumulative_local_targets(self.episodes[key]['poses'],s)
                                for key,s in samples],axis=0)
        stats={name:value.tolist() for name,value in dict(mean=actions.mean(0),std=actions.std(0),
               min=actions.min(0),max=actions.max(0),q01=np.quantile(actions,.01,axis=0),
               q99=np.quantile(actions,.99,axis=0)).items()}
        stats.update(mask=[True]*4,action_representation=ACTION_REPRESENTATION,
                     units=['m','m','m','rad'],source_split='train' if not self.is_eval else 'val')
        return {self.action_stats_key:stats}
