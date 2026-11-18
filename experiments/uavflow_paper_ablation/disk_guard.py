"""Opt-in synchronized early save/pause; does not change checkpoint contents."""
import os,shutil
import torch
import torch.distributed as dist

def low_space(output,step,device,distributed=False,rank=0,force=False):
    minimum=float(os.environ.get('UAVFLOW_MIN_FREE_GIB','0'))
    if minimum<=0 or (not force and step%25):return False
    from .checkpoint_io import checkpoint_lock,read_policy,write_policy
    flag=False
    if rank==0:
        with checkpoint_lock():
            policy=read_policy();minimum=float(policy.get('minimum_free_gib',minimum))
            flag=bool(policy.get('pause_requested')) or shutil.disk_usage(output).free < minimum*2**30
            if flag and policy and not policy.get('pause_requested'):
                import time
                policy.update(pause_requested=True,pause_reason='free disk below configured floor',pause_requested_unix=time.time());write_policy(policy)
    if distributed:
        value=torch.tensor(int(flag),dtype=torch.int32,device=device);dist.broadcast(value,src=0);flag=bool(value.item())
    return flag
