"""Opt-in serialized writes and preallocated emergency full-state storage."""
import os,json,time,fcntl
from pathlib import Path
from contextlib import contextmanager
import torch

@contextmanager
def checkpoint_lock():
    name=os.environ.get('UAVFLOW_CHECKPOINT_LOCK_FILE')
    if not name:
        yield;return
    path=Path(name);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def read_policy():
    name=os.environ.get('UAVFLOW_DISK_POLICY_FILE')
    return json.loads(Path(name).read_text()) if name and Path(name).exists() else {}

def write_policy(policy):
    path=Path(os.environ['UAVFLOW_DISK_POLICY_FILE']);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));temp.write_text(json.dumps(policy,indent=2));temp.replace(path)

def _link(target,destination):
    destination=Path(destination);temp=destination.with_name('.'+destination.name+'.link-'+str(os.getpid()));temp.unlink(missing_ok=True);temp.symlink_to(Path(target).resolve());os.replace(temp,destination)

def emergency_save(payload,destination,policy):
    role=os.environ['UAVFLOW_TRAINING_ROLE'];entry=policy['emergency'][role];reserve=Path(entry['reserve']);final=Path(entry['checkpoint']);step=int(payload['step']);epoch=int(payload['epoch']);batch=int(payload['batch'])
    existing=entry.get('saved')
    if existing:
        budget=int(entry.get('steps_per_epoch',12881))
        normalize=lambda e,b:(e+b//budget,b%budget)
        assert existing['step']==step and normalize(existing['epoch'],existing['batch'])==normalize(epoch,batch),'Cannot overwrite an emergency state after advancing training'
        assert final.is_file()
    else:
        assert reserve.is_file(),'Missing preallocated emergency checkpoint reserve'
        with reserve.open('r+b') as stream:
            stream.seek(0);torch.save(payload,stream);end=stream.tell();assert end<=entry['capacity'],'Emergency checkpoint exceeded reserved capacity';stream.truncate(end);stream.flush();os.fsync(stream.fileno())
        os.replace(reserve,final)
        entry['saved']=dict(step=step,epoch=epoch,batch=batch,bytes=end,time_unix=time.time())
    # All trainers observe this on their next update boundary, even if moving
    # last off the data disk increases reported free space again.
    policy.update(pause_requested=True,pause_reason='5GiB floor or checkpoint temporary-space requirement',pause_requested_unix=policy.get('pause_requested_unix',time.time()));write_policy(policy)
    _link(final,destination)
    last=Path(entry['output'])/'last.pt'
    if Path(destination)!=last:_link(final,last)
    return final

def save_full(payload,destination):
    destination=Path(destination)
    with checkpoint_lock():
        policy=read_policy();role=os.environ.get('UAVFLOW_TRAINING_ROLE')
        if policy and role in policy.get('emergency',{}) and all(key in payload for key in ('step','epoch','batch','optimizer')):
            import shutil
            needed=int(policy['emergency'][role]['capacity'])
            if policy.get('pause_requested') or shutil.disk_usage(destination.parent).free<needed+64*2**20:
                return emergency_save(payload,destination,policy)
        temp=destination.with_name('.'+destination.name+'.tmp-'+str(os.getpid()))
        try:torch.save(payload,temp);os.replace(temp,destination)
        finally:temp.unlink(missing_ok=True)
        return destination

def prepare_policy(root,outputs,*,commit=True):
    """Reserve full states on system disk, outside the space-limited data volume."""
    import shutil
    root=Path(root).resolve();policy_path=root/'disk_policy.json';old=json.loads(policy_path.read_text()) if policy_path.exists() else {};reserve_root=Path('/root/uavflow-emergency-sixgpu-20261109');reserve_root.mkdir(exist_ok=True)
    entries={};recoverable=0
    for role,output in outputs.items():
        output=Path(output).resolve();size=(output/'last.pt').stat().st_size;capacity=size+64*2**20;prior=old.get('emergency',{}).get(role,{})
        if prior.get('saved') and Path(prior['checkpoint']).exists():recoverable+=Path(prior['checkpoint']).stat().st_size
        entries[role]=dict(output=str(output),reserve=str(reserve_root/(role+'.reserve')),checkpoint=str(reserve_root/(role+'_last.pt')),capacity=capacity,steps_per_epoch=12881)
    existing_reserved=sum(Path(e['reserve']).stat().st_size for e in entries.values() if Path(e['reserve']).exists());need=sum(e['capacity'] for e in entries.values());available=shutil.disk_usage(reserve_root).free+existing_reserved+recoverable
    assert available>need+256*2**20,f'System disk lacks emergency capacity: need {need/2**30:.2f}GiB including every active full state'
    if not commit:return dict(minimum_free_gib=5,emergency_reserved_gib=need/2**30)
    # On a later user-authorized resume, return emergency states to data disk
    # before replenishing reserves; every alias remains a complete state.
    for role,prior in old.get('emergency',{}).items():
        final=Path(prior['checkpoint'])
        if not prior.get('saved') or not final.exists():continue
        folder=Path(prior['output']);destination=folder/f'emergency_recovered_{prior["saved"]["step"]:07d}.pt';temp=destination.with_name('.'+destination.name+'.tmp')
        if not destination.exists():shutil.copyfile(final,temp);os.replace(temp,destination)
        assert destination.stat().st_size==final.stat().st_size
        for link in folder.glob('*.pt'):
            if link.is_symlink() and link.resolve()==final.resolve():
                name=link.with_name('.'+link.name+'.recover');name.unlink(missing_ok=True);os.link(destination,name);os.replace(name,link)
        final.unlink()
    for entry in entries.values():
        p=Path(entry['reserve']);fd=os.open(p,os.O_CREAT|os.O_RDWR,0o600)
        try:os.posix_fallocate(fd,0,entry['capacity']);os.ftruncate(fd,entry['capacity'])
        finally:os.close(fd)
    policy=dict(minimum_free_gib=5,pause_requested=False,emergency=entries,emergency_reserved_bytes=need,created_unix=time.time(),checkpoint_lock=str(root/'checkpoint_write.lock'))
    temp=policy_path.with_suffix('.tmp');temp.write_text(json.dumps(policy,indent=2));temp.replace(policy_path);return policy
