"""Change instruction wording after native sampling, never episode membership."""
import hashlib
import json
import re
from pathlib import Path


def repair_raw_target_word(raw,reviewed):
    """Apply the seven reviewed person/dog fixes without rewriting the sentence."""
    pattern=r'\b(?:person|dog)\b'
    old=re.findall(pattern,raw)
    new=re.findall(pattern,reviewed)
    if len(old)!=1 or len(new)!=1:
        raise ValueError('Expected one person/dog noun in the reviewed raw repair')
    return re.sub(pattern,new[0],raw,count=1)


def apply_instruction_map(dataset,path,expected_sha256=None):
    data=Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest()!=str(expected_sha256):
        raise ValueError('Raw instruction map checksum changed')
    mapping=json.loads(data)
    missing=set(dataset.logs)-set(mapping)
    if missing:raise ValueError('Missing raw instructions: '+str(len(missing)))
    if any(not isinstance(mapping[k],str) or not mapping[k].strip() for k in dataset.logs):
        raise ValueError('Raw instructions must be nonempty strings')
    # Keep the same reviewed semantic repairs in both arms. The released raw
    # column predates these fixes (e.g. person/dog mislabeled), so applying it
    # blindly would introduce label errors in addition to changing wording.
    repairs=getattr(dataset,'instruction_overrides',{})
    values={k:repair_raw_target_word(mapping[k],repairs[k]) if k in repairs else mapping[k]
            for k in dataset.logs}
    changed=sum(text!=values[k] for k,(_,text) in dataset.logs.items())
    repaired=sum(k in repairs for k in dataset.logs)
    dataset.logs={k:(raw,values[k]) for k,(raw,_) in dataset.logs.items()}
    return dict(episodes=len(dataset.logs),changed_instructions=changed,semantic_repairs_preserved=repaired,sha256=expected_sha256)


def verify_legacy_initialization(model,text,path):
    """Audit rank0 before DDP broadcast without replacing either action head."""
    import torch
    from experiments.uavflow_matched_repeat.common import tensor_hash
    if str(path).endswith('.json'):
        import json
        with open(path) as handle:bundle=json.load(handle)
    else:
        bundle=torch.load(path,map_location='cpu',weights_only=False)
    current=dict(
        encoder={n:p.detach().cpu() for n,p in text.encoder.named_parameters() if p.requires_grad},
        pose=text.state_encoder.state_dict(),
        projection=model.rgbwarp_correction.proposal_projection.state_dict(),
        head=model.parallel_action_head.decoder.state_dict())
    actual={k:tensor_hash(v) for k,v in current.items()}
    if actual!=bundle['hashes']:
        raise ValueError('Legacy constructor initialization changed: '+str(actual))
    if 'final_head' in bundle:
        final_hash=tensor_hash(model.parallel_action_correction_head.decoder.state_dict())
        if final_hash != bundle['final_head']:
            raise ValueError('Final action head initialization changed: '+final_hash)
        actual['final_head']=final_hash
    return actual
