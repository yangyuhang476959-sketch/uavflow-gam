"""IndoorUAV Pure Qwen control with the exact Warp24 data and common initial tensors."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from robot.data.dataset import ActionNormalizer
from robot.data.indooruav_dataset import IndoorUAVVLADataset
from experiments.uavflow_qwen_pure.model import PureQwenActionPolicy
from experiments.uavflow_qwen_pure.utils import masked_l1, rng_state, restore_rng
from experiments.uavflow_matched_repeat.common import apply_pure, tensor_hash
from experiments.uavflow_paper_ablation.sampling import Warp24MicrogroupSampler
from experiments.uavflow_predictor_idm.rgbwarp_objectives import pose5_state
from experiments.uavflow_predictor_idm.train import optimizer_batches
from experiments.uavflow_pure_stop_probe.numerics import snapshot_fla_kernels, restore_fla_kernels


def atomic_save(payload, path):
    temp = path.with_name('.' + path.name + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def tensors(batch, device):
    # Only observed current RGB/pose and action labels enter this policy.
    image = batch['current_images'][:, None].to(device, non_blocking=True)
    pose = pose5_state(batch['openvla_prompt_pose'], batch['frame_indices'])[:, 0].to(device)
    target = batch['actions'][:, :1].to(device)
    mask = batch['action_loss_mask'][:, :1].to(device)
    return image, pose, target, mask


def training_mode(model):
    """eval on Qwen keeps LoRA autograd, while disabling language recomputation."""
    model.train()
    if not hasattr(model, 'encoder'):
        return
    qwen = model.encoder.qwen
    qwen.train(getattr(model, '_qwen_language_mode', 'train') == 'train')
    visual = qwen.model.visual
    visual.train()
    count = getattr(model, '_vision_checkpoint_blocks', len(visual.blocks))
    if not 0 <= count <= len(visual.blocks):
        raise ValueError('Invalid number of vision checkpoint blocks')
    for index, block in enumerate(visual.blocks):
        block.gradient_checkpointing = index < count


def resume_config_changes(old, new, allow_budget_change=False):
    """Only the explicitly requested mode/recomputation policy may migrate."""
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    keys = ('qwen_language_mode', 'vision_gradient_checkpointing_blocks')
    changes = {}
    for key in keys:
        default = 'train' if key == 'qwen_language_mode' else 24
        before = a['pure'].pop(key, default)
        after = b['pure'].pop(key, default)
        if before != after:
            changes[key] = [before, after]
    if allow_budget_change:
        for key in ('results_dir','max_epochs','max_steps'):
            before=a['training'].pop(key,None);after=b['training'].pop(key,None)
            if before!=after:changes['training.'+key]=[before,after]
    if a != b:
        raise ValueError('Pure resume cannot change data, LR, batch, initial weights or other configuration')
    return changes


def forward(model, normalizer, batch, device):
    image, pose, truth, mask = tensors(batch, device)
    keys = list(batch['action_stats_key'])
    target = normalizer.normalize(truth, stats_keys=keys)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(image, list(batch['task_description']), pose)
        loss = masked_l1(prediction, target, mask)
    raw = normalizer.denormalize(prediction.float(), stats_keys=keys)
    return loss, raw, truth, mask


@torch.inference_mode()
def evaluate(model, normalizer, loader, device):
    model.eval()
    normalized_sum = count = windows = 0
    moments = torch.zeros(6, 4, device=device, dtype=torch.float64)
    for batch in loader:
        loss, raw, truth, mask = forward(model, normalizer, batch, device)
        n = int(mask.sum())
        normalized_sum += float(loss) * n
        count += n
        windows += raw.shape[0]
        p, t, w = raw.double().reshape(-1, 4), truth.double().reshape(-1, 4), mask.double().reshape(-1, 4)
        moments += torch.stack((w.sum(0), (p*w).sum(0), (p*p*w).sum(0),
                               (t*w).sum(0), (t*t*w).sum(0), ((p-t).abs()*w).sum(0)))
    assert windows == len(loader.dataset) == 24937
    n, ps, pp, ts, tt, ae = moments.cpu()
    result = dict(H1_refine_action=normalized_sum/count,
                  H1_rgbwarp_reference_final_l1=normalized_sum/count,
                  H1_evaluated_windows=windows)
    for i, axis in enumerate(('x', 'y', 'z', 'yaw')):
        result['H1_action_raw_mae_'+axis] = float(ae[i]/n[i])
        result['H1_action_raw_pred_mean_'+axis] = float(ps[i]/n[i])
        result['H1_action_raw_pred_std_'+axis] = float((pp[i]/n[i]-(ps[i]/n[i])**2).clamp_min(0).sqrt())
        result['H1_action_raw_gt_std_'+axis] = float((tt[i]/n[i]-(ts[i]/n[i])**2).clamp_min(0).sqrt())
    training_mode(model)
    return result


def run(config, resume=None, smoke_steps=0, smoke_longest=False, allow_resume_budget_change=False):
    cfg = yaml.safe_load(config.read_text())
    tr = cfg['training']
    out = Path(tr['results_dir'])
    out.mkdir(parents=True, exist_ok=True)
    microbatch = int(tr['batch_size'])
    accumulation = int(tr['gradient_accumulation_steps'])
    assert (microbatch, accumulation) in ((8,3),(24,1))
    assert tr['qwen_lora_lr'] == 1e-5 and tr['lr']*tr['direct_action_head_lr_mult'] == 5e-4
    assert cfg['pure']['action_weight'] == 3 and tr['lr_schedule'] == 'constant'
    assert tr['weight_decay'] == 0 and tr['grad_clip'] == 1
    warp = yaml.safe_load(Path(cfg['pure']['matched_warp_config']).read_text())
    assert cfg['dataset'] == warp['dataset'] and cfg['stage1'] == warp['stage1']
    seed = int(cfg['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    from transformers.models.qwen3_5 import modeling_qwen3_5
    assert modeling_qwen3_5.is_fast_path_available, 'Qwen CUDA fast kernels required'
    train = IndoorUAVVLADataset(**cfg['dataset'], is_eval=False)
    val = IndoorUAVVLADataset(**cfg['dataset'], is_eval=True)
    assert len(train) == 474970 and len(val) == 24937
    model = PureQwenActionPolicy(cfg['stage1']['qwen_model'], decoder_width=1536,
        pose_mode='numeric_token', numeric_pose_dim=5,
        numeric_pose_scale=[1000,1000,1000,180,1], pose_init_seed=43).to(device)
    hashes = apply_pure(model, cfg['pure']['common_init_file'])
    model._qwen_language_mode = cfg['pure'].get('qwen_language_mode','train')
    model._vision_checkpoint_blocks = int(cfg['pure'].get('vision_gradient_checkpointing_blocks',24))
    assert model._qwen_language_mode in ('train','eval')
    training_mode(model)
    assert [(p.in_features,p.out_features,p.bias is not None) for p in model.state_encoder if isinstance(p,torch.nn.Linear)] == [(5,256,True),(256,2048,True)]
    normalizer = ActionNormalizer.from_stats_dir(cfg['stage1']['action_stats_dir'],
                                               [cfg['dataset']['action_stats_key']],
                                               norm_mode='q01_q99')
    groups = model.optimizer_groups(1e-5, 5e-4, pose_lr=5e-4)
    optimizer = torch.optim.AdamW(groups, betas=(.9,.999), eps=1e-8, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    preflight = dict(common_initial_tensor_hashes=hashes,
        manifest_sha256=hashlib.sha256(Path(cfg['dataset']['manifest']).read_bytes()).hexdigest(),
        train_windows=len(train), validation_windows=len(val), train_tasks=len(train.episodes), validation_tasks=len(val.episodes),
        global_batch=24, microbatch=microbatch, accumulation=accumulation, pose_mlp=str(model.state_encoder),
        optimizer=[dict(name=g['name'],lr=g['lr'],elements=sum(p.numel() for p in g['params'])) for g in groups],
        action_normalizer=normalizer.state_dict(),
        expected_steps_per_epoch=19790, expected_total_steps=19790*int(tr['max_epochs']),
        qwen_language_mode=model._qwen_language_mode,
        language_checkpointing_active=model.encoder.qwen.model.language_model.training,
        vision_checkpoint_blocks=sum(block.gradient_checkpointing for block in model.encoder.qwen.model.visual.blocks),
        architecture='Existing PureQwenActionPolicy, common Warp proposal-path head. No DA3 or depth loss.',
        sampling='Warp24MicrogroupSampler world1: original rank0/1/2 micro8 groups, same seed+epoch and dropped tail.')
    # Tensor stats in normalizer state are represented separately in the checkpoint.
    preflight['action_normalizer'] = {k:repr(v) for k,v in preflight['action_normalizer'].items()}
    write_json(out/'preflight.json', preflight)
    print('PURE_PREFLIGHT '+json.dumps(preflight), flush=True)
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    epoch = cursor = step = 0
    best = math.inf
    last_eval_step = None
    if resume:
        ck = torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['format'] == 'indooruav_pure_v1'
        runtime_changes = resume_config_changes(ck['config'],cfg,allow_resume_budget_change)
        if runtime_changes:
            write_json(out/'runtime_policy_transition.json',dict(source=str(resume),step=ck['step'],
                changes=runtime_changes,optimizer_and_cursor_preserved=True,
                note='Earlier segment retains its original runtime policy; no retrospective claim of uniform checkpointing.'))
        model.load_trainable_state(ck['policy_state'])
        optimizer.load_state_dict(ck['optimizer'])
        from experiments.uavflow_pose_factorial.transitions import verify_optimizer_state
        print('PURE_OPTIMIZER_EXACTLY_RESTORED '+json.dumps(
            verify_optimizer_state(ck['optimizer'],optimizer.state_dict())),flush=True)
        scaler.load_state_dict(ck['scaler'])
        restore_fla_kernels(ck['numerics'])
        restore_rng(ck['rng'],device)
        epoch,cursor,step = ck['epoch'],ck['batch'],ck['step']
        best,last_eval_step = ck['best'],ck['last_eval_step']
        print(f'PURE_RESUMED step={step} epoch={epoch} batch={cursor}',flush=True)
    train_sampler = Warp24MicrogroupSampler(train, 1, 0, seed,
        resume_epoch=epoch,resume_batch=cursor,batch_size=24)
    options = dict(num_workers=int(tr['num_workers']),pin_memory=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train,batch_size=microbatch,sampler=train_sampler,drop_last=True,generator=generator,**options)
    val_loader = DataLoader(val,batch_size=7,shuffle=False,generator=torch.Generator().manual_seed(seed),**options)
    longest_batch = None
    if smoke_longest:
        from torch.utils.data import default_collate
        key=max(train.episodes,key=lambda k:len(model.encoder.processor.tokenizer.encode(
            'What action should the uav take to '+train.episodes[k]['instruction'].rstrip('.!?')+'?')))
        index=next(i for i,(k,t) in enumerate(train.samples) if k==key)
        longest_batch=default_collate([train[index]]*microbatch)
        print('SMOKE_LONGEST '+key+' '+train.episodes[key]['instruction'],flush=True)
    start_step = step
    begun = time.time()
    interval = []
    def log(line):
        print(line,flush=True)
        with (out/'train.log').open('a') as f:f.write(line+'\n')
    def progress(status):
        write_json(out/'progress.json',dict(status=status,step=step,epoch=epoch,batch=cursor,
            target_steps=19790*int(tr['max_epochs']),last_eval_step=last_eval_step,best=best if math.isfinite(best) else None,
            elapsed_s=time.time()-begun,updates_this_run=step-start_step))
    def save(path):
        atomic_save(dict(format='indooruav_pure_v1',config=cfg,policy_state=model.trainable_state(),
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(device),
            numerics=snapshot_fla_kernels(),action_normalizer=normalizer.state_dict(),
            epoch=epoch,batch=cursor,step=step,best=best,last_eval_step=last_eval_step),path)
    def validate():
        nonlocal best,last_eval_step
        if last_eval_step == step:return
        progress('validating')
        result = evaluate(model,normalizer,val_loader,device)
        last_eval_step = step
        log(f'[step={step:07d}] eval '+' '.join(f'{k}={v:.8f}' for k,v in result.items()))
        if result['H1_refine_action']<best:
            best=result['H1_refine_action'];save(out/'best_action.pt')
            write_json(out/'best_action.json',dict(value=best,step=step,epoch=epoch,metric='H1_refine_action'))
    training_mode(model)
    while epoch < int(tr['max_epochs']):
        train_sampler.set_epoch(epoch)
        for batches in optimizer_batches(loader,accumulation):
            optimizer.zero_grad(set_to_none=True)
            loss_mean=0
            for micro,batch in enumerate(batches):
                if longest_batch is not None:batch=longest_batch
                # Warp's per-rank generation/augmentation are inactive here;
                # all configured LoRA/dropout/augmentation probabilities are0.
                loss,raw,truth,mask = forward(model,normalizer,batch,device)
                (3*loss/accumulation).backward()
                loss_mean+=float(loss.detach())/accumulation
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1;cursor+=1;interval.append(loss_mean)
            if step%25==0 or smoke_steps:
                log(f'[step={step:07d}] action={sum(interval)/len(interval):.8f} weighted_action={3*sum(interval)/len(interval):.8f} grad={float(grad):.6f} peak_mem_GiB={torch.cuda.max_memory_allocated()/2**30:.4f} global_batch=24')
                interval.clear();progress('training')
            if not smoke_steps and step%5000==0:validate()
            if not smoke_steps and step%1000==0:save(out/'last.pt')
            if smoke_steps and step-start_step>=smoke_steps:
                save(out/'last.pt');progress('smoke_complete');return
        assert cursor==19790
        if not smoke_steps:validate()
        epoch+=1;cursor=0
        save(out/f'epoch_{epoch:02d}.pt');save(out/'last.pt');progress('training')
    progress('complete')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--smoke-longest',action='store_true')
    parser.add_argument('--allow-resume-budget-change',action='store_true')
    args=parser.parse_args()
    run(args.config,args.resume,args.smoke_steps,args.smoke_longest,args.allow_resume_budget_change)
