"""One shared initial policy state; native, identical repeat/sampling protocol."""
import hashlib,json
from pathlib import Path
import torch


def tensor_hash(state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()):
  h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()


def bundle_for(policy):
 tokens=policy.encoder.action_placeholder_tokens+['<|uav_state|>'];tokenizer=policy.encoder.processor.tokenizer
 ids=torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in tokens]);embedding=policy.encoder.qwen.get_input_embeddings().weight
 encoder={n:p.detach().cpu().clone() for n,p in policy.encoder.named_parameters() if p.requires_grad}
 result=dict(version=1,encoder=encoder,pose=policy.state_encoder.state_dict(),projection=policy.projection.state_dict(),head=policy.head.state_dict(),tokens=tokens,embedding_rows=embedding.detach().cpu()[ids].clone())
 result['hashes']={k:tensor_hash(result[k]) for k in ['encoder','pose','projection','head']};return result


def apply_encoder(encoder,state_encoder,bundle):
 params=dict(encoder.named_parameters());expected={n for n,p in params.items() if p.requires_grad}
 if expected!=set(bundle['encoder']):raise ValueError('Shared Qwen trainable parameter names differ')
 with torch.no_grad():
  for name,value in bundle['encoder'].items():params[name].copy_(value)
  ids=torch.tensor([encoder.processor.tokenizer.convert_tokens_to_ids(t) for t in bundle['tokens']],device=encoder.qwen.get_input_embeddings().weight.device)
  encoder.qwen.get_input_embeddings().weight.index_copy_(0,ids,bundle['embedding_rows'].to(encoder.qwen.get_input_embeddings().weight))
 state_encoder.load_state_dict(bundle['pose'],strict=True)
 actual={n:p for n,p in encoder.named_parameters() if p.requires_grad}
 if tensor_hash(actual)!=bundle['hashes']['encoder'] or tensor_hash(state_encoder.state_dict())!=bundle['hashes']['pose']:raise AssertionError('Shared Qwen/Pose initialization mismatch')


def apply_pure(policy,path):
 bundle=torch.load(path,map_location='cpu',weights_only=False)
 apply_encoder(policy.encoder,policy.state_encoder,bundle);policy.projection.load_state_dict(bundle['projection'],strict=True);policy.head.load_state_dict(bundle['head'],strict=True)
 for name,module in [('projection',policy.projection),('head',policy.head)]:
  if tensor_hash(module.state_dict())!=bundle['hashes'][name]:raise AssertionError(name+' initial mismatch')
 return bundle['hashes']


def apply_warp(model,conditioner,path):
 bundle=torch.load(path,map_location='cpu',weights_only=False)
 apply_encoder(conditioner.encoder,conditioner.state_encoder,bundle)
 model.rgbwarp_correction.proposal_projection.load_state_dict(bundle['projection'],strict=True)
 for head in [model.parallel_action_head,model.parallel_action_correction_head]:
  if not head.decoder_step_embedding:raise ValueError('Matched head requires ActionHeadV2')
  head.decoder.load_state_dict(bundle['head'],strict=True)
  if tensor_hash(head.decoder.state_dict())!=bundle['hashes']['head']:raise AssertionError('Warp head initial mismatch')
 if tensor_hash(model.rgbwarp_correction.proposal_projection.state_dict())!=bundle['hashes']['projection']:raise AssertionError('Warp projection initial mismatch')
 return bundle['hashes']


def sources(warp_config,metadata_only=False):
 from omegaconf import OmegaConf
 from experiments.uavflow_pure_stop_probe.data import build_native
 cfg=OmegaConf.to_container(OmegaConf.load(warp_config),resolve=True);settings=dict(cfg['dataset']);split=json.loads(Path(settings.pop('split_file')).read_text());result=[]
 for is_eval,key in [(False,'train_episode_ids'),(True,'eval_episode_ids')]:
  dc=dict(settings,include_episodes=split[key]);result.append(build_native(dc,is_eval,metadata_only=metadata_only))
 return result


def compact(payload):
 cfg=payload.get('config',{});reference=cfg.get('common_init_file') or cfg.get('matched_common_init')
 if reference:
  from .checkpoint_policy import full_epochs_required
  if full_epochs_required(Path(reference).parent):return dict(payload)
 result=dict(payload)
 for k in ['optimizer','scaler','rng','numerics']:result.pop(k,None)
 result.update(weights_only=True,resume_supported=False)
 return result
