"""Frozen T5 with the exact 5->256->2048 action Pose MLP from Warp."""
import torch
from torch import nn
from robot.modeling.conditioning import TextConditioner


class T5PoseConditioner(nn.Module):
    hidden_size=2048
    pose5_enabled=True
    pose_progress_enabled=False
    pose_conditioned=True
    variable_length_tokens=True
    is_multimodal=False

    def __init__(self,model_name,pose_seed=43,projection_seed=46,*,native_text=False):
        super().__init__()
        self.text=TextConditioner(encoder_type='t5',t5_model=model_name,
            proj_dim=768,cache_token_embeddings=True,cache_max_entries=4096,cache_device='cpu')
        self.text.requires_grad_(False)
        self.action_interface_dim=2048
        self.hidden_size=int(self.text.hidden_size) if native_text else 2048
        self.native_text=bool(native_text)
        self.variable_length_tokens=not self.native_text
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(projection_seed)
            self.text_projection=nn.Identity() if native_text else nn.Linear(self.text.hidden_size,self.hidden_size)
            torch.random.default_generator.manual_seed(pose_seed);first=nn.Linear(5,256,bias=True)
            torch.random.default_generator.manual_seed(pose_seed+1);second=nn.Linear(256,2048,bias=True)
            self.state_encoder=nn.Sequential(first,nn.GELU(),second)
        self.register_buffer('state_scale',torch.tensor([1000.,1000.,1000.,180.,1.]))

    def train(self,mode=True):
        super().train(mode)
        self.text.eval()
        return self

    def encode_tokens(self,texts,*,pad_to=None,current_pose=None):
        if current_pose is None or current_pose.shape!=(len(texts),5):
            raise ValueError('T5/Pose requires canonical current [B,5] cm/deg/is_start')
        if not torch.isfinite(current_pose).all():raise ValueError('Nonfinite current Pose')
        if not ((current_pose[:,-1]==0)|(current_pose[:,-1]==1)).all():raise ValueError('is_start must be 0/1')
        length=int(pad_to or 256)
        if length<1:raise ValueError('T5 requires positive text length')
        encoded=self.text.encode_tokens(list(texts),pad_to=length)
        text=(encoded['last_hidden_state'] if self.native_text else self.text_projection(encoded['last_hidden_state'].to(self.text_projection.weight.dtype)))
        pose=self.state_encoder((current_pose.float()/self.state_scale).to(self.state_encoder[0].weight.dtype))
        mask=encoded['attention_mask'].bool()
        return dict(last_hidden_state=text,attention_mask=mask,pose_embedding=pose)
