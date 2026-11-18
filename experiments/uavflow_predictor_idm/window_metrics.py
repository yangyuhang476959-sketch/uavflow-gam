"""Native H1 normal/extra-absorbing window metrics; no small-action filtering."""
import torch
CATEGORIES=('all','normal','extra_absorbing')
def action_window_sums(proposal,final,target,mask,frame_indices,stride=5):
    if frame_indices.ndim!=2 or frame_indices.shape[1]!=2:raise ValueError('Requires native H1 two-anchor frame indices')
    normal=(frame_indices[:,1]-frame_indices[:,0])==stride
    groups=torch.stack((torch.ones_like(normal),normal,~normal)).double()
    weight=mask.float();p=((proposal.float()-target.float()).abs()*weight).flatten(1).sum(1).double();f=((final.float()-target.float()).abs()*weight).flatten(1).sum(1).double();count=weight.flatten(1).sum(1).double()
    groups=groups*count.gt(0).double()[None]
    return torch.stack(((groups*p).sum(1),(groups*f).sum(1),(groups*count).sum(1),groups.sum(1)),dim=1)
def action_window_results(sums,prefix=''):
    result={}
    for name,row in zip(CATEGORIES,sums):
        denom=row[2].clamp_min(1)
        result.update({prefix+name+'_proposal_loss':row[0]/denom,prefix+name+'_refine_loss':row[1]/denom,prefix+name+'_windows':row[3]})
    return result
