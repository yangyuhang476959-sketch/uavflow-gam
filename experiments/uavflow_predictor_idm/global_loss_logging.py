"""Detached, element-weighted action metrics across ranks and update intervals."""
import torch
import torch.distributed as dist

class GlobalLossWindow:
    def __init__(self,*,current_only=False,future_only=False,final_only=False):
        self.buffer=None;self.updates=0;self.current_only=bool(current_only);self.future_only=bool(future_only);self.final_only=bool(final_only)
    def add(self,values,loss):
        actions=values['_global_log_action_stats'].detach().double()
        keys=['depth_current_total','depth_future_total','depth_current_linear_l1','depth_current_log_l1','depth_current_grad','depth_future_linear_l1','depth_future_log_l1','depth_future_grad']
        # Depth losses follow the trainer's equal-microbatch objective; action
        # metrics use exact valid-element weighting (including variable masks).
        def depth_value(key):
            if (self.current_only and key.startswith('depth_future_')) or (self.future_only and key.startswith('depth_current_')):
                return actions.new_zeros(())
            return values[key].detach().double()
        vector=torch.cat([actions,loss.detach().double().reshape(1),torch.stack([depth_value(k) for k in keys]),actions.new_ones(1),values['action_raw_moments'].detach().double().flatten(),values['raw_l1'].detach().double().reshape(1)])
        window=values.get('_window_action_stats',actions.new_zeros(3,4)).detach().double().flatten()
        stop=values.get('_global_log_stop_stats',actions.new_zeros(8)).detach().double()
        vector=torch.cat((vector,window,stop))
        if self.future_only:
            vector=torch.cat((vector,torch.stack([values[k].detach().double() for k in ['feature','feature_patch','feature_cls','feature_register','feature_cos']])))
        self.buffer=vector if self.buffer is None else self.buffer+vector;self.updates+=1
    def finish(self,distributed=False):
        if self.buffer is None:raise ValueError('Empty loss interval')
        vector=self.buffer.clone()
        if distributed:dist.all_reduce(vector,op=dist.ReduceOp.SUM)
        denom=vector[2].clamp_min(1);updates=vector[12].clamp_min(1)
        proposal=vector[0]/denom;final=vector[1]/denom;current=vector[4]/updates;future=vector[5]/updates
        result=dict(action=(proposal+final)/2,direct_action=proposal,refine_action=final,rgbwarp_proposal_loss=proposal,rgbwarp_refine_loss=final,depth=(current+future)/2,depth_global=(current+future)/2,depth_current_total=current,depth_future_total=future,depth_linear_l1=(vector[6]+vector[9])/(2*updates),depth_log_l1=(vector[7]+vector[10])/(2*updates),depth_grad=(vector[8]+vector[11])/(2*updates),action_raw_moments=vector[13:37].reshape(6,4),raw_l1=vector[37]/updates)
        for i,name in enumerate(['depth_current_linear_l1','depth_current_log_l1','depth_current_grad','depth_future_linear_l1','depth_future_log_l1','depth_future_grad'],6):result[name]=vector[i]/updates
        if self.current_only:
            result.update(depth=current,depth_global=current,
                          depth_linear_l1=vector[6]/updates,depth_log_l1=vector[7]/updates,
                          depth_grad=vector[8]/updates)
            for key in list(result):
                if key.startswith('depth_future_'):result.pop(key)
        if self.future_only:
            result.update(depth=future,depth_global=future,depth_linear_l1=vector[9]/updates,depth_log_l1=vector[10]/updates,depth_grad=vector[11]/updates)
            for key in list(result):
                if key.startswith('depth_current_'):result.pop(key)
        if self.future_only:
            for i,key in enumerate(['feature','feature_patch','feature_cls','feature_register','feature_cos'],58):
                result[key]=vector[i]/updates
            for name,key in [('feature_per_horizon','feature'),('feature_patch_per_horizon','feature_patch'),('feature_cls_per_horizon','feature_cls'),('feature_register_per_horizon','feature_register'),('feature_cos_per_horizon','feature_cos')]:result[name]=result[key].reshape(1)
        if self.final_only:
            result['action']=final
            result.pop('rgbwarp_proposal_loss',None)
        from .window_metrics import action_window_results
        result.update(action_window_results(vector[38:50].reshape(3,4),prefix='rgbwarp_window_'))
        if self.final_only:
            for key in list(result):
                if key.endswith('_proposal_loss'):result.pop(key)
        n=vector[51].clamp_min(1);tp,fp,fn,tn=vector[52:56];positive=(tp+fn).clamp_min(1);negative=(fp+tn).clamp_min(1)
        result.update(stop=vector[50]/n,stop_accuracy=(tp+tn)/n,stop_recall=tp/positive,
                      stop_rate=(tp+fn)/n,stop_positive_probability=vector[56]/positive,
                      stop_negative_probability=vector[57]/negative)
        metadata=dict(log_scope='global_interval',log_updates=self.updates,log_world=dist.get_world_size() if distributed else 1,log_valid_elements=int(vector[2]))
        total=vector[3]/updates;self.buffer=None;self.updates=0
        return result,total,metadata
