"""P2BR state interventions, named optimizer checkpoints and RNG isolation.

Historical runners stay unchanged. A single optimizer survives each transition;
only designated backbone momentum buffers are zeroed. Original-head state has
the same policy within each head level and is never forcibly equalized in H0.
"""
import copy
from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch
from torch import nn

from src.training.factorial_config import ARMS, stream_seed


def capture_rng():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(algorithm=n[0],keys=n[1].tolist(),position=n[2],has_gauss=n[3],cached_gaussian=n[4]),
                torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    n=state['numpy']; random.setstate(state['python'])
    np.random.set_state((n['algorithm'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda'])!=torch.cuda.device_count():
            raise ValueError('CUDA RNG topology differs from checkpoint')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


@contextmanager
def isolated_rng(seed=None):
    before=capture_rng()
    try:
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(before)


def cpu_copy(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def state_hash(value):
    """Hash tensor contents and structured state, independent of pickle IDs."""
    h=hashlib.sha256()
    def put(v):
        if torch.is_tensor(v):
            v=v.detach().cpu().contiguous()
            h.update(f'tensor:{v.dtype}:{tuple(v.shape)}:'.encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b'{')
            for key in sorted(v,key=str): put(key); put(v[key])
            h.update(b'}')
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode()+b'[')
            for item in v: put(item)
            h.update(b']')
        else: h.update(f'{type(v).__name__}:{v!r};'.encode())
    put(value)
    return h.hexdigest()


def named_optimizer_state(model,optimizer):
    names={id(p):name for name,p in model.named_parameters()}
    state={}
    for p,value in optimizer.state.items():
        if id(p) not in names: raise ValueError('Orphan optimizer parameter')
        state[names[id(p)]]=cpu_copy(value)
    groups=[]
    for g in optimizer.param_groups:
        groups.append({**{k:cpu_copy(v) for k,v in g.items() if k!='params'},'params':[names[id(p)] for p in g['params']]})
    return {'state':state,'param_groups':groups}


def restore_named_optimizer(model,saved):
    params=dict(model.named_parameters()); groups=[]
    flat=[name for group in saved['param_groups'] for name in group['params']]
    if len(flat)!=len(set(flat)) or set(flat)!=set(params): raise ValueError('Optimizer/model named-parameter coverage mismatch')
    for group in saved['param_groups']:
        groups.append({**{k:copy.deepcopy(v) for k,v in group.items() if k!='params'},'params':[params[n] for n in group['params']]})
    first=saved['param_groups'][0]
    optimizer=torch.optim.SGD(groups,lr=first['lr'],momentum=first['momentum'],weight_decay=first['weight_decay'])
    for name,state in saved['state'].items():
        if name not in params: raise ValueError('Unknown optimizer state key')
        p=params[name]
        optimizer.state[p]={k:v.to(p.device).clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in state.items()}
    return optimizer


def backbone_names(model):
    return [name for name,_ in model.named_parameters() if not name.startswith('head_')]


def head_state(model,optimizer):
    state=named_optimizer_state(model,optimizer)['state']
    return {'weights':{n:cpu_copy(p) for n,p in model.named_parameters() if n.startswith('head_a.')},
            'optimizer':{n:state.get(n,{}) for n,_ in model.named_parameters() if n.startswith('head_a.')}}


def restore_head(model,optimizer,saved):
    params=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in saved['weights'].items(): params[name].copy_(value.to(params[name].device))
    for name,value in saved['optimizer'].items():
        p=params[name]
        optimizer.state[p]={k:v.to(p.device).clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in value.items()}


def backbone_velocity(model,optimizer):
    result={}
    for name,p in model.named_parameters():
        if not name.startswith('head_'):
            value=optimizer.state.get(p,{}).get('momentum_buffer')
            result[name]=cpu_copy(value) if value is not None else None
    return result


def clear_backbone_momentum(model,optimizer):
    for name,p in model.named_parameters():
        if not name.startswith('head_'):
            buffer=optimizer.state.get(p,{}).get('momentum_buffer')
            if buffer is None: raise ValueError(f'Missing trained backbone momentum: {name}')
            buffer.zero_()


def transition(model,optimizer,arm_name,boundary,seed,stash=None):
    """Return stash and exact before/after evidence for 15k or 20k policy."""
    arm=ARMS[arm_name]
    if boundary not in ['enter_middle','return_target']: raise ValueError('Unknown boundary')
    before_backbone=backbone_velocity(model,optimizer); before_head=head_state(model,optimizer)
    before_weights={n:cpu_copy(p) for n,p in model.named_parameters() if not n.startswith('head_b.')}
    if any(v is None for v in before_backbone.values()): raise ValueError('All trained backbone velocity buffers must be present')
    if boundary=='enter_middle':
        if stash is not None or model.head_b is not None: raise ValueError('Middle transition already applied')
        if arm.temporary_head:
            stash=before_head
            model.freeze_head_a()
            for p in model.head_a.parameters(): p.grad=None
            # Head initialization does not consume model/data/probe RNG streams.
            with isolated_rng(stream_seed(seed,'temporary_head')):
                model.init_auxiliary_head(arm.middle_classes,device=next(model.parameters()).device)
            optimizer.add_param_group({**{k:copy.deepcopy(v) for k,v in optimizer.param_groups[0].items() if k!='params'},'params':list(model.head_b.parameters())})
    elif arm.temporary_head:
        if stash is None or model.head_b is None: raise ValueError('Missing temporary-head history')
        if state_hash(head_state(model,optimizer))!=state_hash(stash): raise ValueError('Frozen head A or its optimizer state changed during middle interval')
        temp_ids={id(p) for p in model.head_b.parameters()}
        for p in list(optimizer.state):
            if id(p) in temp_ids: del optimizer.state[p]
        optimizer.param_groups[:]=[g for g in optimizer.param_groups if not all(id(p) in temp_ids for p in g['params'])]
        model.discard_auxiliary_head(); model.unfreeze_head_a(); restore_head(model,optimizer,stash)
    if arm.reset_backbone: clear_backbone_momentum(model,optimizer)
    after_backbone=backbone_velocity(model,optimizer); after_head=head_state(model,optimizer)
    if arm.reset_backbone:
        if any(torch.count_nonzero(v).item() for v in after_backbone.values()): raise AssertionError('Backbone clearing failed')
    elif state_hash(before_backbone)!=state_hash(after_backbone): raise AssertionError('Carry policy erased backbone momentum')
    if state_hash(before_head)!=state_hash(after_head): raise AssertionError('Transition changed head-A weights or velocity')
    if state_hash(before_weights)!=state_hash({n:cpu_copy(p) for n,p in model.named_parameters() if not n.startswith('head_b.')}):
        raise AssertionError('Transition changed trained parameter weights')
    # Verify optimizer coverage after removing the temporary head.
    named_optimizer_state(model,optimizer)
    evidence=dict(arm=arm_name,boundary=boundary,policy='clear_both_boundaries' if arm.reset_backbone else 'carry_both_boundaries',
                  backbone_before_hash=state_hash(before_backbone),backbone_after_hash=state_hash(after_backbone),
                  head_before_hash=state_hash(before_head),head_after_hash=state_hash(after_head),
                  backbone_before=before_backbone,backbone_after=after_backbone,
                  head_before=before_head,head_after=after_head,
                  cleared_names=backbone_names(model) if arm.reset_backbone else [])
    return stash,evidence


def checkpoint(model,optimizer,seed,arm,global_update,phase,stash,identity,counters):
    return dict(format_version=1,seed=seed,arm=arm,global_update=global_update,phase=phase,
                model=cpu_copy(model.state_dict()),optimizer=named_optimizer_state(model,optimizer),stash=cpu_copy(stash),
                requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
                training=model.training,rng=capture_rng(),identity=identity,counters=copy.deepcopy(counters),
                schedule_position={'next_global_update':global_update+1},
                scheduler={'policy':'constant','lr':[g['lr'] for g in optimizer.param_groups]},scaler=None)


def load_checkpoint(saved,model_factory,device,expected_identity):
    if saved['format_version']!=1 or saved['identity']!=expected_identity: raise ValueError('Checkpoint/config/source/data identity mismatch')
    with isolated_rng(): model=model_factory().to(device)
    if any(n.startswith('head_b.') for n in saved['model']):
        with isolated_rng(): model.init_auxiliary_head(saved['model']['head_b.weight'].shape[0],device=device)
    model.load_state_dict(saved['model'],strict=True)
    for n,p in model.named_parameters(): p.requires_grad=saved['requires_grad'][n]
    model.train(saved['training']); optimizer=restore_named_optimizer(model,saved['optimizer'])
    restore_rng(saved['rng'])
    return model,optimizer,cpu_copy(saved['stash'])


def training_step(model,optimizer,images,labels,head):
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss=nn.functional.cross_entropy(model(images,head=head),labels)
    if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss; run invalid')
    loss.backward(); optimizer.step()
    return float(loss.detach())


def fit_linear_probe(model,train_loader,val_loader,device,epochs,lr,weight_decay):
    """Consume the declared probe recipe without changing historical evaluators."""
    model.eval()
    probe=nn.Linear(model.feat_dim,100).to(device)
    optimizer=torch.optim.Adam(probe.parameters(),lr=lr,weight_decay=weight_decay)
    for _ in range(epochs):
        probe.train()
        for x,y in train_loader:
            with torch.no_grad(): features=model(x.to(device),head='features')
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(probe(features),y.to(device))
            loss.backward(); optimizer.step()
    probe.eval(); correct=total=0
    with torch.no_grad():
        for x,y in val_loader:
            predicted=probe(model(x.to(device),head='features')).argmax(1)
            correct+=int((predicted==y.to(device)).sum()); total+=len(y)
    return 100.*correct/total


def recovery_summary(points,threshold):
    """Treat update-zero recovery and right censoring explicitly, never mean -1."""
    points=sorted(points)
    if not points or points[0][0]!=0: raise ValueError('Recovery must include the update-zero observation')
    hit=next((step for step,value in points if value>=threshold),None)
    return dict(recovery_status='already_recovered' if hit==0 else 'first_sampled_crossing' if hit is not None else 'not_recovered_by_horizon',
                recovery_update=hit,recovery_threshold=threshold,recovery_horizon=points[-1][0])
