"""Consume the explicit focused three-arm P2BR configuration; default is read-only description, never training.

Synthetic checks use generated data and a tiny network, with no scientific
outcomes. Pilot/confirmatory training require the recovery plan's release gate.
Test images/labels are opened only by the separate, completeness-gated evaluate
mode after all locked confirmatory seeds and all three final models are present.
"""
from pathlib import Path
import argparse
import copy
import csv
import datetime as dt
import fcntl
import json
import os
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.training.focused_config import ARMS,digest_json,exposure_counts,file_hash,read_config,require_scientific_release,stream_seed


def json_text(value): return json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'


def atomic_json(path,value,immutable=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); text=json_text(value)
    if immutable and path.exists():
        if path.read_text()!=text: raise ValueError(f'Existing artifact differs: {path}')
        return
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp'); tmp.write_text(text)
    try:
        if immutable: os.link(tmp,path)
        else: os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def source_manifest():
    paths=['scripts/run_focused.py','scripts/analyze_focused.py','scripts/launch_focused_job.py','src/training/focused_config.py','src/training/factorial.py','src/training/factorial_config.py','src/training/runtime.py','src/data/paired_batches.py',
           'src/data/datasets.py','src/models/resnet_gn.py','src/training/evaluator.py','src/utils/metrics.py']
    return {p:file_hash(ROOT/p) for p in paths}


def configure_torch(device):
    import torch
    if device not in ['cpu','cuda']: raise ValueError('Only CPU and CUDA are supported by this protocol')
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    if device=='cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in [':4096:8',':16:8']:
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting CUDA execution')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False


def environment_identity(device):
    import torch
    import torchvision
    import numpy as np
    return dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),numpy=np.__version__,python=platform.python_version(),
                platform=platform.platform(),device=device,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_num_threads=torch.get_num_threads(),
                cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def synthetic_config(config):
    c=copy.deepcopy(config); c['study_id']+='-SYNTHETIC-CHECK'
    c['schedule']={'initial':3,'middle':2,'final':3,'target_comparator':6}
    c['data'].update(validation_size=16,batch_size=4,geometry_count=8)
    c['diagnostics'].update(validation_interval=2,probe_epochs=1)
    return c


def synthetic_data(config):
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset
    from src.data.paired_batches import target_split,indices_hash
    class Images(Dataset):
        def __init__(self,n,classes,seed):
            rng=np.random.RandomState(seed); self.images=rng.randint(0,256,(n,32,32,3),dtype=np.uint8); self.labels=np.arange(n)%classes
        def __len__(self): return len(self.labels)
        def __getitem__(self,index): return Image.fromarray(self.images[index]),int(self.labels[index])
    target=Images(48,100,221); auxiliary=Images(20,10,222)
    train,val=target_split(len(target),config['data']['validation_size'],config['data']['split_seed'])
    geometry=val[:config['data']['geometry_count']]
    return dict(target=target,auxiliary=auxiliary,train=train,validation=val,geometry=geometry,
                manifest=dict(dataset='GENERATED-SYNTHETIC-NONSCIENTIFIC',train_indices=train.tolist(),validation_indices=val.tolist(),geometry_indices=geometry.tolist(),
                              train_hash=indices_hash(train),validation_hash=indices_hash(val),geometry_hash=indices_hash(geometry)))


def tiny_model():
    import torch
    from torch import nn
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.feat_dim=8
            self.backbone=nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,8),nn.Tanh()); self.head_a=nn.Linear(8,100); self.head_b=None
        def forward(self,x,head='A'):
            features=self.backbone(x)
            return features if head=='features' else (self.head_a if head=='A' else self.head_b)(features)
        def freeze_head_a(self):
            for p in self.head_a.parameters(): p.requires_grad=False
        def unfreeze_head_a(self):
            for p in self.head_a.parameters(): p.requires_grad=True
        def init_auxiliary_head(self,num_classes_b,device='cpu'): self.head_b=nn.Linear(8,num_classes_b).to(device)
        def discard_auxiliary_head(self): self.head_b=None
    return Tiny()


def measure(model,data,config,seed,phase,device):
    import numpy as np
    from src.training.factorial import fit_linear_probe,isolated_rng,state_hash
    from src.data.paired_batches import BatchSchedule,ProbeTrainingLoader,evaluation_loader
    from src.utils.metrics import evaluate_accuracy,extract_probe_features,compute_effective_rank
    # Copies isolate mode, gradients, weights and probe fitting from training.
    before=state_hash(model.state_dict())
    with isolated_rng(stream_seed(seed,'diagnostic-'+phase)):
        clone=copy.deepcopy(model).to(device)
        val=evaluation_loader(data['target'],data['validation'],config['data']['batch_size'],stream_seed(seed,'validation-'+phase))
        accuracy=evaluate_accuracy(clone,val,device,head='A')
        probe_loader=ProbeTrainingLoader(BatchSchedule(data['target'],data['train'],config['data']['batch_size'],seed,namespace='probe-'+phase))
        d=config['diagnostics']
        probe=fit_linear_probe(clone,probe_loader,val,device,epochs=d['probe_epochs'],lr=d['probe_lr'],weight_decay=d['probe_weight_decay'])
        geometry=evaluation_loader(data['target'],data['geometry'],config['data']['batch_size'],stream_seed(seed,'geometry-'+phase))
        features=extract_probe_features(clone,geometry,device)
        rank=compute_effective_rank(features)
        result=dict(validation_accuracy=accuracy,probe_validation_accuracy=probe,effective_rank=rank,features=features,
                    feature_ids_sha256=data['manifest']['geometry_hash'],phase=phase,geometry_split='validation_inputs')
        del clone
    if state_hash(model.state_dict())!=before: raise AssertionError('Diagnostic changed training weights')
    return result


def validation(model,data,config,seed,device):
    from src.training.factorial import isolated_rng
    from src.data.paired_batches import evaluation_loader
    from src.utils.metrics import evaluate_accuracy
    modes={n:m.training for n,m in model.named_modules()}
    try:
        with isolated_rng():
            loader=evaluation_loader(data['target'],data['validation'],config['data']['batch_size'],stream_seed(seed,'validation'))
            return evaluate_accuracy(model,loader,device,head='A')
    finally:
        for n,m in model.named_modules(): m.training=modes[n]


def save_state(path,state):
    import torch
    from src.training.factorial import state_hash
    from src.training.runtime import event
    started=time.monotonic(); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        prior=read_state(path)
        # Wall-clock measurements differ after retry; every scientific field,
        # RNG state and tensor must agree. Keep the original published file.
        timing={'continuation_seconds','prefix_execution_seconds_including_diagnostics','runtime_profile'}
        if state_hash({k:v for k,v in prior.items() if k not in timing})!=state_hash({k:v for k,v in state.items() if k not in timing}):
            raise FileExistsError(f'Existing checkpoint is not an identical scientific state: {path}')
        atomic_json(path.parent/(path.name+'.recovery.json'),{'checkpoint':path.name,'reason':'idempotent recovery of published state; original measurements retained','sha256':file_hash(path)})
    else:
        tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        try:
            torch.save(state,tmp); os.link(tmp,path)
        finally: tmp.unlink(missing_ok=True)
    entry={'checkpoint':path.name,'sha256':file_hash(path)}
    atomic_json(path.parent/(path.name+'.sha256.json'),entry,immutable=True)
    atomic_json(path.parent/'latest.json',entry)
    event(path.parent,'checkpoint_io',checkpoint=path.name,seconds=time.monotonic()-started,bytes=path.stat().st_size)


def read_state(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)


def latest_state(folder):
    entry=json.loads((folder/'latest.json').read_text()); path=folder/entry['checkpoint']
    if file_hash(path)!=entry['sha256']: raise ValueError('Checkpoint checksum mismatch')
    return read_state(path)


def verified_state(path):
    path=Path(path); sidecar=path.parent/(path.name+'.sha256.json')
    entry=json.loads((sidecar if sidecar.exists() else path.parent/'latest.json').read_text())
    if entry['checkpoint']!=path.name or file_hash(path)!=entry['sha256']:
        raise ValueError('Named checkpoint checksum mismatch or missing checksum evidence')
    return read_state(path)


def counters_at_prefix(n):
    return dict(total_updates=n,backbone_target_updates=n,backbone_auxiliary_updates=0,original_head_updates=n,temporary_head_updates=0)


def prepare_trace(folder,update,resume):
    path=folder/'training_trace.csv'
    if resume and path.exists():
        with path.open(newline='') as f: rows=list(csv.DictReader(f))
        archive=folder/'attempts'; archive.mkdir(exist_ok=True)
        number=len(list(archive.glob('training_trace_*.csv')))
        path.rename(archive/f'training_trace_{number}.csv')
        # Preserve the complete interrupted attempt; canonical trace rolls back
        # to its checkpoint, rather than silently counting replayed updates twice.
        kept=[r for r in rows if int(r['global_update'])<=update]
        if rows:
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(kept)
    return path


def validate_training_trace(path,config,arm=None):
    with Path(path).open(newline='') as f: rows=list(csv.DictReader(f))
    s=config['schedule']; middle=s['initial']+s['middle']; total=middle+s['final']
    expected=range(1,s['initial']+1) if arm is None else range(s['initial']+1,total+1)
    if [int(r['global_update']) for r in rows]!=list(expected): raise ValueError('Incomplete/duplicate training update trace')
    for row in rows:
        step=int(row['global_update'])
        source=ARMS[arm].middle_source if arm and step<=middle else 'target'
        head='B' if arm and step<=middle and ARMS[arm].temporary_head else 'A'
        if row['source']!=source or row['head']!=head or float(row['lr'])!=config['optimizer']['lr'] or len(row['batch_hash'])!=64:
            raise ValueError('Training trace violates source/head/LR schedule')
    return file_hash(path)


def run_seed(config,data,seed,folder,device='cpu',resume=False,synthetic=False,stop_after=None,arm_order=None):
    import torch
    from src.data.paired_batches import BatchSchedule
    from src.models.resnet_gn import ResNet18GroupNorm
    from src.training.factorial import capture_rng,checkpoint,isolated_rng,load_checkpoint,named_optimizer_state,recovery_summary,state_hash,training_step,transition
    from src.training.runtime import disk_bytes,event,gpu_memory
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    seed_started=time.monotonic()
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    event(folder,'seed_execution_start',seed=seed,device=device,resume=resume,synthetic=synthetic)
    factory=tiny_model if synthetic else ResNet18GroupNorm
    identity=dict(config_sha256=digest_json(config),protocol_sha256=file_hash(ROOT/config['protocol_path']),source_sha256=source_manifest(),
                  data_manifest_sha256=digest_json(data['manifest']),seed=seed,device=device,synthetic=synthetic,
                  environment=environment_identity(device))
    atomic_json(folder/'identity.json',identity,immutable=True)
    atomic_json(folder/'consumed_config.json',config,immutable=True)
    atomic_json(folder/'split_manifest.json',data['manifest'],immutable=True)
    s=config['schedule']; end_middle=s['initial']+s['middle']; total=end_middle+s['final']
    target=BatchSchedule(data['target'],data['train'],config['data']['batch_size'],seed)
    def diagnostic(model,phase):
        started=time.monotonic(); result=measure(model,data,config,seed,phase,device)
        event(folder,'diagnostic',phase=phase,arm=current_arm,seconds=time.monotonic()-started,**gpu_memory(device))
        return result
    current_arm='shared_prefix'
    prefix=folder/'common'; prefix.mkdir(exist_ok=True)
    common=prefix/'theta1.pt'
    prefix_was_reused=common.exists()
    if common.exists():
        if not resume: raise FileExistsError('Existing seed requires explicit --resume')
        shared=verified_state(common)
        if shared['identity']!=identity: raise ValueError('Shared-prefix identity mismatch')
        if validate_training_trace(prefix/'training_trace.csv',config)!=shared['prefix_trace_sha256']:
            raise ValueError('Shared-prefix trace checksum mismatch')
    else:
        init=prefix/'theta0.pt'
        if init.exists():
            if not resume: raise FileExistsError('Existing prefix requires explicit --resume')
            initial=verified_state(init); model,optimizer,_=load_checkpoint(initial,factory,device,identity)
        else:
            with isolated_rng(stream_seed(seed,'model_initialization')): model=factory().to(device)
            import random
            import numpy as np
            forward_seed=stream_seed(seed,'training_forward')
            random.seed(forward_seed); np.random.seed(forward_seed); torch.manual_seed(forward_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(forward_seed)
            optimizer=torch.optim.SGD(model.parameters(),**config['optimizer'])
            initial=checkpoint(model,optimizer,seed,'shared',0,'prefix',None,identity,counters_at_prefix(0))
            save_state(init,initial)
        started=time.monotonic(); prefix_trace=[]
        for step in range(1,s['initial']+1):
            x,y,meta=target.batch(step); loss=training_step(model,optimizer,x.to(device),y.to(device),'A')
            prefix_trace.append(dict(global_update=step,source='target',head='A',lr=optimizer.param_groups[0]['lr'],batch_hash=meta['batch_hash'],loss=loss))
            if step%2500==0 or step==s['initial']: print(f'Seed {seed} shared prefix update {step}/{s["initial"]}',flush=True)
        shared=checkpoint(model,optimizer,seed,'shared',s['initial'],'before_middle',None,identity,counters_at_prefix(s['initial']))
        shared['diagnostics']={'theta1':diagnostic(model,'theta1')}
        shared['prefix_execution_seconds_including_diagnostics']=time.monotonic()-started
        shared['rng']=capture_rng()
        if (prefix/'training_trace.csv').exists():
            if not resume: raise FileExistsError('Unassociated prefix trace exists')
            prepare_trace(prefix,0,True)
            (prefix/'training_trace.csv').unlink(missing_ok=True)
        with (prefix/'training_trace.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(prefix_trace[0])); writer.writeheader(); writer.writerows(prefix_trace)
        shared['prefix_trace_sha256']=validate_training_trace(prefix/'training_trace.csv',config)
        save_state(common,shared)
    event(folder,'shared_prefix_ready',reused=prefix_was_reused,**gpu_memory(device))
    prefix_hash=state_hash({'model':shared['model'],'optimizer':shared['optimizer'],'rng':shared['rng']})
    outputs=[]
    order=arm_order or list(ARMS)
    if set(order)!=set(ARMS) or len(order)!=len(ARMS): raise ValueError('Exactly the three declared focused arms required')
    auxiliary=BatchSchedule(data['auxiliary'],range(len(data['auxiliary'])),config['data']['batch_size'],seed,'stl10')
    for arm_name in order:
        current_arm=arm_name
        arm=ARMS[arm_name]; arm_folder=folder/arm_name; arm_folder.mkdir(exist_ok=True)
        event(folder,'arm_start',arm=arm_name)
        print(f'Seed {seed}: {arm_name} continuation',flush=True)
        result_path=arm_folder/'outcome.json'
        if result_path.exists():
            if not resume: raise FileExistsError('Outcome already exists; explicit --resume required')
            result=json.loads(result_path.read_text())
            saved=latest_state(arm_folder)
            if saved['identity']!=identity or saved['phase']!='complete' or saved.get('shared_prefix_hash')!=prefix_hash or result.get('shared_prefix_hash')!=prefix_hash or result['final_checkpoint_sha256']!=file_hash(arm_folder/'theta3.pt') or result['training_trace_sha256']!=validate_training_trace(arm_folder/'training_trace.csv',config,arm_name):
                raise ValueError('Completed outcome/checkpoint identity mismatch')
            outputs.append(result); continue
        if (arm_folder/'latest.json').exists():
            if not resume: raise FileExistsError('Existing continuation requires explicit --resume')
            state=latest_state(arm_folder)
        else:
            state=copy.deepcopy(shared); state['arm']=arm_name; state['shared_prefix_hash']=prefix_hash
            save_state(arm_folder/'theta1_pre.pt',state)
        if state['arm']!=arm_name or state.get('shared_prefix_hash')!=prefix_hash: raise ValueError('Checkpoint arm/shared-prefix association mismatch')
        model,optimizer,stash=load_checkpoint(state,factory,device,identity)
        update=state['global_update']; phase=state['phase']; counts=copy.deepcopy(state['counters'])
        diagnostics=state.get('diagnostics',copy.deepcopy(shared['diagnostics'])); recovery=state.get('recovery_points',[])
        elapsed=state.get('continuation_seconds',0.); start=time.monotonic()
        trace_path=prepare_trace(arm_folder,update,resume)
        def save(name,new_phase):
            nonlocal phase
            phase=new_phase
            bundle=checkpoint(model,optimizer,seed,arm_name,update,phase,stash,identity,counts)
            bundle.update(diagnostics=diagnostics,recovery_points=recovery,shared_prefix_hash=prefix_hash,continuation_seconds=elapsed+time.monotonic()-start)
            save_state(arm_folder/name,bundle)
        def apply_boundary(boundary):
            nonlocal stash
            stash,evidence=transition(model,optimizer,arm_name,boundary,seed,stash)
            path=arm_folder/f'{boundary}_optimizer_snapshots.pt'
            # A crash between evidence and post-checkpoint may replay this
            # transition from its immutable pre-checkpoint; require exact match.
            if path.exists():
                if state_hash(read_state(path))!=state_hash(evidence): raise ValueError('Transition evidence differs on resume')
            else:
                temp=path.with_suffix('.tmp'); torch.save(evidence,temp)
                try: os.link(temp,path)
                finally: temp.unlink(missing_ok=True)
        if phase=='before_middle':
            apply_boundary('enter_middle'); save('theta1_post.pt','middle')
        with trace_path.open('a',newline='') as f:
            fields=['global_update','phase','source','head','lr','batch_hash','loss','total_updates','backbone_target_updates','backbone_auxiliary_updates','original_head_updates','temporary_head_updates']
            writer=csv.DictWriter(f,fieldnames=fields)
            if f.tell()==0: writer.writeheader()
            while update<total:
                if phase=='before_return':
                    apply_boundary('return_target'); save('theta2_post.pt','final')
                step=update+1
                in_middle=step<=end_middle
                source=arm.middle_source if in_middle else 'target'
                head='B' if in_middle and arm.temporary_head else 'A'
                schedule=target if source=='target' else auxiliary
                x,y,meta=schedule.batch(step if source=='target' else step-s['initial'])
                loss=training_step(model,optimizer,x.to(device),y.to(device),head); update=step
                counts['total_updates']+=1; counts['backbone_target_updates' if source=='target' else 'backbone_auxiliary_updates']+=1
                counts['original_head_updates' if head=='A' else 'temporary_head_updates']+=1
                writer.writerow(dict(global_update=update,phase=phase,source=source,head=head,lr=optimizer.param_groups[0]['lr'],batch_hash=meta['batch_hash'],loss=loss,**counts)); f.flush()
                if update>end_middle and (update-end_middle)%config['diagnostics']['validation_interval']==0:
                    recovery.append((update-end_middle,validation(model,data,config,seed,device)))
                if update%2500==0 or update==total: print(f'Seed {seed} {arm_name} update {update}/{total}',flush=True)
                if update==end_middle:
                    diagnostics['theta2']=diagnostic(model,'theta2')
                    recovery=[(0,diagnostics['theta2']['validation_accuracy'])]
                    save('theta2_pre.pt','before_return')
                if update==s['target_comparator']:
                    diagnostics['target_comparator']={'validation_accuracy':validation(model,data,config,seed,device)}
                    save('theta30k.pt',phase)
                if update==total:
                    diagnostics['theta3']=diagnostic(model,'theta3')
                    if not recovery or recovery[-1][0]!=s['final']: recovery.append((s['final'],diagnostics['theta3']['validation_accuracy']))
                    save('theta3.pt','complete')
                elif stop_after==update:
                    save(f'interrupted_at_{update}.pt',phase)
                    return {'interrupted':True,'arm':arm_name,'update':update}
        trace_hash=validate_training_trace(trace_path,config,arm_name)
        expected=exposure_counts(config,arm_name)
        if counts!={k:expected[k] for k in counts}: raise AssertionError('Training/head exposure accounting mismatch')
        clean={name:{k:v for k,v in values.items() if k!='features'} for name,values in diagnostics.items()}
        # Centered linear CKA in feature space; no Gram matrices or test labels.
        x=diagnostics['theta1']['features'].double(); y=diagnostics['theta2']['features'].double()
        denom=torch.linalg.norm(x.T@x)*torch.linalg.norm(y.T@y)
        cka=float(torch.square(torch.linalg.norm(x.T@y))/denom) if denom>0 else None
        result=dict(study_id=config['study_id'],seed=seed,arm=arm_name,synthetic=synthetic,scientific_result=not synthetic,
                    result_role='synthetic_check_only' if synthetic else 'engineering_pilot_only' if config['stage']=='pilot' else 'prospectively_specified_budget_limited_followup_without_test_access',
                    config_sha256=digest_json(config),protocol_sha256=identity['protocol_sha256'],data_manifest_sha256=identity['data_manifest_sha256'],
                    shared_prefix_hash=prefix_hash,training_trace_sha256=trace_hash,counts=counts,diagnostics=clean,theta2_linear_cka_with_theta1=cka,
                    recovery=recovery_summary(recovery,diagnostics['theta1']['validation_accuracy']),
                    final_checkpoint_sha256=file_hash(arm_folder/'theta3.pt'),target_comparator_checkpoint_sha256=file_hash(arm_folder/'theta30k.pt'),final_model_hash=state_hash(model.state_dict()),
                    final_optimizer_hash=state_hash(named_optimizer_state(model,optimizer)),
                    continuation_seconds=elapsed+time.monotonic()-start,runtime_profile=gpu_memory(device),primary_test_accuracy=None,
                    test_status='not accessed; separate locked completeness-gated evaluation required')
        atomic_json(result_path,result,immutable=True); outputs.append(result)
        event(folder,'arm_complete',arm=arm_name,continuation_seconds=result['continuation_seconds'],**gpu_memory(device))
    with (folder/'C0/training_trace.csv').open(newline='') as f:
        reference={int(r['global_update']):r['batch_hash'] for r in csv.DictReader(f)}
    for arm in ARMS:
        with (folder/arm/'training_trace.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                if row['source']=='target' and row['batch_hash']!=reference[int(row['global_update'])]:
                    raise AssertionError('Target minibatch/augmentation pairing failed')
    atomic_json(folder/'seed_complete.json',dict(seed=seed,arms=list(ARMS),outcome_sha256={a:file_hash(folder/a/'outcome.json') for a in ARMS},synthetic=synthetic),immutable=True)
    event(folder,'seed_execution_complete',wall_seconds=time.monotonic()-seed_started,storage_bytes=disk_bytes(folder),**gpu_memory(device))
    return outputs


def evaluate_complete(config,data_root,folder,device):
    require_scientific_release(config,ROOT,'confirmatory',config['confirmatory_seeds'])
    folder=Path(folder)
    if not folder.is_dir(): raise FileNotFoundError('Complete locked study directory is missing')
    with (folder/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _evaluate_complete_locked(config,data_root,folder,device)


def _evaluate_complete_locked(config,data_root,folder,device):
    """Evaluate only the complete locked study; append a durable access record first."""
    import torch
    from src.data.paired_batches import official_test_data,evaluation_loader
    from src.models.resnet_gn import ResNet18GroupNorm
    from src.training.factorial import isolated_rng,load_checkpoint
    from src.utils.metrics import evaluate_accuracy
    require_scientific_release(config,ROOT,'confirmatory',config['confirmatory_seeds'])
    candidates=[]
    for seed in config['confirmatory_seeds']:
        seed_folder=folder/f'seed_{seed}'
        complete=json.loads((seed_folder/'seed_complete.json').read_text())
        identity=json.loads((seed_folder/'identity.json').read_text())
        if complete['synthetic'] or identity['synthetic'] or complete['arms']!=list(ARMS) or set(complete['outcome_sha256'])!=set(ARMS) or identity['config_sha256']!=digest_json(config) or identity['source_sha256']!=source_manifest() or identity['protocol_sha256']!=file_hash(ROOT/config['protocol_path']) or identity['environment']!=environment_identity(device):
            raise ValueError('Study completeness/source/configuration identity failed')
        for arm in ARMS:
            path=seed_folder/arm/'outcome.json'
            if file_hash(path)!=complete['outcome_sha256'][arm]: raise ValueError('Outcome checksum mismatch')
            outcome=json.loads(path.read_text()); final=seed_folder/arm/'theta3.pt'
            if outcome['synthetic'] or not outcome['scientific_result'] or outcome['seed']!=seed or outcome['arm']!=arm or outcome['study_id']!=config['study_id'] or outcome['config_sha256']!=digest_json(config) or outcome['protocol_sha256']!=identity['protocol_sha256'] or outcome['data_manifest_sha256']!=identity['data_manifest_sha256'] or file_hash(final)!=outcome['final_checkpoint_sha256'] or outcome['training_trace_sha256']!=validate_training_trace(seed_folder/arm/'training_trace.csv',config,arm): raise ValueError('Invalid final evidence')
            candidates.append((seed,arm,final,identity,'official_cifar100_test_top1_at_35k'))
            if arm=='C0':
                comparator=seed_folder/arm/'theta30k.pt'
                if file_hash(comparator)!=outcome['target_comparator_checkpoint_sha256']: raise ValueError('30k comparator checksum mismatch')
                candidates.append((seed,arm,comparator,identity,'official_cifar100_test_top1_at_30k'))
    output=folder/'final_test_outcomes.json'
    if output.exists(): raise FileExistsError('Final test outcomes already exist; read existing release instead of reevaluating')
    log=folder/'test_access.jsonl'
    with log.open('a') as f:
        f.write(json.dumps(dict(event='begin_final_test_evaluation',utc=dt.datetime.now(dt.timezone.utc).isoformat(),config_sha256=digest_json(config),source_sha256=source_manifest(),models=len(candidates),test_dataset_sha256=file_hash(Path(data_root)/'cifar-100-python/test'),historical_test_exposed=True))+'\n'); f.flush(); os.fsync(f.fileno())
    test=official_test_data(data_root); rows=[]; comparator_rows=[]
    for seed,arm,path,identity,metric in candidates:
        cached=folder/'test_evaluations'/f'seed_{seed}_{arm}_{path.stem}.json'
        if cached.exists():
            row=json.loads(cached.read_text())
            if row['seed']!=seed or row['arm']!=arm or row['metric']!=metric or row['study_id']!=config['study_id'] or not isinstance(row['accuracy'],(int,float)) or not 0<=row['accuracy']<=100 or row['checkpoint_sha256']!=file_hash(path) or row['config_sha256']!=digest_json(config) or row['protocol_sha256']!=identity['protocol_sha256']:
                raise ValueError('Prior test evaluation identity mismatch')
        else:
            saved=read_state(path); model,_,_=load_checkpoint(saved,ResNet18GroupNorm,device,identity)
            with isolated_rng():
                loader=evaluation_loader(test,range(len(test)),config['data']['batch_size'],stream_seed(seed,'final_test'))
                accuracy=evaluate_accuracy(model,loader,device)
            row=dict(study_id=config['study_id'],seed=seed,arm=arm,metric=metric,accuracy=accuracy,
                     checkpoint_sha256=file_hash(path),config_sha256=digest_json(config),protocol_sha256=identity['protocol_sha256'])
            atomic_json(cached,row,immutable=True)
        (rows if metric.endswith('35k') else comparator_rows).append(row)
    atomic_json(output,dict(rows=rows,target_comparator_rows=comparator_rows,config_sha256=digest_json(config),historical_test_exposed=True),immutable=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/focused_p2br_13h_candidate_20261017.json')
    p.add_argument('--mode',choices=['describe','synthetic-check','confirmatory','evaluate'],default='describe')
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data_cache')
    p.add_argument('--output-root',type=Path,default=ROOT/'runs/p2br_focused_13h')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); config=read_config(args.config)
    if args.mode=='describe':
        print(json_text(dict(study_id=config['study_id'],stage=config['stage'],arms={a:exposure_counts(config,a) for a in ARMS},release=config['release'],status='Read-only; no datasets or models loaded'))); return
    seeds=args.seeds or ([190001] if args.mode=='synthetic-check' else config['pilot_seeds'] if args.mode=='pilot' else config['confirmatory_seeds'])
    if args.mode!='synthetic-check': require_scientific_release(config,ROOT,'confirmatory' if args.mode=='evaluate' else args.mode,seeds)
    if not args.output_root.resolve().is_relative_to((ROOT/'runs').resolve()): raise ValueError('New execution outputs must be under project runs/; historical directories are protected')
    configure_torch(args.device)
    if args.mode=='synthetic-check':
        config=synthetic_config(config); data=synthetic_data(config)
    elif args.mode!='evaluate':
        from src.data.paired_batches import scientific_data
        data=scientific_data(args.data_dir,config)
    folder=args.output_root/('confirmatory' if args.mode=='evaluate' else args.mode)/config['study_id']
    if args.mode=='evaluate':
        print(evaluate_complete(config,args.data_dir,folder,args.device)); return
    folder.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        with (folder/f'.seed_{seed}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=run_seed(config,data,seed,folder/f'seed_{seed}',args.device,args.resume,args.mode=='synthetic-check')
            print(json_text(dict(seed=seed,arms_completed=len(result),scientific_result=args.mode!='synthetic-check',folder=str(folder/f'seed_{seed}'))))


if __name__=='__main__': main()
