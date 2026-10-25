"""Engineering-only actual-data CUDA timing/resume check; no efficacy outcome.

This is NOT the two full 15k/5k/15k seed pilots. It estimates component costs and
checks actual CUDA behavior before pilot release. It never opens official test.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def main():
    import torch
    from src.data.paired_batches import BatchSchedule,scientific_data
    from src.models.resnet_gn import ResNet18GroupNorm
    from src.training.factorial import checkpoint,isolated_rng,load_checkpoint,named_optimizer_state,state_hash,training_step,transition
    from src.training.factorial_config import file_hash,read_config
    from scripts.run_factorial import configure_torch,environment_identity,measure,save_state,source_manifest,verified_state
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/factorial_p2br_preliminary.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/cuda_preflight_v0.2')
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--full-diagnostic',action='store_true')
    args=p.parse_args()
    if args.steps<6: p.error('--steps must be at least 6 (five warmup updates plus a measured update)')
    if not args.output_dir.resolve().is_relative_to((ROOT/'runs').resolve()): raise ValueError('Use an isolated runs/ directory')
    args.output_dir.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    config=read_config(args.config); configure_torch('cuda'); data=scientific_data(args.data_dir,config)
    (args.output_dir/'data_manifest.json').write_text(json.dumps(data['manifest'],indent=2)+'\n')
    with isolated_rng(810001): model=ResNet18GroupNorm().cuda()
    optimizer=torch.optim.SGD(model.parameters(),**config['optimizer'])
    stream=BatchSchedule(data['target'],data['train'],config['data']['batch_size'],810001)
    torch.cuda.reset_peak_memory_stats(); measurements=[]
    for step in range(1,args.steps+1):
        tick=time.monotonic(); x,y,meta=stream.batch(step); batch_seconds=time.monotonic()-tick
        tick=time.monotonic(); training_step(model,optimizer,x.cuda(),y.cuda(),'A'); torch.cuda.synchronize()
        measurements.append(dict(step=step,batch_seconds=batch_seconds,update_seconds=time.monotonic()-tick,batch_hash=meta['batch_hash']))
    identity={'preflight':True,'source':source_manifest()}
    saved=checkpoint(model,optimizer,810001,'C3',args.steps,'middle',None,identity,{'total_updates':args.steps})
    path=args.output_dir/'resume_source.pt'; tick=time.monotonic(); save_state(path,saved)
    checkpoint_seconds=time.monotonic()-tick
    x,y,_=stream.batch(args.steps+1); training_step(model,optimizer,x.cuda(),y.cuda(),'A')
    expected=state_hash({'model':model.state_dict(),'optimizer':named_optimizer_state(model,optimizer)})
    del optimizer,model
    model,optimizer,_=load_checkpoint(verified_state(path),ResNet18GroupNorm,'cuda',identity)
    training_step(model,optimizer,x.cuda(),y.cuda(),'A')
    actual=state_hash({'model':model.state_dict(),'optimizer':named_optimizer_state(model,optimizer)})
    if actual!=expected: raise AssertionError('Actual CUDA resumed update differs from uninterrupted update')
    stash,enter=transition(model,optimizer,'C3','enter_middle',810001)
    auxiliary=BatchSchedule(data['auxiliary'],range(len(data['auxiliary'])),config['data']['batch_size'],810001,'stl10')
    # C3 temporary head is 100-class; S1 is separately 10-class. Measure one
    # actual auxiliary update with a ten-class head and the same clear policy.
    transition(model,optimizer,'C3','return_target',810001,stash)
    stash,_=transition(model,optimizer,'S1','enter_middle',810001)
    a,b,meta=auxiliary.batch(1); tick=time.monotonic(); training_step(model,optimizer,a.cuda(),b.cuda(),'B'); torch.cuda.synchronize()
    auxiliary_update_seconds=time.monotonic()-tick
    transition(model,optimizer,'S1','return_target',810001,stash)
    diagnostic_seconds=None
    if args.full_diagnostic:
        tick=time.monotonic(); observation=measure(model,data,config,810001,'preflight','cuda'); torch.cuda.synchronize()
        diagnostic_seconds=time.monotonic()-tick; del observation
    # Deliberately do not emit accuracy, losses, probe accuracy or efficacy tests.
    report=dict(status='cuda_engineering_preflight_passed',full_scientific_pilot_completed=False,seed=810001,
                environment=environment_identity('cuda'),source_sha256=source_manifest(),script_sha256=file_hash(Path(__file__)),
                training_dataset_hashes=data['manifest']['file_sha256'],measurements=measurements,
                warmup_steps_excluded_from_timing_summary=5,
                seconds_per_target_update_including_batch=sum(r['batch_seconds']+r['update_seconds'] for r in measurements[5:])/len(measurements[5:]),
                auxiliary_update_seconds_excluding_batch=auxiliary_update_seconds,checkpoint_seconds=checkpoint_seconds,checkpoint_bytes=path.stat().st_size,
                full_diagnostic_seconds=diagnostic_seconds,wall_seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cuda_resume_bitwise_equal=True,test_data_accessed=False,
                limitation='Short component measurements on the actual training data. No measured full-pilot runtime or reliable new-study variance is inferred.')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='measurements'},indent=2),flush=True)


if __name__=='__main__': main()
