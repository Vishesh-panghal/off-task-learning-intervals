"""Measured runtime/resource events; no accuracy-based continuation rules."""
from pathlib import Path
import datetime as dt
import json
import time


def gpu_memory(device):
    import torch
    if str(device).startswith('cuda'):
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return dict(peak_allocated_bytes=None,peak_reserved_bytes=None)


def event(folder,kind,**values):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    with (folder/'runtime_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=kind,utc=dt.datetime.now(dt.timezone.utc).isoformat(),**values),allow_nan=False)+'\n')
        f.flush()


def disk_bytes(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())
