"""Indexed batches whose order and augmentation depend on global update, not arm.

No workers/prefetch state is hidden from checkpoints. Rebuilding a stream at a
saved global update recreates the same batch, including after an auxiliary gap.
All dataset construction is download=False; test data is constructed separately.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.data.datasets import CIFAR100_MEAN,CIFAR100_STD,STL10_MEAN,STL10_STD
from src.training.factorial_config import file_hash,stream_seed


def target_split(size,validation_size,split_seed):
    if not 0<validation_size<size: raise ValueError('Invalid validation size')
    order=np.random.RandomState(split_seed).permutation(size)
    val=order[:validation_size].astype(np.int64); train=order[validation_size:].astype(np.int64)
    if set(train)&set(val) or len(set(train)|set(val))!=size: raise AssertionError('Split overlap or incomplete coverage')
    return train,val


def indices_hash(values):
    return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()


class BatchSchedule:
    def __init__(self,raw,indices,batch_size,seed,source='target',namespace=None):
        self.raw=raw; self.indices=np.asarray(indices,dtype=np.int64)
        self.batch_size=batch_size; self.seed=seed; self.source=source
        self.namespace=namespace or source
        self.batches_per_epoch=len(self.indices)//batch_size
        if self.batches_per_epoch<1 or len(set(self.indices))!=len(self.indices): raise ValueError('Insufficient/duplicate training indices')
        self.epoch=None; self.order=None

    def batch(self,update):
        if type(update) is not int or update<1: raise ValueError('Positive global update required')
        epoch,offset=divmod(update-1,self.batches_per_epoch)
        if self.epoch!=epoch:
            self.order=np.random.RandomState(stream_seed(self.seed,self.namespace+'-sampler',epoch)).permutation(self.indices)
            self.epoch=epoch
        ids=self.order[offset*self.batch_size:(offset+1)*self.batch_size]
        gen=torch.Generator().manual_seed(stream_seed(self.seed,self.namespace+'-augmentation',update))
        crops=torch.randint(0,9,(self.batch_size,2),generator=gen).tolist()
        flips=(torch.rand(self.batch_size,generator=gen)<.5).tolist()
        images=[]; labels=[]
        mean,std=(CIFAR100_MEAN,CIFAR100_STD) if self.source=='target' else (STL10_MEAN,STL10_STD)
        for index,(top,left),flip in zip(ids,crops,flips):
            image,label=self.raw[int(index)]
            if self.source=='stl10': image=TF.resize(image,[32,32],InterpolationMode.BILINEAR,antialias=True)
            image=TF.crop(TF.pad(image,4),top,left,32,32)
            if flip: image=TF.hflip(image)
            images.append(TF.normalize(TF.to_tensor(image),mean,std)); labels.append(int(label))
        key=dict(source=self.source,namespace=self.namespace,update=update,epoch=epoch,ids=ids.tolist(),crops=crops,flips=flips)
        key['batch_hash']=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
        return torch.stack(images),torch.tensor(labels,dtype=torch.long),key


class EvaluationSubset(Dataset):
    def __init__(self,raw,indices): self.raw=raw; self.indices=list(map(int,indices))
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        image,label=self.raw[self.indices[i]]
        return TF.normalize(TF.to_tensor(image),CIFAR100_MEAN,CIFAR100_STD),int(label)


def evaluation_loader(raw,indices,batch_size,seed):
    return DataLoader(EvaluationSubset(raw,indices),batch_size=batch_size,shuffle=False,num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


class ProbeTrainingLoader:
    def __init__(self,schedule): self.schedule=schedule; self.epoch=0
    def __iter__(self):
        epoch=self.epoch; self.epoch+=1
        for offset in range(self.schedule.batches_per_epoch):
            x,y,_=self.schedule.batch(epoch*self.schedule.batches_per_epoch+offset+1)
            yield x,y


def scientific_data(root,config):
    """Read only public training partitions. Never construct official test here."""
    root=Path(root)
    target=torchvision.datasets.CIFAR100(root=str(root),train=True,download=False,transform=None)
    auxiliary=torchvision.datasets.STL10(root=str(root),split='train',download=False,transform=None)
    if len(target)!=50000 or len(auxiliary)!=5000: raise ValueError('Unexpected dataset cardinality')
    train,val=target_split(len(target),config['data']['validation_size'],config['data']['split_seed'])
    geometry=np.random.RandomState(config['data']['geometry_seed']).choice(val,size=config['data']['geometry_count'],replace=False)
    files=['cifar-100-python/train','cifar-100-python/meta','stl10_binary/train_X.bin','stl10_binary/train_y.bin']
    hashes={name:file_hash(root/name) for name in files}
    return dict(target=target,auxiliary=auxiliary,train=train,validation=val,geometry=geometry,
                manifest=dict(dataset='CIFAR100-45k-train+STL10-5k-train',file_sha256=hashes,
                              train_indices=train.tolist(),validation_indices=val.tolist(),geometry_indices=geometry.tolist(),
                              train_hash=indices_hash(train),validation_hash=indices_hash(val),geometry_hash=indices_hash(geometry),
                              split_seed=config['data']['split_seed'],geometry_seed=config['data']['geometry_seed'],
                              test_access='not constructed; geometry uses validation inputs',
                              stream_recipe='global-update-indexed RandomState epoch permutations and private torch crop/flip generators; drop-last each epoch'))


def official_test_data(root):
    return torchvision.datasets.CIFAR100(root=str(root),train=False,download=False,transform=None)
