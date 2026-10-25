"""Prepare disabled S2 labels from the actual STL train_y.bin; no training."""
from pathlib import Path
import argparse
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.training.factorial_config import file_hash
from src.data.label_controls import shuffle_instance_labels


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels-bin',type=Path,required=True)
    p.add_argument('--images-bin',type=Path,required=True,help='Actual matching STL10 train_X.bin; its content is hashed here')
    p.add_argument('--seed',type=int,default=271830)
    p.add_argument('--output-dir',type=Path,default=ROOT/'protocol/labels/STL10-S2-v1')
    args=p.parse_args()
    labels=np.frombuffer(args.labels_bin.read_bytes(),dtype=np.uint8).astype(np.int64)-1
    if len(labels)!=5000 or labels.min()!=0 or labels.max()!=9: raise ValueError('Expected official 5000 one-based STL training labels')
    if args.images_bin.stat().st_size!=5000*3*96*96: raise ValueError('Expected official 5000-image STL training binary')
    images_sha256=file_hash(args.images_bin)
    shuffled,permutation,report=shuffle_instance_labels(labels,args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'fixed_labels.csv').open('x',newline='') as f:
        w=csv.writer(f); w.writerow(['stl10_train_image_id','true_label_0based','fixed_shuffled_label_0based','label_source_image_id'])
        w.writerows(zip(range(5000),labels.tolist(),shuffled.tolist(),permutation.tolist()))
    report.update(source_label_file_sha256=file_hash(args.labels_bin),source_train_images_sha256=images_sha256,
                  source_train_images_bytes=args.images_bin.stat().st_size,
                  prepared_mapping_sha256=file_hash(args.output_dir/'fixed_labels.csv'),study_stage='conditional_preparation_only',execution_enabled=False,
                  reuse_rule='Use fresh paired S1/S2 seeds under a separate locked H2 family; primary-study S1 is not automatically reused.')
    (args.output_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cross_tabulation'},indent=2))


if __name__=='__main__': main()
