"""Fixed instance-level label randomization; exact histogram, original images."""
import hashlib
import numpy as np


def vector_hash(values):
    return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()


def shuffle_instance_labels(labels,seed):
    labels=np.asarray(labels,dtype=np.int64)
    if labels.ndim!=1 or len(labels)<2 or set(labels.tolist())!=set(range(10)):
        raise ValueError('Expected a nonempty ten-class label vector encoded 0..9')
    permutation=np.random.RandomState(seed).permutation(len(labels))
    shuffled=labels[permutation].copy()
    table=np.zeros((10,10),dtype=np.int64)
    np.add.at(table,(labels,shuffled),1)
    original_counts=np.bincount(labels,minlength=10); shuffled_counts=np.bincount(shuffled,minlength=10)
    if not np.array_equal(original_counts,shuffled_counts): raise AssertionError('Class histogram changed')
    # A global class rename has exactly one occupied destination per row.
    if np.all((table>0).sum(axis=1)==1): raise ValueError('Shuffle is a global class mapping; use a prespecified different seed')
    return shuffled,permutation,dict(seed=seed,n_images=len(labels),true_label_hash=vector_hash(labels),
        shuffled_label_hash=vector_hash(shuffled),permutation_hash=vector_hash(permutation),
        true_counts=original_counts.tolist(),shuffled_counts=shuffled_counts.tolist(),cross_tabulation=table.tolist(),
        changed_instances=int((labels!=shuffled).sum()),global_class_permutation=False,
        interpretation='Fixed per-image label assignments; preserves histogram; memorization remains possible. Not a semantic-transfer test by itself.')


class FixedLabels:
    """Return the same underlying image with its one immutable prepared label."""
    def __init__(self,raw,labels):
        self.raw=raw; self.labels=np.asarray(labels,dtype=np.int64).copy()
        if len(raw)!=len(self.labels): raise ValueError('Image/label count mismatch')
        self.labels.setflags(write=False)
    def __len__(self): return len(self.labels)
    def __getitem__(self,index):
        image,_=self.raw[index]
        return image,int(self.labels[index])
