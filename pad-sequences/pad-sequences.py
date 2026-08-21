import numpy as np

def pad_sequences(seqs:list,pad_value:int=0,max_len:int|None=None)->np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.array([], dtype=int).reshape(0, 0)
    _max=max_len if max_len is not None else max(len(seq) for seq in seqs)

    arr=np.full((len(seqs),_max),pad_value,dtype=int)

    for i in range(len(seqs)):
        seq=seqs[i]
        seq=seq[:max_len]
        arr[i,:len(seq)]=seq

    return arr