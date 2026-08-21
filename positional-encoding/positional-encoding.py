import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    T=np.arange(seq_len,dtype=np.int16)[:,None]
    freqs=1/base**(np.arange(0,d_model,2)/d_model)

    angles=T*freqs

    PE=np.zeros((seq_len,d_model),dtype=np.float32)
    PE[:,0::2]=np.sin(angles)
    PE[:,1::2]=np.cos(angles[:,:(d_model//2)])

    return PE