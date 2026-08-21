import numpy as np

def apply_homogeneous_transform(T:list,points:list)->np.ndarray:
    T=np.asarray(T,dtype=float)
    points=np.asarray(points,dtype=float)

    single=points.ndim==1

    if single:
        points=points[None,:]

    points=np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
    points=(T@points.T).T[:,:-1]

    return points[0] if single else points