import numpy as np

def cross_entropy_loss(y_true:list[int],y_pred:list[list[float]])->float:
    y_pred=np.asarray(y_pred,np.float32)
    loss=0

    for i in range(len(y_pred)):
        loss+=-np.log(y_pred[i][y_true[i]])

    return (loss/len(y_pred)).item()