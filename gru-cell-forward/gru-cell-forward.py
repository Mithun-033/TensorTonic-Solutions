import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    x, h_prev = np.asarray(x, dtype = np.float32), np.asarray(h_prev, dtype = np.float32)
    parameters = {name: np.asarray(value, dtype=float) for name, value in params.items()}
    flag = False
    if x.ndim == 1:
        x = np.expand_dims(x, axis = 0)
        h_prev = np.expand_dims(h_prev, axis = 0)
        flag = True

    def sigmoid(input):
        return np.where(input >= 0, 1.0 / (1.0 + np.exp(-input)), np.exp(input) / (1.0 + np.exp(input)))

    
    z = sigmoid(x @ parameters["Wz"] + h_prev @ parameters["Uz"] + parameters["bz"])
    r = sigmoid(x @ parameters["Wr"] + h_prev @ parameters["Ur"] + parameters["br"])

    candidate = np.tanh(
        x @ parameters["Wh"] + (r * h_prev) @ parameters["Uh"] + parameters["bh"]
    )
    hidden = (1.0 - z) * h_prev + z * candidate
    return hidden.squeeze(0) if flag else hidden
    