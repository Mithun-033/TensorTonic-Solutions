import torch

def forward(x, y, weights, biases):
    hidden_acts = [x]
    curr = x
    for weight, bias in zip(weights, biases):
        curr = torch.tanh(curr @ weight.transpose(0, 1) + bias)
        hidden_acts.append(curr)
    pred = curr[:, 0]
    loss = ((pred - y) ** 2).sum()
    return pred, loss, hidden_acts

def backward(pred, targets, acts, weights):
    upstream = (2 * (pred - targets)).unsqueeze(1)
    weight_grad = [None] * len(weights)
    bias_grad = [None] * len(weights)
    for layer_idx in range(len(weights) - 1, -1, -1):
        out = acts[layer_idx + 1]
        delta = upstream * (1 - out * out)
        weight_grad[layer_idx] = delta.transpose(0, 1) @ acts[layer_idx]
        bias_grad[layer_idx] = delta.sum(dim=0)
        upstream = delta @ weights[layer_idx]
    return weight_grad, bias_grad


def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    losses = []
    trained_weights = [w.clone() for w in weights]
    trained_biases = [b.clone() for b in biases]
    learning_rate = float(learning_rate)

    for _ in range(steps):
        pred, loss, acts = forward(inputs, targets, trained_weights, trained_biases)
        losses.append(loss.clone())
        weight_grad, bias_grad = backward(pred, targets, acts, trained_weights)
        trained_weights = [w - learning_rate * grad for w, grad in zip(trained_weights, weight_grad)]
        trained_biases = [b - learning_rate * grad for b, grad in zip(trained_biases, bias_grad)]

    final_predictions, final_loss, _ = forward(inputs, targets, trained_weights, trained_biases)
    return (final_predictions, final_loss, trained_weights, trained_biases, losses)