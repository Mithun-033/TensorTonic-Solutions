# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Mithun Kannaa's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/mithunkannaa609.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Apply 4×4 Homogeneous Transform | Apply a 4x4 homogeneous transformation matrix to 3D points using rotation, translation, and homogeneous coordinates. | https://www.tensortonic.com/problems/homogeneous-transform |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Implement Nadam (Nesterov + Adam) | Implement one Nadam optimizer step in NumPy by combining Adam moments with Nesterov momentum. | https://www.tensortonic.com/problems/nadam-optimizer |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Fused Matmul + Bias + ReLU | Fuse tiled matrix multiplication, per-column bias, and ReLU in one Triton kernel with tail-safe memory access. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-fused-matmul-bias-relu |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-fused-multiply-add |
| Fused Row-Wise Softmax | Implement fused row-wise softmax in Triton with stable max subtraction, register reductions, and masked column tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-fused-softmax |
| GELU | Implement exact GELU activation in Triton with device error-function math and masked contiguous tiles. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-gelu |
| GEMV: Matrix Vector Product | Implement Triton matrix-vector multiplication with row-block programs, float32 accumulation, and masked matrix tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-gemv |
| L2 Vector Norm | Compute a Triton L2 vector norm with tiled sum-of-squares reduction, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-l2-norm |
| Row-Wise LogSumExp | Implement numerically stable row-wise LogSumExp in Triton with max subtraction and masked register reductions. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-logsumexp |
| Tiled Matrix Multiplication | Implement tiled matrix multiplication in Triton with a two-dimensional grid, float32 accumulation, and tail masks. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-matmul |
| Autotuned Matrix Multiplication | Autotune Triton matrix multiplication across tile and pipeline configurations while preserving masked boundary handling. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-matmul-autotune |
| Vector Max Reduction | Compute a vector maximum with one Triton reduction program and masked tail lanes that cannot win comparisons. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-max |
| Single-Pass Mean and Variance | Compute population mean and variance in Triton with single-pass statistics, atomic partials, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-mean-variance |
| ReLU | Implement ReLU activation in Triton with contiguous program tiles, branch-free rectification, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-relu |
| RMSNorm Forward | Implement RMSNorm forward in Triton with per-row square reduction, numerical stability, scaling, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-rmsnorm |
| SiLU | Implement fused SiLU or Swish activation in Triton with contiguous tiles, sigmoid weighting, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-silu |
| Vector Sum Reduction | Implement tiled vector sum reduction in Triton with register partials, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-sum |
| Tiled Transpose | Implement tiled matrix transpose in Triton by swapping load and store strides with masked boundary tiles. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-transpose |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-vector-addition |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/mithunkannaa609)
<!-- tensortonic:end -->
