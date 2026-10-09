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
  <img src="https://www.tensortonic.com/api/badge/mithun_kannaa.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Fused Tiled Matmul and ReLU | Fuse tiled matrix multiplication and ReLU in Triton for strided float16 or bfloat16 CUDA inputs and preallocated outputs. | https://www.tensortonic.com/problems/cs336-l06-triton-matmul-relu |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Build a Mini GRU Cell (Forward Pass) | Implement a GRU cell forward pass with reset, update, and candidate gates for one sequence timestep. | https://www.tensortonic.com/problems/gru-cell-forward |
| Apply 4×4 Homogeneous Transform | Apply a 4x4 homogeneous transformation matrix to 3D points using rotation, translation, and homogeneous coordinates. | https://www.tensortonic.com/problems/homogeneous-transform |
| Implement Huber Loss | Compute Huber loss with quadratic errors near zero and linear penalties beyond a configurable threshold. | https://www.tensortonic.com/problems/huber-loss |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete probability distributions with safe zero-probability handling. | https://www.tensortonic.com/problems/kl-divergence |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Learning Rate Scheduler (Linear Decay) | Compute a linearly decaying learning rate across training steps between configured start and end values. | https://www.tensortonic.com/problems/linear-lr-scheduler |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Implement Nadam (Nesterov + Adam) | Implement one Nadam optimizer step in NumPy by combining Adam moments with Nesterov momentum. | https://www.tensortonic.com/problems/nadam-optimizer |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Basic SELECT | Write a SQL SELECT query that aliases product names and calculates inventory value from unit price and stock quantity. | https://www.tensortonic.com/problems/sql-basic-select |
| Conditional Aggregation | Summarize support tickets by department with conditional SQL counts for open, in-progress, and closed statuses. | https://www.tensortonic.com/problems/sql-conditional-aggregation |
| COUNT, SUM, AVG | Aggregate sales by category with SQL COUNT, SUM, and AVG while handling NULL discounts and deterministic ordering. | https://www.tensortonic.com/problems/sql-count-sum-avg |
| Cross Join | Generate every segment and metric combination with a SQL CROSS JOIN for a complete reporting grid. | https://www.tensortonic.com/problems/sql-cross-join |
| DISTINCT Values | Return each customer and their distinct product count with SQL aggregation and deterministic sorting. | https://www.tensortonic.com/problems/sql-distinct-values |
| GROUP BY | Group orders by customer in SQL to calculate total order count and spending, ordered by highest spend. | https://www.tensortonic.com/problems/sql-group-by |
| HAVING Clause | Use SQL GROUP BY and HAVING to find customers with at least two orders and summarize their total spending. | https://www.tensortonic.com/problems/sql-having-clause |
| INNER JOIN | Join employees to matching departments with SQL INNER JOIN and return employee name, salary, and department. | https://www.tensortonic.com/problems/sql-inner-join |
| LEFT JOIN | Use SQL LEFT JOIN to include every customer and calculate total spending, returning zero for customers without orders. | https://www.tensortonic.com/problems/sql-left-join |
| LIMIT and OFFSET | Use SQL ORDER BY, LIMIT, and OFFSET to return the second through fourth highest-revenue sales with tie-breaking. | https://www.tensortonic.com/problems/sql-limit-offset |
| Multiple Joins | Join users, experiment assignments, and conversion events in SQL to report converted users, variants, and revenue. | https://www.tensortonic.com/problems/sql-multiple-joins |
| Nested Aggregations | Use a SQL subquery or CTE to compute daily order totals, average daily revenue, and the busiest day. | https://www.tensortonic.com/problems/sql-nested-aggregations |
| ORDER BY | Sort student exam results in SQL by descending score and ascending name for deterministic ties. | https://www.tensortonic.com/problems/sql-order-by |
| RANK and DENSE_RANK | Rank ML models within each dataset using SQL RANK and DENSE_RANK over descending accuracy. | https://www.tensortonic.com/problems/sql-rank-dense-rank |
| ROW_NUMBER | Assign deterministic per-segment activity ranks with SQL ROW_NUMBER ordered by engagement score and username. | https://www.tensortonic.com/problems/sql-row-number |
| Self Join | Use a SQL self join to pair users with their referrers while labeling organic signups without a referral. | https://www.tensortonic.com/problems/sql-self-join |
| WHERE Clauses | Filter employees by department and salary with SQL WHERE conditions, returning only qualifying names and salaries. | https://www.tensortonic.com/problems/sql-where-clauses |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Implement Triplet Loss | Compute triplet loss from anchor, positive, and negative embeddings using distances and a margin. | https://www.tensortonic.com/problems/triplet-loss |
| Block-Pointer Matmul | Implement tiled Triton matrix multiplication with block pointers, boundary checks, and tail-safe loads and stores. | https://www.tensortonic.com/problems/triton-block-pointer-matmul |
| Cross Entropy Loss (Mean Reduction) | Implement mean categorical cross-entropy in Triton with stable row-wise log-sum-exp and atomic loss accumulation. | https://www.tensortonic.com/problems/triton-cross-entropy |
| Dropout (Inverted Scaling) | Implement inverted dropout in Triton with a supplied mask, register scaling, and tail-safe tiled memory access. | https://www.tensortonic.com/problems/triton-dropout |
| Fused Matmul + Bias + ReLU | Fuse tiled matrix multiplication, per-column bias, and ReLU in one Triton kernel with tail-safe memory access. | https://www.tensortonic.com/problems/triton-fused-matmul-bias-relu |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/problems/triton-fused-multiply-add |
| Fused Row-Wise Softmax | Implement fused row-wise softmax in Triton with stable max subtraction, register reductions, and masked column tails. | https://www.tensortonic.com/problems/triton-fused-softmax |
| GELU | Implement exact GELU activation in Triton with device error-function math and masked contiguous tiles. | https://www.tensortonic.com/problems/triton-gelu |
| GEMV: Matrix Vector Product | Implement Triton matrix-vector multiplication with row-block programs, float32 accumulation, and masked matrix tails. | https://www.tensortonic.com/problems/triton-gemv |
| Grouped Program-ID Matmul | Implement grouped program-ID matrix multiplication in Triton to improve L2 reuse while preserving tail-safe tiled computation. | https://www.tensortonic.com/problems/triton-grouped-matmul |
| KV Cache Append | Append one autoregressive decoding row to key and value caches in Triton without modifying other cache positions. | https://www.tensortonic.com/problems/triton-kv-append |
| L2 Vector Norm | Compute a Triton L2 vector norm with tiled sum-of-squares reduction, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/problems/triton-l2-norm |
| LayerNorm Forward | Implement LayerNorm forward in Triton with per-row mean and variance reductions, affine parameters, and masked tails. | https://www.tensortonic.com/problems/triton-layernorm |
| Row-Wise LogSumExp | Implement numerically stable row-wise LogSumExp in Triton with max subtraction and masked register reductions. | https://www.tensortonic.com/problems/triton-logsumexp |
| Tiled Matrix Multiplication | Implement tiled matrix multiplication in Triton with a two-dimensional grid, float32 accumulation, and tail masks. | https://www.tensortonic.com/problems/triton-matmul |
| Autotuned Matrix Multiplication | Autotune Triton matrix multiplication across tile and pipeline configurations while preserving masked boundary handling. | https://www.tensortonic.com/problems/triton-matmul-autotune |
| Vector Max Reduction | Compute a vector maximum with one Triton reduction program and masked tail lanes that cannot win comparisons. | https://www.tensortonic.com/problems/triton-max |
| Single-Pass Mean and Variance | Compute population mean and variance in Triton with single-pass statistics, atomic partials, and masked tails. | https://www.tensortonic.com/problems/triton-mean-variance |
| Online Softmax | Implement chunked online softmax in Triton with running maxima and denominators followed by a normalized output pass. | https://www.tensortonic.com/problems/triton-online-softmax |
| ReLU | Implement ReLU activation in Triton with contiguous program tiles, branch-free rectification, and masked tails. | https://www.tensortonic.com/problems/triton-relu |
| RMSNorm Forward | Implement RMSNorm forward in Triton with per-row square reduction, numerical stability, scaling, and masked tails. | https://www.tensortonic.com/problems/triton-rmsnorm |
| Rotary Position Embedding | Implement Rotary Position Embeddings in Triton with per-token pair rotations, precomputed sine and cosine, and tail masks. | https://www.tensortonic.com/problems/triton-rope |
| SiLU | Implement fused SiLU or Swish activation in Triton with contiguous tiles, sigmoid weighting, and masked tails. | https://www.tensortonic.com/problems/triton-silu |
| Split-K Matmul | Implement split-K matrix multiplication in Triton with parallel reduction slices, atomic output accumulation, and tail masks. | https://www.tensortonic.com/problems/triton-split-k-matmul |
| Vector Sum Reduction | Implement tiled vector sum reduction in Triton with register partials, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/problems/triton-sum |
| Tiled Transpose | Implement tiled matrix transpose in Triton by swapping load and store strides with masked boundary tiles. | https://www.tensortonic.com/problems/triton-transpose |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/problems/triton-vector-addition |
| Vectorized Vector Add | Implement vector addition in Triton with larger per-program tiles to reduce launch overhead while masking the final tail. | https://www.tensortonic.com/problems/triton-vectorized-load |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/mithun_kannaa)
<!-- tensortonic:end -->
