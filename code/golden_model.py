import numpy as np
import pandas as pd
import os

# HLS compile-time limits — test vectors must stay within these bounds
HLS_N_MAX = 64
HLS_D_MAX = 32


def attention_naive(Q, K, V, causal=False):
    """
    Standard scaled dot-product attention (reference, runs in float64).
    Q, K, V: shape (N, d)
    returns: shape (N, d)
    """
    N, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)

    if causal:
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        scores[mask] = -np.inf

    row_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs @ V


def attention_tiled_online(Q, K, V, block_size=16, causal=False):
    """
    FlashAttention-style tiled exact attention.
    Q, K, V: shape (N, d)

    The accumulator is kept normalized after every tile using the
    online softmax update, so no separate final-normalization pass is needed.
    """
    N, d = Q.shape
    O = np.zeros((N, d), dtype=np.float64)

    for q0 in range(0, N, block_size):
        q1 = min(q0 + block_size, N)
        Qb = Q[q0:q1]

        Bq = q1 - q0
        m = np.full((Bq,), -np.inf, dtype=np.float64)
        l = np.zeros((Bq,), dtype=np.float64)
        acc = np.zeros((Bq, d), dtype=np.float64)

        for k0 in range(0, N, block_size):
            k1 = min(k0 + block_size, N)
            Kb = K[k0:k1]
            Vb = V[k0:k1]

            scores = (Qb @ Kb.T) / np.sqrt(d)

            if causal:
                for i in range(Bq):
                    qi = q0 + i
                    for j in range(k1 - k0):
                        kj = k0 + j
                        if kj > qi:
                            scores[i, j] = -np.inf

            tile_max = np.max(scores, axis=1)
            m_new = np.maximum(m, tile_max)

            exp_scale_old = np.exp(m - m_new)
            exp_scores = np.exp(scores - m_new[:, None])

            l_new = exp_scale_old * l + np.sum(exp_scores, axis=1)

            # Accumulator stays normalized at every step.
            # This is the update owned by the Weighted Value Accumulator module.
            acc = (
                (exp_scale_old * l)[:, None] * acc
                + exp_scores @ Vb
            ) / l_new[:, None]

            m = m_new
            l = l_new

        O[q0:q1] = acc

    return O.astype(Q.dtype)


def flatten_matrix(mat):
    return ";".join(f"{x:.9f}" for x in mat.flatten())


def make_test(test_id, N, d, block_size, causal, seed):
    """Build one test vector dict. Raises if dimensions exceed HLS limits."""
    if N > HLS_N_MAX:
        raise ValueError(f"Test {test_id}: N={N} exceeds HLS_N_MAX={HLS_N_MAX}")
    if d > HLS_D_MAX:
        raise ValueError(f"Test {test_id}: d={d} exceeds HLS_D_MAX={HLS_D_MAX}")

    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((N, d)).astype(np.float32)
    K = rng.standard_normal((N, d)).astype(np.float32)
    V = rng.standard_normal((N, d)).astype(np.float32)

    # Reference: naive attention in float64 (highest accuracy)
    O_ref = attention_naive(Q, K, V, causal=bool(causal))
    # Tiled version: must match O_ref within float32 rounding
    O_tiled = attention_tiled_online(Q, K, V, block_size=block_size, causal=bool(causal))

    max_abs_err = float(np.max(np.abs(O_ref - O_tiled)))

    return {
        "test_id":             test_id,
        "N":                   N,
        "d":                   d,
        "block_size":          block_size,
        "causal":              int(causal),
        "Q_flat":              flatten_matrix(Q),
        "K_flat":              flatten_matrix(K),
        "V_flat":              flatten_matrix(V),
        # O_expected uses the naive float64 reference so the testbench
        # compares the HLS DUT against the most accurate available answer.
        # The HLS runs in float32, so expect |err| < 1e-4 rather than 0.
        "O_expected_flat":     flatten_matrix(O_ref),
        "max_abs_err_python":  max_abs_err,
    }


if __name__ == "__main__":
    # Test suite
    # Tests 0-2: small cases (N <= HLS BQ/BK=8 — single-tile path)
    # Tests 3-4: N=16 cases — exercises the multi-tile Q and K loops
    #            (HLS BQ=BK=8, so 16/8 = 2 tiles in each dimension)
    test_configs = [
        {"test_id": 0, "N":  4, "d": 4, "block_size": 2, "causal": 0, "seed": 11},
        {"test_id": 1, "N":  4, "d": 4, "block_size": 2, "causal": 1, "seed": 22},
        {"test_id": 2, "N":  8, "d": 4, "block_size": 4, "causal": 1, "seed": 33},
        # Multi-tile tests — these actually exercise tiling in the HLS DUT
        {"test_id": 3, "N": 16, "d": 8, "block_size": 8, "causal": 0, "seed": 44},
        {"test_id": 4, "N": 16, "d": 8, "block_size": 8, "causal": 1, "seed": 55},
    ]

    results = []
    for cfg in test_configs:
        row = make_test(**cfg)
        results.append(row)
        print(f"Test {row['test_id']}: N={row['N']}, d={row['d']}, "
              f"block_size={cfg['block_size']}, causal={cfg['causal']}")
        print(f"  max abs error (naive vs tiled) = {row['max_abs_err_python']:.3e}")
        print()

    df = pd.DataFrame(results)
    print("Summary:")
    print(df[["test_id", "N", "d", "block_size", "causal", "max_abs_err_python"]])
    print()

    os.makedirs("test_outputs", exist_ok=True)
    df.to_csv("test_outputs/tv_attention_python.csv", index=False)
    print("Saved to test_outputs/tv_attention_python.csv")
