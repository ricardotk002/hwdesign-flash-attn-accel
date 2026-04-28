#include <hls_math.h>
#include <ap_fixed.h>

#define N_MAX 64
#define D_MAX 32
#define BQ    8
#define BK    8

// Sentinel for -infinity in float32. exp(NEG_INF) underflows to 0,
// which is the desired behavior for both causal masking and m initialization.
#define NEG_INF (-1e30f)

typedef float data_t;
// Switch to fixed-point for synthesis on PYNQ-Z2:
// typedef ap_fixed<16,6> data_t;

void flash_attention_hls(
    data_t *Q,
    data_t *K,
    data_t *V,
    data_t *O,
    int N,
    int d,
    int causal      // bool mapped as int for AXI4-Lite compatibility
) {
#pragma HLS INLINE off

// ── AXI4 Master ports (burst read/write to DDR) ──────────────────────────────
#pragma HLS INTERFACE m_axi port=Q depth=4096 bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi port=K depth=4096 bundle=gmem1 offset=slave
#pragma HLS INTERFACE m_axi port=V depth=4096 bundle=gmem2 offset=slave
#pragma HLS INTERFACE m_axi port=O depth=4096 bundle=gmem3 offset=slave

// ── AXI4-Lite slave (control registers: N, d, causal, base addresses) ────────
#pragma HLS INTERFACE s_axilite port=Q       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=K       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=V       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=O       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=N       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=d       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=causal  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return  bundle=ctrl

    data_t Qbuf[BQ][D_MAX];
    data_t Kbuf[BK][D_MAX];
    data_t Vbuf[BK][D_MAX];
    data_t scores[BQ][BK];
    data_t acc[BQ][D_MAX];
    data_t m[BQ];
    data_t l[BQ];

#pragma HLS ARRAY_PARTITION variable=Qbuf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=Kbuf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=Vbuf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=acc  cyclic factor=4 dim=2

    const data_t inv_sqrt_d = 1.0f / hls::sqrtf((float)d);

    Q_TILE_LOOP:
    for (int q0 = 0; q0 < N; q0 += BQ) {
        int q_lim = ((q0 + BQ) < N) ? BQ : (N - q0);

        // --- Q/K/V Loader: load Q tile into on-chip buffer ---
        LOAD_Q:
        for (int i = 0; i < q_lim; i++) {
            for (int x = 0; x < d; x++) {
#pragma HLS PIPELINE II=1
                Qbuf[i][x] = Q[(q0 + i) * D_MAX + x];
            }
        }

        // --- Top-Level Controller: initialize online softmax state ---
        INIT_STATE:
        for (int i = 0; i < q_lim; i++) {
            m[i] = NEG_INF;
            l[i] = 0.0f;
            for (int x = 0; x < d; x++) {
#pragma HLS PIPELINE II=1
                acc[i][x] = 0.0f;
            }
        }

        K_TILE_LOOP:
        for (int k0 = 0; k0 < N; k0 += BK) {
            int k_lim = ((k0 + BK) < N) ? BK : (N - k0);

            // --- Q/K/V Loader: load K and V tiles ---
            LOAD_KV:
            for (int j = 0; j < k_lim; j++) {
                for (int x = 0; x < d; x++) {
#pragma HLS PIPELINE II=1
                    Kbuf[j][x] = K[(k0 + j) * D_MAX + x];
                    Vbuf[j][x] = V[(k0 + j) * D_MAX + x];
                }
            }

            // --- Dot-Product Engine: compute scaled scores ---
            // Pipeline the j loop; fully unroll the inner dot-product over d
            // so HLS can achieve II=1 on the j loop.
            SCORE_LOOP_I:
            for (int i = 0; i < q_lim; i++) {
                SCORE_LOOP_J:
                for (int j = 0; j < k_lim; j++) {
#pragma HLS PIPELINE II=1
                    data_t dot = 0.0f;
                    SCORE_LOOP_X:
                    for (int x = 0; x < D_MAX; x++) {
#pragma HLS UNROLL
                        if (x < d) dot += Qbuf[i][x] * Kbuf[j][x];
                    }

                    data_t s = dot * inv_sqrt_d;

                    // Causal mask: future positions get NEG_INF so their
                    // exponential underflows to 0 (same effect as -inf).
                    if (causal && ((k0 + j) > (q0 + i))) {
                        s = NEG_INF;
                    }

                    scores[i][j] = s;
                }
            }

            // --- Online Softmax Unit + Weighted Value Accumulator ---
            // Row-by-row: find tile max, compute exponentials, update
            // running (m, l, acc). The accumulator stays normalized at
            // every step, so no separate final-normalization pass is needed.
            UPDATE_LOOP_I:
            for (int i = 0; i < q_lim; i++) {
                // Online Softmax Unit: find tile row max
                data_t tile_max = NEG_INF;
                FIND_MAX:
                for (int j = 0; j < k_lim; j++) {
#pragma HLS PIPELINE II=1
                    if (scores[i][j] > tile_max) tile_max = scores[i][j];
                }

                data_t m_new     = (m[i] > tile_max) ? m[i] : tile_max;
                data_t old_scale = hls::expf(m[i] - m_new);   // correction factor

                // Online Softmax Unit: compute exponentials and
                // accumulate weighted V values for this tile.
                data_t exp_sum = 0.0f;
                data_t weighted[D_MAX];
#pragma HLS ARRAY_PARTITION variable=weighted cyclic factor=4 dim=1

                INIT_WEIGHTED:
                for (int x = 0; x < D_MAX; x++) {
#pragma HLS UNROLL
                    weighted[x] = 0.0f;
                }

                EXP_ACCUM:
                for (int j = 0; j < k_lim; j++) {
                    data_t e = hls::expf(scores[i][j] - m_new);
                    exp_sum += e;
                    for (int x = 0; x < D_MAX; x++) {
#pragma HLS UNROLL factor=4
                        if (x < d) weighted[x] += e * Vbuf[j][x];
                    }
                }

                // Online Softmax Unit: update l
                data_t l_new = old_scale * l[i] + exp_sum;

                // Weighted Value Accumulator: update normalized running output.
                // Division by l_new keeps acc normalized after every tile.
                UPDATE_ACC:
                for (int x = 0; x < D_MAX; x++) {
#pragma HLS UNROLL factor=4
                    if (x < d) {
                        acc[i][x] =
                            (old_scale * l[i] * acc[i][x] + weighted[x]) / l_new;
                    }
                }

                m[i] = m_new;
                l[i] = l_new;
            }
        }

        // --- Output Writeback: store final normalized output tile ---
        STORE_O:
        for (int i = 0; i < q_lim; i++) {
            for (int x = 0; x < d; x++) {
#pragma HLS PIPELINE II=1
                O[(q0 + i) * D_MAX + x] = acc[i][x];
            }
        }
    }
}
