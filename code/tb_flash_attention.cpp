#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

static std::string get_tb_dir() {
    std::string path = __FILE__;
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? "." : path.substr(0, pos);
}

static const std::string tb_dir = get_tb_dir();

// Must match the defines in flash_attention.cpp
#define N_MAX 64
#define D_MAX 32

// Forward declaration of DUT
// Note: causal is int (not bool) for AXI4-Lite register compatibility.
void flash_attention_hls(
    float *Q,
    float *K,
    float *V,
    float *O,
    int N,
    int d,
    int causal
);

struct TestVector {
    int test_id;
    int N;
    int d;
    int block_size;   // block size used by the Python golden model — informational only.
                      // The HLS DUT uses compile-time constants BQ=BK=8. For tests
                      // where block_size != BQ/BK the algorithm is still mathematically
                      // equivalent; only the tile boundaries differ.
    int causal;
    std::vector<float> Q_flat;
    std::vector<float> K_flat;
    std::vector<float> V_flat;
    std::vector<float> O_expected_flat;  // from attention_naive (float64 reference)
    float max_abs_err_python;
};

// Split a CSV line while respecting quoted fields
std::vector<std::string> split_csv_quoted(const std::string &line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;

    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    fields.push_back(cur);
    return fields;
}

// Parse semicolon-separated float list
std::vector<float> parse_float_list(const std::string &s) {
    std::vector<float> vals;
    std::stringstream ss(s);
    std::string token;

    while (std::getline(ss, token, ';')) {
        if (!token.empty()) {
            vals.push_back(std::stof(token));
        }
    }
    return vals;
}

bool parse_csv_line(const std::string &line, TestVector &tv) {
    auto fields = split_csv_quoted(line);
    if (fields.size() != 10) {
        return false;
    }

    tv.test_id             = std::stoi(fields[0]);
    tv.N                   = std::stoi(fields[1]);
    tv.d                   = std::stoi(fields[2]);
    tv.block_size          = std::stoi(fields[3]);
    tv.causal              = std::stoi(fields[4]);
    tv.Q_flat              = parse_float_list(fields[5]);
    tv.K_flat              = parse_float_list(fields[6]);
    tv.V_flat              = parse_float_list(fields[7]);
    tv.O_expected_flat     = parse_float_list(fields[8]);
    tv.max_abs_err_python  = std::stof(fields[9]);

    return true;
}

int main(int argc, char** argv) {
    bool test_rtl = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "rtl") {
            test_rtl = true;
            break;
        }
    }

    std::string csv_path = tb_dir + "/test_outputs/tv_attention_python.csv";
    std::cout << "Resolved CSV path = " << csv_path << std::endl;

    std::ifstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "Error: Could not open CSV file" << std::endl;
        return 1;
    }

    std::string vitis_path;
    if (test_rtl) {
        std::cout << "Running in RTL co-simulation mode" << std::endl;
        vitis_path = tb_dir + "/test_outputs/tv_attention_rtl.csv";
    } else {
        std::cout << "Running in C simulation mode" << std::endl;
        vitis_path = tb_dir + "/test_outputs/tv_attention_csim.csv";
    }

    std::ofstream vitis_csv(vitis_path);
    if (!vitis_csv.is_open()) {
        std::cerr << "Error: Could not open output CSV file" << std::endl;
        return 1;
    }

    vitis_csv << "test_id,N,d,block_size,causal,max_abs_err_python,max_abs_err_dut" << std::endl;

    std::vector<TestVector> test_vectors;
    std::string line;

    // Skip header
    std::getline(csv, line);

    while (std::getline(csv, line)) {
        if (line.empty()) continue;

        TestVector tv;
        if (parse_csv_line(line, tv)) {
            test_vectors.push_back(tv);
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }
    csv.close();

    int pass_count = 0;
    int fail_count = 0;

    std::cout << "========================================" << std::endl;
    std::cout << "FlashAttention HLS Testbench" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total test vectors: " << test_vectors.size() << std::endl;
    std::cout << std::endl;

    // Tolerance accounts for float32 DUT vs float64 reference.
    // Empirically, float32 flash-attention accumulation error is well below 1e-4.
    const float tol_out = 1e-4f;

    for (size_t t = 0; t < test_vectors.size(); t++) {
        const TestVector &tv = test_vectors[t];

        // Bounds check: skip tests that exceed HLS array dimensions
        if (tv.N > N_MAX || tv.d > D_MAX) {
            std::cerr << "Test " << tv.test_id
                      << ": N=" << tv.N << " d=" << tv.d
                      << " exceeds HLS limits (N_MAX=" << N_MAX
                      << ", D_MAX=" << D_MAX << ") — SKIP" << std::endl;
            fail_count++;
            continue;
        }

        // Flat 1D arrays matching the DUT's pointer-based AXI4 Master interface.
        // Elements are stored row-major with stride D_MAX (matching the DUT's indexing).
        float Q[N_MAX * D_MAX] = {};
        float K[N_MAX * D_MAX] = {};
        float V[N_MAX * D_MAX] = {};
        float O[N_MAX * D_MAX] = {};

        if ((int)tv.Q_flat.size() != tv.N * tv.d ||
            (int)tv.K_flat.size() != tv.N * tv.d ||
            (int)tv.V_flat.size() != tv.N * tv.d ||
            (int)tv.O_expected_flat.size() != tv.N * tv.d) {
            std::cerr << "Size mismatch in test " << tv.test_id << std::endl;
            fail_count++;
            continue;
        }

        // Load inputs into flat arrays with D_MAX stride
        for (int i = 0; i < tv.N; i++) {
            for (int j = 0; j < tv.d; j++) {
                int src = i * tv.d + j;
                int dst = i * D_MAX + j;
                Q[dst] = tv.Q_flat[src];
                K[dst] = tv.K_flat[src];
                V[dst] = tv.V_flat[src];
            }
        }

        // Run DUT (always uses compile-time BQ=BK=8; block_size from CSV is informational)
        flash_attention_hls(Q, K, V, O, tv.N, tv.d, tv.causal);

        // Compare DUT output against float64 reference (parsed back as float32)
        float max_abs_err_dut = 0.0f;
        for (int i = 0; i < tv.N; i++) {
            for (int j = 0; j < tv.d; j++) {
                int ref_idx = i * tv.d + j;
                int dut_idx = i * D_MAX + j;
                float expected = tv.O_expected_flat[ref_idx];
                float got = O[dut_idx];
                float err = std::fabs(got - expected);
                if (err > max_abs_err_dut) {
                    max_abs_err_dut = err;
                }
            }
        }

        vitis_csv << tv.test_id << ","
                  << tv.N << ","
                  << tv.d << ","
                  << tv.block_size << ","
                  << tv.causal << ","
                  << std::fixed << std::setprecision(9)
                  << tv.max_abs_err_python << ","
                  << max_abs_err_dut << std::endl;

        bool test_pass = (max_abs_err_dut < tol_out);

        std::cout << "Test " << tv.test_id
                  << ": N=" << tv.N
                  << ", d=" << tv.d
                  << ", causal=" << tv.causal;

        if (test_pass) {
            std::cout << " [PASS]";
            pass_count++;
        } else {
            std::cout << " [FAIL]";
            fail_count++;
        }

        std::cout << "  max_abs_err_dut=" << std::scientific << max_abs_err_dut
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test Results: " << pass_count << " passed, "
              << fail_count << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    vitis_csv.close();

    return (fail_count == 0) ? 0 : 1;
}
