#include "../src/codegen.h"
#include "../src/jit.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

void naiveMatmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

double timeMs(const std::function<void()>& fn, int repeats) {
    const auto start = Clock::now();
    for (int r = 0; r < repeats; ++r) {
        fn();
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
}

tensorjit::MatmulKernelFn buildKernel(int tileHint) {
    tensorjit::CodeGen codegen(tileHint > 0 ? "bench_tile" : "bench_notile");
    codegen.generateMatmulKernel(tileHint);
    return tensorjit::compileMatmulKernel(codegen.takeModule(), codegen.takeContext());
}

int repeatsFor(int size) {
    if (size <= 64) {
        return 20;
    }
    if (size <= 256) {
        return 5;
    }
    return 1;
}

} // namespace

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto jitNoTile = buildKernel(-1);
    auto jitTile64 = buildKernel(64);

    std::cout << std::left << std::setw(10) << "Size"
              << " | " << std::right << std::setw(12) << "Naive (ms)"
              << " | " << std::setw(16) << "JIT no-tile"
              << " | " << std::setw(16) << "JIT tile=64"
              << " | " << std::setw(8) << "Speedup" << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (int size : {64, 256, 512, 1024}) {
        const int M = size;
        const int N = size;
        const int K = size;
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        std::vector<float> C(M * N);

        std::generate(A.begin(), A.end(), [&] { return dist(rng); });
        std::generate(B.begin(), B.end(), [&] { return dist(rng); });

        const int repeats = repeatsFor(size);
        const double naiveMs = timeMs([&] {
            std::fill(C.begin(), C.end(), 0.0f);
            naiveMatmul(A.data(), B.data(), C.data(), M, N, K);
        }, repeats);

        const double jitNoTileMs = timeMs([&] {
            std::fill(C.begin(), C.end(), 0.0f);
            jitNoTile(A.data(), B.data(), C.data(), M, N, K);
        }, repeats);

        const double jitTileMs = timeMs([&] {
            std::fill(C.begin(), C.end(), 0.0f);
            jitTile64(A.data(), B.data(), C.data(), M, N, K);
        }, repeats);

        const double speedup = naiveMs / jitTileMs;
        const std::string label = std::to_string(size) + "x" + std::to_string(size);
        std::cout << std::left << std::setw(10) << label
                  << " | " << std::right << std::setw(12) << std::fixed << std::setprecision(3) << naiveMs
                  << " | " << std::setw(16) << jitNoTileMs
                  << " | " << std::setw(16) << jitTileMs
                  << " | " << std::setw(7) << std::setprecision(2) << speedup << "x\n";
    }

    return 0;
}
