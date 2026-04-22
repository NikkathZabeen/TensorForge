#include "jit.h"

#include "codegen.h"
#include "optimizer.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tensorjit {
namespace {

std::unique_ptr<llvm::orc::LLJIT> createJit() {
    static bool initialized = [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        return true;
    }();
    (void)initialized;

    auto created = llvm::orc::LLJITBuilder().create();
    if (!created) {
        throw std::runtime_error(llvm::toString(created.takeError()));
    }
    return std::move(*created);
}

std::vector<std::unique_ptr<llvm::orc::LLJIT>>& ownedJits() {
    static std::vector<std::unique_ptr<llvm::orc::LLJIT>> jits;
    return jits;
}

} // namespace

MatmulKernelFn compileMatmulKernel(std::unique_ptr<llvm::Module> module,
                                   std::unique_ptr<llvm::LLVMContext> context) {
    auto jit = createJit();
    module->setDataLayout(jit->getDataLayout());
    runPassPipeline(*module);

    if (auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(context)))) {
        throw std::runtime_error(llvm::toString(std::move(err)));
    }

    auto sym = jit->lookup("matmul_kernel");
    if (!sym) {
        throw std::runtime_error(llvm::toString(sym.takeError()));
    }
    auto fn = sym->toPtr<MatmulKernelFn>();
    ownedJits().push_back(std::move(jit));
    return fn;
}

bool runJitIdentitySmokeTest(int tileHint) {
    constexpr int M = 4;
    constexpr int N = 4;
    constexpr int K = 4;

    std::vector<float> A(M * K, 0.0f);
    std::vector<float> B(K * N, 0.0f);
    std::vector<float> C(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        A[i * K + i] = 1.0f;
        B[i * N + i] = 1.0f;
    }

    CodeGen codegen("jit_smoke");
    codegen.generateMatmulKernel(tileHint);
    auto fn = compileMatmulKernel(codegen.takeModule(), codegen.takeContext());
    fn(A.data(), B.data(), C.data(), M, N, K);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float expected = i == j ? 1.0f : 0.0f;
            if (std::fabs(C[i * N + j] - expected) > 1.0e-5f) {
                std::cerr << "JIT smoke test mismatch at " << i << "," << j
                          << ": got " << C[i * N + j] << ", expected " << expected << "\n";
                return false;
            }
        }
    }

    return true;
}

} // namespace tensorjit
