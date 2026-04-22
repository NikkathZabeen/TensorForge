#pragma once

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <memory>

namespace tensorjit {

using MatmulKernelFn = void (*)(float*, float*, float*, int, int, int);

MatmulKernelFn compileMatmulKernel(std::unique_ptr<llvm::Module> module,
                                   std::unique_ptr<llvm::LLVMContext> context);

bool runJitIdentitySmokeTest(int tileHint = -1);

} // namespace tensorjit
