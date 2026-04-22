#include "optimizer.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

namespace tensorjit {

void runPassPipeline(llvm::Module& module) {
    llvm::legacy::PassManager pm;
    pm.add(llvm::createPromoteMemoryToRegisterPass());
    pm.add(llvm::createInstructionCombiningPass());
    pm.add(llvm::createReassociatePass());
    pm.add(llvm::createGVNPass());
    pm.add(llvm::createCFGSimplificationPass());
    pm.run(module);
}

} // namespace tensorjit
