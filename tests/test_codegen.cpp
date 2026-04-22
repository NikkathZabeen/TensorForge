#include "../src/codegen.h"
#include "../src/jit.h"

#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <iostream>

int main() {
    {
        tensorjit::CodeGen codegen("test_codegen");
        codegen.generateMatmulKernel();
        assert(!llvm::verifyModule(codegen.module(), &llvm::errs()));
    }

    {
        tensorjit::CodeGen codegen("test_codegen_tiled");
        codegen.generateMatmulKernel(64);
        assert(!llvm::verifyModule(codegen.module(), &llvm::errs()));
    }

    assert(tensorjit::runJitIdentitySmokeTest(-1));
    assert(tensorjit::runJitIdentitySmokeTest(2));

    std::cout << "test_codegen passed\n";
    return 0;
}
