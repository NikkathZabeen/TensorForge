#pragma once

namespace llvm {
class Module;
}

namespace tensorjit {

void runPassPipeline(llvm::Module& module);

} // namespace tensorjit
