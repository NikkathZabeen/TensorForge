#pragma once

#include "ast.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <functional>
#include <memory>
#include <string>

namespace tensorjit {

class CodeGen {
public:
    explicit CodeGen(std::string moduleName = "tensorjit");

    llvm::Module& module();
    llvm::LLVMContext& context();

    llvm::Function* generateProgram(const Program& program);
    llvm::Function* generateMatmulKernel(int tileHint = -1, const std::string& name = "matmul_kernel");
    llvm::Function* generateReluKernel(const std::string& name = "relu_kernel");
    llvm::Function* generateSoftmaxKernel(const std::string& name = "softmax_kernel");

    std::unique_ptr<llvm::Module> takeModule();
    std::unique_ptr<llvm::LLVMContext> takeContext();

private:
    using LoopBody = std::function<void(llvm::Value*)>;

    llvm::AllocaInst* createEntryAlloca(llvm::Function* fn, llvm::Type* type, const std::string& name);
    void createForLoop(llvm::Function* fn,
                       const std::string& name,
                       llvm::Value* start,
                       llvm::Value* end,
                       llvm::Value* step,
                       const LoopBody& body);
    llvm::Value* createMinInt(llvm::Value* lhs, llvm::Value* rhs);
    llvm::Value* createFloatPtrGEP(llvm::Value* base, llvm::Value* index, const std::string& name);

    void emitUntiledMatmul(llvm::Function* fn,
                           llvm::Value* A,
                           llvm::Value* B,
                           llvm::Value* C,
                           llvm::Value* M,
                           llvm::Value* N,
                           llvm::Value* K);
    void emitTiledMatmul(llvm::Function* fn,
                         llvm::Value* A,
                         llvm::Value* B,
                         llvm::Value* C,
                         llvm::Value* M,
                         llvm::Value* N,
                         llvm::Value* K,
                         int tile);

    void verifyOrAbort(llvm::Function* fn);

    std::unique_ptr<llvm::LLVMContext> ctx_;
    std::unique_ptr<llvm::Module> mod_;
    llvm::IRBuilder<> builder_;
};

} // namespace tensorjit
