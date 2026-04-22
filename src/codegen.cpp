#include "codegen.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace tensorjit {
namespace {

bool isMatmulCall(const ASTNode* node) {
    const auto* call = dynamic_cast<const FuncCallNode*>(node);
    return call && call->name == "matmul";
}

} // namespace

CodeGen::CodeGen(std::string moduleName)
    : ctx_(std::make_unique<llvm::LLVMContext>()),
      mod_(std::make_unique<llvm::Module>(std::move(moduleName), *ctx_)),
      builder_(*ctx_) {}

llvm::Module& CodeGen::module() {
    return *mod_;
}

llvm::LLVMContext& CodeGen::context() {
    return *ctx_;
}

std::unique_ptr<llvm::Module> CodeGen::takeModule() {
    return std::move(mod_);
}

std::unique_ptr<llvm::LLVMContext> CodeGen::takeContext() {
    return std::move(ctx_);
}

llvm::Function* CodeGen::generateProgram(const Program& program) {
    llvm::Function* last = nullptr;
    int matmulCount = 0;
    for (const auto& stmt : program.stmts) {
        if (isMatmulCall(stmt.rhs.get())) {
            const std::string name = matmulCount++ == 0 ? "matmul_kernel" : stmt.lhs + "_matmul_kernel";
            last = generateMatmulKernel(stmt.tile_hint, name);
        } else if (const auto* call = dynamic_cast<const FuncCallNode*>(stmt.rhs.get())) {
            if (call->name == "relu") {
                last = generateReluKernel(stmt.lhs + "_relu_kernel");
            } else if (call->name == "softmax") {
                last = generateSoftmaxKernel(stmt.lhs + "_softmax_kernel");
            }
        }
    }
    return last;
}

llvm::Function* CodeGen::generateMatmulKernel(int tileHint, const std::string& name) {
    auto* floatTy = builder_.getFloatTy();
    auto* intTy = builder_.getInt32Ty();
    auto* ptrTy = llvm::PointerType::getUnqual(*ctx_);

    auto* fnTy = llvm::FunctionType::get(builder_.getVoidTy(),
                                         {ptrTy, ptrTy, ptrTy, intTy, intTy, intTy},
                                         false);
    auto* fn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, name, mod_.get());

    auto it = fn->arg_begin();
    llvm::Value* A = &*it++;
    A->setName("A");
    llvm::Value* B = &*it++;
    B->setName("B");
    llvm::Value* C = &*it++;
    C->setName("C");
    llvm::Value* M = &*it++;
    M->setName("M");
    llvm::Value* N = &*it++;
    N->setName("N");
    llvm::Value* K = &*it++;
    K->setName("K");

    auto* entry = llvm::BasicBlock::Create(*ctx_, "entry", fn);
    builder_.SetInsertPoint(entry);
    (void)floatTy;

    if (tileHint > 0) {
        emitTiledMatmul(fn, A, B, C, M, N, K, tileHint);
    } else {
        emitUntiledMatmul(fn, A, B, C, M, N, K);
    }

    builder_.CreateRetVoid();
    verifyOrAbort(fn);
    return fn;
}

llvm::Function* CodeGen::generateReluKernel(const std::string& name) {
    auto* floatTy = builder_.getFloatTy();
    auto* intTy = builder_.getInt32Ty();
    auto* ptrTy = llvm::PointerType::getUnqual(*ctx_);
    auto* fnTy = llvm::FunctionType::get(builder_.getVoidTy(), {ptrTy, ptrTy, intTy}, false);
    auto* fn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, name, mod_.get());

    auto it = fn->arg_begin();
    llvm::Value* X = &*it++;
    X->setName("X");
    llvm::Value* Y = &*it++;
    Y->setName("Y");
    llvm::Value* Len = &*it++;
    Len->setName("Len");

    auto* entry = llvm::BasicBlock::Create(*ctx_, "entry", fn);
    builder_.SetInsertPoint(entry);
    auto* zeroI = builder_.getInt32(0);
    auto* oneI = builder_.getInt32(1);
    auto* zeroF = llvm::ConstantFP::get(floatTy, 0.0);

    createForLoop(fn, "idx", zeroI, Len, oneI, [&](llvm::Value* idx) {
        auto* xPtr = createFloatPtrGEP(X, idx, "x.ptr");
        auto* yPtr = createFloatPtrGEP(Y, idx, "y.ptr");
        auto* x = builder_.CreateLoad(floatTy, xPtr, "x");
        auto* cmp = builder_.CreateFCmpOGT(x, zeroF, "x.gt.zero");
        auto* y = builder_.CreateSelect(cmp, x, zeroF, "relu");
        builder_.CreateStore(y, yPtr);
    });

    builder_.CreateRetVoid();
    verifyOrAbort(fn);
    return fn;
}

llvm::Function* CodeGen::generateSoftmaxKernel(const std::string& name) {
    auto* floatTy = builder_.getFloatTy();
    auto* intTy = builder_.getInt32Ty();
    auto* ptrTy = llvm::PointerType::getUnqual(*ctx_);
    auto* fnTy = llvm::FunctionType::get(builder_.getVoidTy(), {ptrTy, ptrTy, intTy}, false);
    auto* fn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, name, mod_.get());

    auto it = fn->arg_begin();
    llvm::Value* X = &*it++;
    X->setName("X");
    llvm::Value* Y = &*it++;
    Y->setName("Y");
    llvm::Value* Len = &*it++;
    Len->setName("Len");

    auto* entry = llvm::BasicBlock::Create(*ctx_, "entry", fn);
    builder_.SetInsertPoint(entry);
    auto* zeroI = builder_.getInt32(0);
    auto* oneI = builder_.getInt32(1);
    auto* zeroF = llvm::ConstantFP::get(floatTy, 0.0);
    auto* expFn = llvm::Intrinsic::getOrInsertDeclaration(mod_.get(), llvm::Intrinsic::exp, {floatTy});
    auto* sumAlloca = createEntryAlloca(fn, floatTy, "sum");
    builder_.CreateStore(zeroF, sumAlloca);

    createForLoop(fn, "sum.idx", zeroI, Len, oneI, [&](llvm::Value* idx) {
        auto* xPtr = createFloatPtrGEP(X, idx, "x.ptr");
        auto* x = builder_.CreateLoad(floatTy, xPtr, "x");
        auto* e = builder_.CreateCall(expFn, {x}, "exp");
        auto* oldSum = builder_.CreateLoad(floatTy, sumAlloca, "sum.old");
        builder_.CreateStore(builder_.CreateFAdd(oldSum, e, "sum.next"), sumAlloca);
    });

    createForLoop(fn, "norm.idx", zeroI, Len, oneI, [&](llvm::Value* idx) {
        auto* xPtr = createFloatPtrGEP(X, idx, "x.ptr");
        auto* yPtr = createFloatPtrGEP(Y, idx, "y.ptr");
        auto* x = builder_.CreateLoad(floatTy, xPtr, "x");
        auto* e = builder_.CreateCall(expFn, {x}, "exp");
        auto* sum = builder_.CreateLoad(floatTy, sumAlloca, "sum");
        builder_.CreateStore(builder_.CreateFDiv(e, sum, "softmax"), yPtr);
    });

    builder_.CreateRetVoid();
    verifyOrAbort(fn);
    return fn;
}

llvm::AllocaInst* CodeGen::createEntryAlloca(llvm::Function* fn, llvm::Type* type, const std::string& name) {
    llvm::IRBuilder<> tmp(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    return tmp.CreateAlloca(type, nullptr, name);
}

void CodeGen::createForLoop(llvm::Function* fn,
                            const std::string& name,
                            llvm::Value* start,
                            llvm::Value* end,
                            llvm::Value* step,
                            const LoopBody& body) {
    auto* intTy = builder_.getInt32Ty();
    auto* idxAlloca = createEntryAlloca(fn, intTy, name + ".addr");
    builder_.CreateStore(start, idxAlloca);

    auto* condBB = llvm::BasicBlock::Create(*ctx_, name + ".cond", fn);
    auto* bodyBB = llvm::BasicBlock::Create(*ctx_, name + ".body", fn);
    auto* afterBB = llvm::BasicBlock::Create(*ctx_, name + ".after", fn);

    builder_.CreateBr(condBB);
    builder_.SetInsertPoint(condBB);
    auto* idx = builder_.CreateLoad(intTy, idxAlloca, name);
    auto* cond = builder_.CreateICmpSLT(idx, end, name + ".lt.end");
    builder_.CreateCondBr(cond, bodyBB, afterBB);

    builder_.SetInsertPoint(bodyBB);
    auto* bodyIdx = builder_.CreateLoad(intTy, idxAlloca, name + ".body.idx");
    body(bodyIdx);
    if (!builder_.GetInsertBlock()->getTerminator()) {
        auto* cur = builder_.CreateLoad(intTy, idxAlloca, name + ".cur");
        auto* next = builder_.CreateAdd(cur, step, name + ".next");
        builder_.CreateStore(next, idxAlloca);
        builder_.CreateBr(condBB);
    }

    builder_.SetInsertPoint(afterBB);
}

llvm::Value* CodeGen::createMinInt(llvm::Value* lhs, llvm::Value* rhs) {
    auto* cmp = builder_.CreateICmpSLT(lhs, rhs, "min.cmp");
    return builder_.CreateSelect(cmp, lhs, rhs, "min");
}

llvm::Value* CodeGen::createFloatPtrGEP(llvm::Value* base, llvm::Value* index, const std::string& name) {
    return builder_.CreateGEP(builder_.getFloatTy(), base, index, name);
}

void CodeGen::emitUntiledMatmul(llvm::Function* fn,
                                llvm::Value* A,
                                llvm::Value* B,
                                llvm::Value* C,
                                llvm::Value* M,
                                llvm::Value* N,
                                llvm::Value* K) {
    auto* intTy = builder_.getInt32Ty();
    auto* floatTy = builder_.getFloatTy();
    auto* zeroI = builder_.getInt32(0);
    auto* oneI = builder_.getInt32(1);
    auto* zeroF = llvm::ConstantFP::get(floatTy, 0.0);

    createForLoop(fn, "i", zeroI, M, oneI, [&](llvm::Value* i) {
        createForLoop(fn, "j", zeroI, N, oneI, [&](llvm::Value* j) {
            auto* sumAlloca = createEntryAlloca(fn, floatTy, "sum");
            builder_.CreateStore(zeroF, sumAlloca);
            createForLoop(fn, "k", zeroI, K, oneI, [&](llvm::Value* k) {
                auto* aIdx = builder_.CreateAdd(builder_.CreateMul(i, K), k, "a.idx");
                auto* bIdx = builder_.CreateAdd(builder_.CreateMul(k, N), j, "b.idx");
                auto* a = builder_.CreateLoad(floatTy, createFloatPtrGEP(A, aIdx, "a.ptr"), "a");
                auto* b = builder_.CreateLoad(floatTy, createFloatPtrGEP(B, bIdx, "b.ptr"), "b");
                auto* oldSum = builder_.CreateLoad(floatTy, sumAlloca, "sum.old");
                builder_.CreateStore(builder_.CreateFAdd(oldSum, builder_.CreateFMul(a, b, "prod"), "sum.next"), sumAlloca);
            });
            auto* cIdx = builder_.CreateAdd(builder_.CreateMul(i, N), j, "c.idx");
            auto* sum = builder_.CreateLoad(floatTy, sumAlloca, "sum");
            builder_.CreateStore(sum, createFloatPtrGEP(C, cIdx, "c.ptr"));
        });
    });
    (void)intTy;
}

void CodeGen::emitTiledMatmul(llvm::Function* fn,
                              llvm::Value* A,
                              llvm::Value* B,
                              llvm::Value* C,
                              llvm::Value* M,
                              llvm::Value* N,
                              llvm::Value* K,
                              int tile) {
    auto* floatTy = builder_.getFloatTy();
    auto* zeroI = builder_.getInt32(0);
    auto* oneI = builder_.getInt32(1);
    auto* tileI = builder_.getInt32(tile);
    auto* zeroF = llvm::ConstantFP::get(floatTy, 0.0);

    createForLoop(fn, "zi", zeroI, M, oneI, [&](llvm::Value* i) {
        createForLoop(fn, "zj", zeroI, N, oneI, [&](llvm::Value* j) {
            auto* cIdx = builder_.CreateAdd(builder_.CreateMul(i, N), j, "zero.c.idx");
            builder_.CreateStore(zeroF, createFloatPtrGEP(C, cIdx, "zero.c.ptr"));
        });
    });

    createForLoop(fn, "ii", zeroI, M, tileI, [&](llvm::Value* ii) {
        createForLoop(fn, "jj", zeroI, N, tileI, [&](llvm::Value* jj) {
            createForLoop(fn, "kk", zeroI, K, tileI, [&](llvm::Value* kk) {
                auto* iEnd = createMinInt(builder_.CreateAdd(ii, tileI, "ii.tile.end"), M);
                auto* jEnd = createMinInt(builder_.CreateAdd(jj, tileI, "jj.tile.end"), N);
                auto* kEnd = createMinInt(builder_.CreateAdd(kk, tileI, "kk.tile.end"), K);
                createForLoop(fn, "ti", ii, iEnd, oneI, [&](llvm::Value* i) {
                    createForLoop(fn, "tj", jj, jEnd, oneI, [&](llvm::Value* j) {
                        createForLoop(fn, "tk", kk, kEnd, oneI, [&](llvm::Value* k) {
                            auto* aIdx = builder_.CreateAdd(builder_.CreateMul(i, K), k, "tile.a.idx");
                            auto* bIdx = builder_.CreateAdd(builder_.CreateMul(k, N), j, "tile.b.idx");
                            auto* cIdx = builder_.CreateAdd(builder_.CreateMul(i, N), j, "tile.c.idx");
                            auto* cPtr = createFloatPtrGEP(C, cIdx, "tile.c.ptr");
                            auto* a = builder_.CreateLoad(floatTy, createFloatPtrGEP(A, aIdx, "tile.a.ptr"), "tile.a");
                            auto* b = builder_.CreateLoad(floatTy, createFloatPtrGEP(B, bIdx, "tile.b.ptr"), "tile.b");
                            auto* oldC = builder_.CreateLoad(floatTy, cPtr, "tile.c.old");
                            auto* next = builder_.CreateFAdd(oldC, builder_.CreateFMul(a, b, "tile.prod"), "tile.c.next");
                            builder_.CreateStore(next, cPtr);
                        });
                    });
                });
            });
        });
    });
}

void CodeGen::verifyOrAbort(llvm::Function* fn) {
    if (llvm::verifyFunction(*fn, &llvm::errs())) {
        llvm::errs() << "LLVM function verification failed for " << fn->getName() << "\n";
        std::abort();
    }
}

} // namespace tensorjit
