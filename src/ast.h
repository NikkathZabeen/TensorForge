#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensorjit {

struct ASTNode {
    virtual ~ASTNode() = default;
};

struct VarNode final : ASTNode {
    explicit VarNode(std::string name) : name(std::move(name)) {}
    std::string name;
};

struct NumberNode final : ASTNode {
    explicit NumberNode(double value) : value(value) {}
    double value;
};

struct BinaryOp final : ASTNode {
    BinaryOp(char op, std::unique_ptr<ASTNode> lhs, std::unique_ptr<ASTNode> rhs)
        : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

    char op;
    std::unique_ptr<ASTNode> lhs;
    std::unique_ptr<ASTNode> rhs;
};

struct FuncCallNode final : ASTNode {
    FuncCallNode(std::string name, std::vector<std::unique_ptr<ASTNode>> args)
        : name(std::move(name)), args(std::move(args)) {}

    std::string name;
    std::vector<std::unique_ptr<ASTNode>> args;
};

struct LetStmt {
    std::string lhs;
    std::unique_ptr<ASTNode> rhs;
    int tile_hint = -1;
};

struct Program {
    std::vector<LetStmt> stmts;
};

inline void printAST(const ASTNode* node, int indent = 0) {
    const std::string pad(static_cast<std::size_t>(indent), ' ');
    if (const auto* var = dynamic_cast<const VarNode*>(node)) {
        std::cout << pad << "Var(" << var->name << ")\n";
    } else if (const auto* num = dynamic_cast<const NumberNode*>(node)) {
        std::cout << pad << "Number(" << num->value << ")\n";
    } else if (const auto* bin = dynamic_cast<const BinaryOp*>(node)) {
        std::cout << pad << "BinaryOp(" << bin->op << ")\n";
        printAST(bin->lhs.get(), indent + 2);
        printAST(bin->rhs.get(), indent + 2);
    } else if (const auto* call = dynamic_cast<const FuncCallNode*>(node)) {
        std::cout << pad << "FuncCall(" << call->name << ")\n";
        for (const auto& arg : call->args) {
            printAST(arg.get(), indent + 2);
        }
    } else {
        std::cout << pad << "<null>\n";
    }
}

inline void printAST(const Program& program) {
    std::cout << "Program\n";
    for (const auto& stmt : program.stmts) {
        std::cout << "  Let(" << stmt.lhs << ", tile=" << stmt.tile_hint << ")\n";
        printAST(stmt.rhs.get(), 4);
    }
}

} // namespace tensorjit
