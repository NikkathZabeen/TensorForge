#include "../src/ast.h"
#include "../src/lexer.h"
#include "../src/parser.h"

#include <cassert>
#include <iostream>

int main() {
    tensorjit::Lexer lexer("let C = matmul(A, B)");
    tensorjit::Parser parser(lexer.tokenize());
    auto program = parser.parse();

    assert(program.stmts.size() == 1);
    assert(program.stmts[0].lhs == "C");
    assert(program.stmts[0].tile_hint == -1);

    auto* call = dynamic_cast<tensorjit::FuncCallNode*>(program.stmts[0].rhs.get());
    assert(call != nullptr);
    assert(call->name == "matmul");
    assert(call->args.size() == 2);
    assert(dynamic_cast<tensorjit::VarNode*>(call->args[0].get()) != nullptr);
    assert(dynamic_cast<tensorjit::VarNode*>(call->args[1].get()) != nullptr);

    tensorjit::Lexer tiledLexer("@tile(64) let C = matmul(A, B)");
    tensorjit::Parser tiledParser(tiledLexer.tokenize());
    auto tiled = tiledParser.parse();
    assert(tiled.stmts[0].tile_hint == 64);

    std::cout << "test_parser passed\n";
    return 0;
}
