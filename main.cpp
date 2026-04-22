#include "src/ast.h"
#include "src/codegen.h"
#include "src/jit.h"
#include "src/lexer.h"
#include "src/parser.h"

#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string readFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "examples/basic.tir";
        const std::string source = readFile(path);

        tensorjit::Lexer lexer(source);
        auto tokens = lexer.tokenize();

        tensorjit::Parser parser(tokens);
        auto program = parser.parse();
        tensorjit::printAST(program);

        tensorjit::CodeGen codegen("tensorjit_cli");
        codegen.generateProgram(program);
        if (llvm::verifyModule(codegen.module(), &llvm::errs())) {
            std::cerr << "Module verification failed\n";
            return 1;
        }

        std::cout << "\nGenerated LLVM IR:\n";
        codegen.module().print(llvm::outs(), nullptr);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
