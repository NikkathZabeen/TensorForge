#include "../src/lexer.h"

#include <cassert>
#include <iostream>

int main() {
    tensorjit::Lexer lexer("let C = matmul(A, B)");
    auto tokens = lexer.tokenize();

    assert(tokens.size() == 10);
    assert(tokens[0].type == tensorjit::TokenType::LET);
    assert(tokens[0].lexeme == "let");
    assert(tokens[1].type == tensorjit::TokenType::IDENT);
    assert(tokens[1].lexeme == "C");
    assert(tokens[2].type == tensorjit::TokenType::EQUALS);
    assert(tokens[3].type == tensorjit::TokenType::IDENT);
    assert(tokens[3].lexeme == "matmul");
    assert(tokens[4].type == tensorjit::TokenType::LPAREN);
    assert(tokens[5].lexeme == "A");
    assert(tokens[6].type == tensorjit::TokenType::COMMA);
    assert(tokens[7].lexeme == "B");
    assert(tokens[8].type == tensorjit::TokenType::RPAREN);
    assert(tokens[9].type == tensorjit::TokenType::END);

    std::cout << "test_lexer passed\n";
    return 0;
}
