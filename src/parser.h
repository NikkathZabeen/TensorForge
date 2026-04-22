#pragma once

#include "ast.h"
#include "lexer.h"

#include <memory>
#include <vector>

namespace tensorjit {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    Program parse();

private:
    LetStmt parseStmt();
    std::unique_ptr<ASTNode> parseExpr();
    std::unique_ptr<ASTNode> parseAdditive();
    std::unique_ptr<ASTNode> parseMultiplicative();
    std::unique_ptr<ASTNode> parsePrimary();
    std::unique_ptr<ASTNode> parseFuncCall();

    bool match(TokenType type);
    const Token& consume(TokenType type, const std::string& message);
    const Token& peek() const;
    const Token& previous() const;
    bool isAtEnd() const;

    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
};

} // namespace tensorjit
