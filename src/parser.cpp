#include "parser.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace tensorjit {

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

Program Parser::parse() {
    Program program;
    while (!isAtEnd()) {
        program.stmts.push_back(parseStmt());
    }
    return program;
}

LetStmt Parser::parseStmt() {
    int tileHint = -1;
    if (match(TokenType::AT)) {
        consume(TokenType::TILE, "Expected 'tile'");
        consume(TokenType::LPAREN, "Expected '('");
        const Token& tile = consume(TokenType::NUMBER, "Expected tile size");
        tileHint = std::atoi(tile.lexeme.c_str());
        consume(TokenType::RPAREN, "Expected ')'");
    }

    consume(TokenType::LET, "Expected 'let'");
    const Token& lhs = consume(TokenType::IDENT, "Expected identifier");
    consume(TokenType::EQUALS, "Expected '='");
    auto rhs = parseExpr();

    return LetStmt{lhs.lexeme, std::move(rhs), tileHint};
}

std::unique_ptr<ASTNode> Parser::parseExpr() {
    return parseAdditive();
}

std::unique_ptr<ASTNode> Parser::parseAdditive() {
    auto expr = parseMultiplicative();
    while (match(TokenType::PLUS) || match(TokenType::MINUS)) {
        const char op = previous().lexeme[0];
        auto rhs = parseMultiplicative();
        expr = std::make_unique<BinaryOp>(op, std::move(expr), std::move(rhs));
    }
    return expr;
}

std::unique_ptr<ASTNode> Parser::parseMultiplicative() {
    auto expr = parsePrimary();
    while (match(TokenType::STAR)) {
        const char op = previous().lexeme[0];
        auto rhs = parsePrimary();
        expr = std::make_unique<BinaryOp>(op, std::move(expr), std::move(rhs));
    }
    return expr;
}

std::unique_ptr<ASTNode> Parser::parsePrimary() {
    if (match(TokenType::IDENT)) {
        if (peek().type == TokenType::LPAREN) {
            --pos_;
            return parseFuncCall();
        }
        return std::make_unique<VarNode>(previous().lexeme);
    }

    if (match(TokenType::NUMBER)) {
        return std::make_unique<NumberNode>(std::stod(previous().lexeme));
    }

    if (match(TokenType::LPAREN)) {
        auto expr = parseExpr();
        consume(TokenType::RPAREN, "Expected ')'");
        return expr;
    }

    throw std::runtime_error("Expected expression at line " + std::to_string(peek().line) + ":" +
                             std::to_string(peek().col));
}

std::unique_ptr<ASTNode> Parser::parseFuncCall() {
    const Token& name = consume(TokenType::IDENT, "Expected function name");
    consume(TokenType::LPAREN, "Expected '('");

    std::vector<std::unique_ptr<ASTNode>> args;
    if (peek().type != TokenType::RPAREN) {
        do {
            args.push_back(parseExpr());
        } while (match(TokenType::COMMA));
    }

    consume(TokenType::RPAREN, "Expected ')'");
    return std::make_unique<FuncCallNode>(name.lexeme, std::move(args));
}

bool Parser::match(TokenType type) {
    if (peek().type != type) {
        return false;
    }
    ++pos_;
    return true;
}

const Token& Parser::consume(TokenType type, const std::string& message) {
    if (peek().type == type) {
        ++pos_;
        return previous();
    }

    throw std::runtime_error(message + " at line " + std::to_string(peek().line) + ":" +
                             std::to_string(peek().col));
}

const Token& Parser::peek() const {
    return tokens_[pos_];
}

const Token& Parser::previous() const {
    return tokens_[pos_ - 1];
}

bool Parser::isAtEnd() const {
    return peek().type == TokenType::END;
}

} // namespace tensorjit
