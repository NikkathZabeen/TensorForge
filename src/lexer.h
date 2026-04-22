#pragma once

#include <string>
#include <vector>

namespace tensorjit {

enum class TokenType {
    LET,
    IDENT,
    LPAREN,
    RPAREN,
    COMMA,
    PLUS,
    STAR,
    MINUS,
    EQUALS,
    NUMBER,
    AT,
    TILE,
    END
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int col;
};

const char* tokenTypeName(TokenType type);

class Lexer {
public:
    explicit Lexer(std::string source);

    std::vector<Token> tokenize();

private:
    char advance();
    char peek() const;
    char peekNext() const;
    bool isAtEnd() const;
    void skipWhitespace();
    void addToken(TokenType type, std::string lexeme, int line, int col);
    void lexIdentifier(int line, int col);
    void lexNumber(int line, int col);

    std::string source_;
    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
    int line_ = 1;
    int col_ = 1;
};

} // namespace tensorjit
