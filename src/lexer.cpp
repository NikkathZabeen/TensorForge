#include "lexer.h"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace tensorjit {

Lexer::Lexer(std::string source) : source_(std::move(source)) {}

const char* tokenTypeName(TokenType type) {
    switch (type) {
    case TokenType::LET: return "LET";
    case TokenType::IDENT: return "IDENT";
    case TokenType::LPAREN: return "LPAREN";
    case TokenType::RPAREN: return "RPAREN";
    case TokenType::COMMA: return "COMMA";
    case TokenType::PLUS: return "PLUS";
    case TokenType::STAR: return "STAR";
    case TokenType::MINUS: return "MINUS";
    case TokenType::EQUALS: return "EQUALS";
    case TokenType::NUMBER: return "NUMBER";
    case TokenType::AT: return "AT";
    case TokenType::TILE: return "TILE";
    case TokenType::END: return "END";
    }
    return "UNKNOWN";
}

std::vector<Token> Lexer::tokenize() {
    while (!isAtEnd()) {
        skipWhitespace();
        if (isAtEnd()) {
            break;
        }

        const int startLine = line_;
        const int startCol = col_;
        const char c = advance();

        switch (c) {
        case '(':
            addToken(TokenType::LPAREN, "(", startLine, startCol);
            break;
        case ')':
            addToken(TokenType::RPAREN, ")", startLine, startCol);
            break;
        case ',':
            addToken(TokenType::COMMA, ",", startLine, startCol);
            break;
        case '+':
            addToken(TokenType::PLUS, "+", startLine, startCol);
            break;
        case '*':
            addToken(TokenType::STAR, "*", startLine, startCol);
            break;
        case '-':
            addToken(TokenType::MINUS, "-", startLine, startCol);
            break;
        case '=':
            addToken(TokenType::EQUALS, "=", startLine, startCol);
            break;
        case '@':
            addToken(TokenType::AT, "@", startLine, startCol);
            break;
        default:
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                --pos_;
                --col_;
                lexIdentifier(startLine, startCol);
            } else if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && std::isdigit(static_cast<unsigned char>(peek())))) {
                --pos_;
                --col_;
                lexNumber(startLine, startCol);
            } else {
                throw std::runtime_error("Unknown character '" + std::string(1, c) + "' at line " +
                                         std::to_string(startLine) + ":" + std::to_string(startCol));
            }
        }
    }

    tokens_.push_back({TokenType::END, "", line_, col_});
    return tokens_;
}

char Lexer::advance() {
    const char c = source_[pos_++];
    if (c == '\n') {
        ++line_;
        col_ = 1;
    } else {
        ++col_;
    }
    return c;
}

char Lexer::peek() const {
    if (isAtEnd()) {
        return '\0';
    }
    return source_[pos_];
}

char Lexer::peekNext() const {
    if (pos_ + 1 >= source_.size()) {
        return '\0';
    }
    return source_[pos_ + 1];
}

bool Lexer::isAtEnd() const {
    return pos_ >= source_.size();
}

void Lexer::skipWhitespace() {
    while (!isAtEnd()) {
        const char c = peek();
        if (c == ' ' || c == '\r' || c == '\t' || c == '\n') {
            advance();
        } else {
            break;
        }
    }
}

void Lexer::addToken(TokenType type, std::string lexeme, int line, int col) {
    tokens_.push_back({type, std::move(lexeme), line, col});
}

void Lexer::lexIdentifier(int line, int col) {
    const std::size_t start = pos_;
    while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_') {
        advance();
    }

    std::string text = source_.substr(start, pos_ - start);
    if (text == "let") {
        addToken(TokenType::LET, text, line, col);
    } else if (text == "tile") {
        addToken(TokenType::TILE, text, line, col);
    } else {
        addToken(TokenType::IDENT, text, line, col);
    }
}

void Lexer::lexNumber(int line, int col) {
    const std::size_t start = pos_;
    while (std::isdigit(static_cast<unsigned char>(peek()))) {
        advance();
    }

    if (peek() == '.' && std::isdigit(static_cast<unsigned char>(peekNext()))) {
        advance();
        while (std::isdigit(static_cast<unsigned char>(peek()))) {
            advance();
        }
    }

    addToken(TokenType::NUMBER, source_.substr(start, pos_ - start), line, col);
}

} // namespace tensorjit
