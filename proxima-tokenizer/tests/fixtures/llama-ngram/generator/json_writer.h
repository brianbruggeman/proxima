#pragma once

#include <cstdio>
#include <string>
#include <vector>

// minimal JSON writer: this fixture generator has no dependency on a JSON
// library, all shapes written here are flat objects/arrays of ints/strings.
class json_writer {
public:
    explicit json_writer(FILE * file) : file(file) {}

    void raw(const std::string & text) { std::fputs(text.c_str(), file); }

    static std::string escape(const std::string & input) {
        std::string out;
        out.reserve(input.size() + 8);
        for (char character : input) {
            if (character == '"' || character == '\\') {
                out.push_back('\\');
            }
            out.push_back(character);
        }
        return out;
    }

    static std::string int_array(const std::vector<int32_t> & values) {
        std::string out = "[";
        for (size_t index = 0; index < values.size(); ++index) {
            if (index > 0) {
                out += ",";
            }
            out += std::to_string(values[index]);
        }
        out += "]";
        return out;
    }

    static std::string string_array(const std::vector<std::string> & values) {
        std::string out = "[";
        for (size_t index = 0; index < values.size(); ++index) {
            if (index > 0) {
                out += ",";
            }
            out += "\"" + escape(values[index]) + "\"";
        }
        out += "]";
        return out;
    }

private:
    FILE * file;
};
