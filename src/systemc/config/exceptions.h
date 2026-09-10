#ifndef CONFIG_EXCEPTIONS_H
#define CONFIG_EXCEPTIONS_H

#include <stdexcept>
#include <string>

namespace config {

class ConfigurationError : public std::runtime_error {
  public:
    explicit ConfigurationError(const std::string& message)
        : std::runtime_error("Configuration Error: " + message) {}
};

class FileNotFoundError : public ConfigurationError {
  public:
    explicit FileNotFoundError(const std::string& filename)
        : ConfigurationError("Cannot open file: " + filename) {}
};

class JSONParseError : public ConfigurationError {
  public:
    JSONParseError(const std::string& filename, const std::string& details)
        : ConfigurationError("JSON parse error in " + filename + ": " + details) {}
};

class ValidationError : public std::runtime_error {
  public:
    explicit ValidationError(const std::string& message)
        : std::runtime_error("Validation Error: " + message) {}
};

}  // namespace config

#endif  // CONFIG_EXCEPTIONS_H