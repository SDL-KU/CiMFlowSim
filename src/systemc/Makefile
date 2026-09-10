# SystemC Pipeline Simulator Makefile

# ===============================
# Build Configuration
# ===============================

# Build type: debug, release, or profile
BUILD_TYPE ?= release

# SystemC installation path (environment variable with fallback)
SYSTEMC_HOME ?= /usr/local/systemc-2.3.3

# Detect SystemC library directory
ifeq ($(wildcard $(SYSTEMC_HOME)/lib64),)
    ifeq ($(wildcard $(SYSTEMC_HOME)/lib-linux64),)
        SYSTEMC_LIB_DIR = $(SYSTEMC_HOME)/lib
    else
        SYSTEMC_LIB_DIR = $(SYSTEMC_HOME)/lib-linux64
    endif
else
    SYSTEMC_LIB_DIR = $(SYSTEMC_HOME)/lib64
endif

# Parallel build support (use -j flag: make -j$(nproc))
MAKEFLAGS += --jobs=1

# Compiler settings
CXX = g++
CXXFLAGS_COMMON = -std=c++17 -Wall -Wextra -MMD -MP -DSC_DISABLE_API_VERSION_CHECK
INCLUDES = -I. -I$(SYSTEMC_HOME)/include
LDFLAGS = -L$(SYSTEMC_LIB_DIR)
LIBS = -lsystemc -lm -lstdc++fs

# Build type specific flags
ifeq ($(BUILD_TYPE),debug)
    CXXFLAGS = $(CXXFLAGS_COMMON) -g -O0 -DDEBUG -fno-omit-frame-pointer
    BUILD_DIR = build/debug
    BUILD_SUFFIX = _debug
else ifeq ($(BUILD_TYPE),profile)
    CXXFLAGS = $(CXXFLAGS_COMMON) -O2 -g -pg -DPROFILE
    BUILD_DIR = build/profile
    BUILD_SUFFIX = _profile
else
    # Use x86-64 for cross-node compatibility (AMD + Intel cluster)
    CXXFLAGS = $(CXXFLAGS_COMMON) -O3 -march=x86-64 -mtune=generic -flto -DNDEBUG
    LDFLAGS += -flto
    BUILD_DIR = build/release
    BUILD_SUFFIX =
endif

# Target executables
TARGET = pipeline_sim

# Output file names (from constants.h)
GANTT_DATA_FILE = gantt_data.txt
DEBUG_STRATEGY_FILE = debug_strategy.txt
EXECUTION_TRACE_FILE = execution_trace.log

# Source files
SOURCES = main.cpp
STRATEGY_SOURCES = cnn_strategy.cpp tiling_strategy.cpp
MEMORY_SOURCES =
CONFIG_SOURCES = config/config_loader.cpp
OUTPUT_SOURCES = output/gantt_generator.cpp
ALL_SOURCES = $(SOURCES) $(STRATEGY_SOURCES) $(CONFIG_SOURCES) $(OUTPUT_SOURCES)

# Object files
OBJECTS = $(ALL_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
DEPS = $(OBJECTS:.o=.d)

# Headers
HEADERS = pipeline_simulator.h cnn_strategy.h memory/tensor_memory_system.h memory/memory_types.h config/config_loader.h config/constants.h config/exceptions.h config/types.h output/gantt_generator.h

# Create build directories
$(shell mkdir -p $(BUILD_DIR) $(BUILD_DIR)/config $(BUILD_DIR)/output)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@

# Object file compilation with dependency tracking
$(BUILD_DIR)/%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Include dependency files
-include $(DEPS)

# Visualization target
visualize: $(TARGET)
	export LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib-linux64:$$LD_LIBRARY_PATH && ./$(TARGET) && python3 ../python/visualization/visualize_pipeline.py

# Clean up
clean:
	rm -f $(TARGET)
	rm -f $(GANTT_DATA_FILE) *.png *.gch
	rm -f *.log *.txt  # Remove log files
	rm -f *.json       # Remove simulation output JSON files
	rm -rf build/      # Remove all build directories
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean only temporary files (keep executables and object files)
clean-temp:
	rm -f *.log *.txt $(GANTT_DATA_FILE) *.png
	@echo "Cleaned temporary files (logs, debug outputs, charts)"

# Clean only build artifacts (keep temporary files)
clean-build:
	rm -f $(TARGET)
	rm -rf build/
	@echo "Cleaned build artifacts"


# Run simulation
run: $(TARGET)
	export LD_LIBRARY_PATH=$(SYSTEMC_LIB_DIR):$$LD_LIBRARY_PATH && ./$(TARGET)


# For systems without SystemC installation, create a simple test
test:
	@echo "Testing SystemC availability..."
	@echo "#include <systemc.h>" > test_systemc.cpp
	@echo "int sc_main(int argc, char* argv[]) { return 0; }" >> test_systemc.cpp
	@if $(CXX) $(INCLUDES) -o test_systemc test_systemc.cpp $(LDFLAGS) $(LIBS) 2>/dev/null; then \
		echo "SystemC found - you can use 'make run'"; \
		rm -f test_systemc test_systemc.cpp; \
	else \
		echo "SystemC not found. Please install SystemC or adjust SYSTEMC_HOME path"; \
		rm -f test_systemc test_systemc.cpp; \
	fi

# Help target
help:
	@echo "CiMFlowSim SystemC Simulator Build System"
	@echo "=========================================="
	@echo ""
	@echo "Build Commands:"
	@echo "  all          - Build main pipeline simulator (default)"
	@echo "  clean        - Remove all build artifacts and temporary files"
	@echo "  clean-build  - Remove only build artifacts (keep logs/temp files)"
	@echo "  clean-temp   - Remove only temporary files (keep binaries)"
	@echo ""
	@echo "Build Types:"
	@echo "  make BUILD_TYPE=debug    - Debug build with symbols and no optimization"
	@echo "  make BUILD_TYPE=release  - Release build with optimization (default)"
	@echo "  make BUILD_TYPE=profile  - Profile build with gprof support"
	@echo ""
	@echo "Run Commands:"
	@echo "  run          - Run the main simulation"
	@echo "  test         - Test SystemC installation"
	@echo "  visualize    - Run simulation and generate visualizations"
	@echo ""
	@echo "Test Commands:"
	@echo ""
	@echo "Parallel Build:"
	@echo "  make -j\$$(nproc)  - Build using all available CPU cores"
	@echo ""
	@echo "Environment Variables:"
	@echo "  SYSTEMC_HOME - Path to SystemC installation (default: /usr/local/systemc-2.3.3)"
	@echo "  BUILD_TYPE   - Build configuration (debug, release, profile)"

.PHONY: all clean clean-build clean-temp run test visualize help
