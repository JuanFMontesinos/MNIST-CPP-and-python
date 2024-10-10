# Compiler and flags
CXX = g++
PYTHON = $(shell which python3)
PYBIND11_INCLUDE = $(shell $(PYTHON) -m pybind11 --includes)
CXXFLAGS = -O3 -Wall $(PYBIND11_INCLUDE) -fPIC -Wall -Wextra -std=c++17 -fPIC -I include
PYTHON_EXTENSION_SUFFIX = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

# Target output
TARGET = libmnist$(PYTHON_EXTENSION_SUFFIX)
BUILD_DIR = build

# Source files
SRC = bindings.cpp \
      src/activations.cpp \
      src/cross_entropy.cpp \
      src/dataloader.cpp \
      src/functionals.cpp \
      src/linear.cpp

# Object files
OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC))

# Default target
all: $(TARGET)

# Create output directories if they don't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/src:
	mkdir -p $(BUILD_DIR)/src

# Compile the shared library
$(TARGET): $(OBJ)
	$(CXX) -shared -o $@ $(OBJ)

# Compile object files for root-level source files
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile object files for src/ source files
$(BUILD_DIR)/src/%.o: src/%.cpp | $(BUILD_DIR)/src
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Show Python interpreter and include paths
python:
	@echo $(PYTHON)
	@echo $(PYBIND11_INCLUDE)
	@echo $(PYTHON_EXTENSION_SUFFIX)

# Clean up build files
clean:
	rm -f $(BUILD_DIR)/src/*.o $(BUILD_DIR)/*.o $(TARGET)
	rm -rf $(BUILD_DIR)

	@if [ -d results/ ]; then rm -rf results/; fi
	@mkdir results/

download_mnist:
	@if [ -d data/ ]; then rm -rf data/; fi
	@mkdir data/

	@wget -P data/ https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
	@wget -P data/ https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
	@wget -P data/ https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
	@wget -P data/ https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz

	@gunzip data/*.gz
