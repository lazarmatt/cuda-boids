NVCC     := nvcc
CC       := gcc
CXXFLAGS := -std=c++20
LDFLAGS  := -lGL -lglfw
IFLAGS   := -I./third_party/glad/include

SRC_DIR  := src
OBJ_DIR  := build
TARGET   := boids

SRCS     := $(wildcard $(SRC_DIR)/*.cu)
OBJS     := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))
DEPS     := $(OBJS:.o=.d)

.PHONY: all clean

all: $(TARGET)

$(OBJ_DIR)/glad.o: third_party/glad/src/glad.c | $(OBJ_DIR)
	$(CC) $(IFLAGS) -std=c11 -c $< -o $@

$(TARGET): $(OBJS) $(OBJ_DIR)/glad.o
	$(NVCC) $(IFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(IFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

run: $(TARGET)
	__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./$(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

-include $(DEPS)