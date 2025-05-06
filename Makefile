CXX = mpic++
CXXFLAGS = -std=c++17 -O2 -Iinclude -fopenmp
SRC = src/main.cpp src/graph.cpp
OUT = main
LIBS = -lmetis

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC) $(LIBS)

clean:
	rm -f $(OUT) 