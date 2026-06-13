CXX      = g++
CXXFLAGS = -std=c++17 -O2 $(shell pkg-config --cflags harfbuzz cairo freetype2)
LIBS     = $(shell pkg-config --libs harfbuzz cairo freetype2)

.PHONY: all clean

all: linebreak_cpp textrender

linebreak: linebreak.cpp linebreak.h
	$(CXX) $(CXXFLAGS) linebreak.cpp $(LIBS) -o linebreak

textrender: textrender.cpp linebreak.h
	$(CXX) $(CXXFLAGS) textrender.cpp $(LIBS) -o textrender

clean:
	rm -f linebreak_cpp textrender
