# MacOS compiler
CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++

# Make it the C++ compiler too
CXX = $(CC)

# OSX include paths for Homebrew
CFLAGS = -Wc++11-extensions -std=c++17 -I/opt/homebrew/Cellar/opencv/4.11.0/include/opencv4 -DENABLE_PRECOMPILED_HEADERS=OFF

# Make these the C++ flags too
CXXFLAGS = $(CFLAGS)

# Library paths, update to where your OpenCV and ONNX Runtime libraries are stored
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.11.0/lib
LDFLAGS += -L/opt/homebrew/lib
LDFLAGS += -L/opt/homebrew/opt/protobuf/lib

# OpenCV libraries and MacOS frameworks
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lprotobuf

# Location to write the executables
BINDIR = ../bin

# rules for making various executables
readfiles1: readfiles-1.o csv_util.o similarity_extraction.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /opt/homebrew/lib $(BINDIR)/$@

calculate_similarity: calculate_similarity.o csv_util.o similarity_extraction.o filters.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /opt/homebrew/lib $(BINDIR)/$@

# filters.o: filters.cpp filters.h
# 	$(CC) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *~
