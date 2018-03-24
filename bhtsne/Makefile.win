CXX = cl.exe
CFLAGS = /nologo /O2 /EHsc /D "_CRT_SECURE_NO_DEPRECATE" /D "USEOMP" /openmp

TARGET = windows

all: $(TARGET) $(TARGET)\bh_tsne.exe

$(TARGET)\bh_tsne.exe: tsne_main.obj tsne.obj sptree.obj
	$(CXX) $(CFLAGS) tsne_main.obj tsne.obj sptree.obj -Fe$(TARGET)\bh_tsne.exe

sptree.obj: sptree.cpp sptree.h
	$(CXX) $(CFLAGS) -c sptree.cpp

tsne.obj: tsne.cpp tsne.h sptree.h vptree.h
	$(CXX) $(CFLAGS) -c tsne.cpp

tsne_main.obj: tsne_main.cpp tsne.h sptree.h vptree.h
	$(CXX) $(CFLAGS) -c tsne_main.cpp

.PHONY: $(TARGET)
$(TARGET):
	-mkdir $(TARGET)

clean:
	-erase /Q *.obj *.exe $(TARGET)\. 
	-rd $(TARGET)
