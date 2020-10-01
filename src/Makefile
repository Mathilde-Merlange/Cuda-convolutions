CXX=g++
CUDA=nvcc
CXXFLAGS=-O3 -march=native
LDLIBSCV=`pkg-config --libs opencv`

trio1: trio1.cpp
	$(CXX) -o $@ $< $(LDLIBSCV)

trio1-opti: trio1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBSCV)

trio1-cu: trio1.cu
	$(CUDA) -o $@ $< $(LDLIBSCV)
	
trio1-cu-fusion: trio1-fusion.cu
	$(CUDA) -o $@ $< $(LDLIBSCV)
	
graysobel: graysobel.cpp
	$(CXX) -o $@ $< $(LDLIBSCV)

graysobel-opti: graysobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBSCV)

graysobel-cu: graysobel.cu
	$(CUDA) -o $@ $< $(LDLIBSCV)

graysobel-fusion: graysobel-fusion.cu
	$(CUDA) -o $@ $< $(LDLIBSCV)
	
graysobel-shared: graysobel-shared.cu
	$(CUDA) -o $@ $< $(LDLIBSCV)
	
emboss: emboss.cpp
	$(CXX) -o $@ $< $(LDLIBSCV)

emboss-opti: emboss.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBSCV)

emboss-cu: emboss.cu
	$(CUDA) -o $@ $< $(LDLIBSCV)
