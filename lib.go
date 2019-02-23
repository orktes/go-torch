package torch

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR} -O3 -Wall -g -Wno-sign-compare -Wno-unused-function -I/Library/Developer/CommandLineTools/usr/include/c++/v1 -I/usr/local/include -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
// #cgo LDFLAGS: -lstdc++ -L/opt/libtorch/lib  -ltorch -lcaffe2 -lc10
// #cgo linux,amd64,gpu CXXFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64,gpu LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -L/opt/libtorch/lib -lcaffe2_gpu -lcudart -lnvrtc-builtins -lnvrtc -lnvToolsExt -lcuda
import "C"
