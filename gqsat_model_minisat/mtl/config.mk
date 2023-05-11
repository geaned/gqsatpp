##
##  This file is for system specific configurations. For instance, on
##  some systems the path to zlib needs to be added. Example:
##
##  CFLAGS += -I/usr/local/include
##  LFLAGS += -L/usr/local/lib

CFLAGS += -I../../../../../pytorch/torch/include -I../../../../../pytorch/torch/csrc/api/include
LFLAGS += -std=gnu++14 -D_GLIBCXX_USE_CXX11_ABI=0 -Wl,-R../../../../../pytorch/build_libtorch/build/lib -Wl,--no-as-needed -L../../../../../pytorch/build_libtorch/build/lib -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda