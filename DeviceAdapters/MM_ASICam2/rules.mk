ver = debug
platform = x86
VERSION = 2.0.0
customer = n
sdkname = ASICamera2
dy = st
FakeUSB = n


CC = i686-pc-linux-gnu-g++
AR= ar

sub:=libusb
APPLE = 0

ifeq ($(platform), mac32)
APPLE = 1
endif

ifeq ($(platform), mac64)
APPLE = 1
endif

ifeq ($(platform), mac)
APPLE = 1
endif

ifeq ($(APPLE), 1)
a=$(shell otool -L ../../linux/lib/$(platform)/libASICamera2.dylib)
else
a=$(shell ldd ../../linux/lib/x86/libASICamera2.so)
endif


$(warning $(sub)) 
$(warning $(a))
b:=$(findstring $(sub),$(a))
$(warning $(b))
#ifeq ($strip $(b)),)
ifeq ($(b),)
$(warning $(sub) is not substring of a)
FakeUSB = y
USB = -framework IOKit -framework foundation
else
$(warning $(sub) is substring of a)
USB = -lusb-1.0 -L../../linux/libusb/$(platform)
endif

ifeq ($(customer), y)
sdkname = Veroptics
endif



ifeq ($(ver), debug)
DEFS = -D_LIN -D_DEBUG
CFLAGS = -fPIC -g $(DEFS) $(USB)
else
DEFS = -D_LIN
CFLAGS = -fPIC -O3 $(DEFS) $(USB)
endif

ifeq ($(customer), y)
CFLAGS += -D_VEROPTICS
endif

ifeq ($(platform), mac32)
CC = g++
CFLAGS += -D_MAC -framework IOKit -framework CoreFoundation
CFLAGS += -m32
endif

ifeq ($(platform), mac64)
CC = g++
CFLAGS += -D_MAC -framework IOKit -framework CoreFoundation
CFLAGS += -m64
endif

ifeq ($(platform), mac)
CC = g++
CFLAGS += -D_MAC  -framework IOKit -framework CoreFoundation
CFLAGS += -arch i386 -arch x86_64
endif

ifeq ($(platform), x86)
CFLAGS += -m32
CFLAGS += -msse 
CFLAGS += -mno-sse2
LDLIB += -lrt
endif

ifeq ($(platform), x64)
CFLAGS += -m64
CFLAGS += -msse 
CFLAGS += -mno-sse2
LDLIB += -lrt
endif

