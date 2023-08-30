AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) -I$(MMTHIRDPARTYPUBLIC)/Zaber/zaber-motion/include

deviceadapter_LTLIBRARIES = libmmgr_dal_Zaber.la
libmmgr_dal_Zaber_la_SOURCES = \
	ObjectiveChanger.cpp \
	ObjectiveChanger.h \
	Illuminator.cpp \
	Illuminator.h \
	FilterCubeTurret.cpp \
	FilterCubeTurret.h \
	FilterWheel.cpp \
	FilterWheel.h \
	XYStage.cpp \
	XYStage.h \
	Zaber.cpp \
	Zaber.h \
	ConnectionManager.cpp \
	ConnectionManager.h \
	Stage.cpp \
	Stage.h
libmmgr_dal_Zaber_la_LIBADD = $(MMDEVAPI_LIBADD)

if HOST_LINUX
ZML_DIR_LIB = $(MMTHIRDPARTYPUBLIC)/Zaber/zaber-motion/linux-amd64/lib
RPATH_FLAGS = -Wl,-rpath -Wl,\$$ORIGIN
dist_deviceadapter_DATA = \
	$(ZML_DIR_LIB)/libzml.so.3.4 \
	$(ZML_DIR_LIB)/libzaber-motion-lib-linux-amd64.so.3.4.3
endif

if HOST_OSX
ZML_DIR_LIB = $(MMTHIRDPARTYPUBLIC)/Zaber/zaber-motion/darwin/lib
RPATH_FLAGS = -Wl,-rpath -Wl,@loader_path
dist_deviceadapter_DATA = \
	$(ZML_DIR_LIB)/libzaber-motion-lib-darwin-uni.3.4.3.dylib \
	$(ZML_DIR_LIB)/libzml.3.4.dylib
endif

libmmgr_dal_Zaber_la_LDFLAGS = $(MMDEVAPI_LDFLAGS) -L$(ZML_DIR_LIB) -lzml $(RPATH_FLAGS)

EXTRA_DIST = Zaber.vcxproj Zaber.vcxproj.filters license.txt
