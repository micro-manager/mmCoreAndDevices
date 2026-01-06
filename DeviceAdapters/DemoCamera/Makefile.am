
AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) $(BOOST_CPPFLAGS)
deviceadapter_LTLIBRARIES = libmmgr_dal_DemoCamera.la
libmmgr_dal_DemoCamera_la_SOURCES = \
	DemoCamera.cpp \
	DemoCameraModule.cpp \
	DemoHub.cpp \
	DemoImageGeneration.cpp \
	DemoStateDevices.cpp \
	DemoStages.cpp \
	DemoShutter.cpp \
	DemoDA.cpp \
	DemoMagnifier.cpp \
	DemoAutoFocus.cpp \
	DemoGalvo.cpp \
	DemoImageProcessors.cpp \
	DemoPumps.cpp \
	DemoCamera.h \
	../../MMDevice/MMDevice.h
libmmgr_dal_DemoCamera_la_LDFLAGS = $(MMDEVAPI_LDFLAGS)
libmmgr_dal_DemoCamera_la_LIBADD = $(MMDEVAPI_LIBADD)

EXTRA_DIST = DemoCamera.vcproj license.txt
