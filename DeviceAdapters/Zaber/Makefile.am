AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) $(ZML_CPPFLAGS)

deviceadapter_LTLIBRARIES = libmmgr_dal_Zaber.la
libmmgr_dal_Zaber_la_SOURCES = \
	WdiAutofocus.cpp \
	WdiAutofocus.h \
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
libmmgr_dal_Zaber_la_LIBADD = $(MMDEVAPI_LIBADD) $(ZML_LIBS)
libmmgr_dal_Zaber_la_LDFLAGS = $(MMDEVAPI_LDFLAGS) $(ZML_LDFLAGS)

dist_deviceadapter_DATA = $(ZML_LIBS_TO_COPY)

EXTRA_DIST = Zaber.vcxproj Zaber.vcxproj.filters license.txt
