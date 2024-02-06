AUTOMAKE_OPTIONS = subdir-objects
AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) $(VIMBA_X_CPPFLAGS)
deviceadapter_LTLIBRARIES = libmmgr_dal_AlliedVisionCamera.la

libmmgr_dal_AlliedVisionCamera_la_SOURCES = \
					    AlliedVisionCamera.cpp \
					    AlliedVisionCamera.h \
					    AlliedVisionDeviceBase.cpp \
					    AlliedVisionDeviceBase.h \
					    AlliedVisionHub.cpp \
					    AlliedVisionHub.h \
					    Loader/Constants.h \
					    Loader/LibLoader.cpp \
					    Loader/LibLoader.h

libmmgr_dal_AlliedVisionCamera_la_LIBADD = $(MMDEVAPI_LIBADD)
libmmgr_dal_AlliedVisionCamera_la_LDFLAGS = $(MMDEVAPI_LDFLAGS)
