
AM_CPPFLAGS = $(ARAVIS_CPPFLAGS)
AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) $(ARAVIS_CFLAGS)
AM_LDFLAGS = $(MMDEVAPI_LDFLAGS) $(ARAVIS_LDFLAGS)

deviceadapter_LTLIBRARIES=libmmgr_dal_AravisCamera.la

libmmgr_dal_AravisCamera_la_SOURCES=AravisCamera.cpp AravisCamera.h
libmmgr_dal_AravisCamera_la_LIBADD=$(MMDEVAPI_LIBADD) $(ARAVIS_LIBS)

