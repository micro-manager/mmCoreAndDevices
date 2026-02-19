
AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) $(BOOST_CPPFLAGS) -std=c++17
deviceadapter_LTLIBRARIES = libmmgr_dal_EvidentIX85.la
libmmgr_dal_EvidentIX85_la_SOURCES = EvidentHub.cpp EvidentHub.h \
                                      EvidentIX85.cpp EvidentIX85.h \
                                      EvidentModel.cpp EvidentModel.h \
                                      EvidentProtocol.h \
                                      ../../MMDevice/MMDevice.h
libmmgr_dal_EvidentIX85_la_LDFLAGS = $(MMDEVAPI_LDFLAGS)
libmmgr_dal_EvidentIX85_la_LIBADD = $(MMDEVAPI_LIBADD)

EXTRA_DIST = EvidentIX85.vcxproj
