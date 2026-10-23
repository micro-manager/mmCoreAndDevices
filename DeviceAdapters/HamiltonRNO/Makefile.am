AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS)
deviceadapter_LTLIBRARIES = libmmgr_dal_HamiltonRNO.la
libmmgr_dal_HamiltonRNO_la_SOURCES = HamiltonRNO.cpp \
				     HamiltonCommands.h \
				     HamiltonValves.h \
				     HamiltonSyringes.h
libmmgr_dal_HamiltonRNO_la_LIBADD = $(MMDEVAPI_LIBADD)
libmmgr_dal_HamiltonRNO_la_LDFLAGS = $(MMDEVAPI_LDFLAGS)

if BUILD_CPP_TESTS
UNITTESTS = unittest
endif

SUBDIRS = . $(UNITTESTS)
