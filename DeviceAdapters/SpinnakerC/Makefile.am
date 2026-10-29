
AM_CXXFLAGS=$(MMDEVAPI_CXXFLAGS)

# Linux default install location
SPINNAKER_ROOT=/opt/spinnaker

SPINNAKERCPPFLAGS = -I$(SPINNAKER_ROOT)/include/spinc
SPINNAKERLDFLAGS = -L$(SPINNAKER_ROOT)/lib -Wl,--enable-new-dtags -Wl,-rpath,$(SPINNAKER_ROOT)/lib
SPINNAKERLDLIBS = -lSpinnaker_C

deviceadapter_LTLIBRARIES=libmmgr_dal_SpinnakerC.la
libmmgr_dal_SpinnakerC_la_SOURCES=SpinnakerCCamera.cpp SpinnakerCCamera.h
libmmgr_dal_SpinnakerC_la_CPPFLAGS=$(SPINNAKERCPPFLAGS)
libmmgr_dal_SpinnakerC_la_LIBADD=$(MMDEVAPI_LIBADD) $(SPINNAKERLDLIBS)
libmmgr_dal_SpinnakerC_la_LDFLAGS=$(MMDEVAPI_LDFLAGS) $(SPINNAKERLDFLAGS)

EXTRA_DIST = SpinnakerC.vcxproj SpinnakerC.vcxproj.filters LICENSE
