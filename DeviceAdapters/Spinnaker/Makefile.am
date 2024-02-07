
AM_CXXFLAGS=$(MMDEVAPI_CXXFLAGS)

# Linux default install location?
SPINNAKER_ROOT=/opt/spinnaker

SPINNAKERCPPFLAGS = -I$(SPINNAKER_ROOT)/include
SPINNAKERLDFLAGS = -Wl,--enable-new-dtags -Wl,-rpath,$(SPINNAKER_ROOT)/lib,-L$(SPINNAKER_ROOT)/lib
SPINNAKERLDLIBS =-lSpinnaker

deviceadapter_LTLIBRARIES=libmmgr_dal_SpinnakerCamera.la
libmmgr_dal_SpinnakerCamera_la_SOURCES=SpinnakerCamera.cpp SpinnakerCamera.h
libmmgr_dal_SpinnakerCamera_la_CPPFLAGS=$(SPINNAKERCPPFLAGS) 
libmmgr_dal_SpinnakerCamera_la_LIBADD=$(MMDEVAPI_LIBADD) $(SPINNAKERLDLIBS)
libmmgr_dal_SpinnakerCamera_la_LDFLAGS=$(MMDEVAPI_LDFLAGS) $(SPINNAKERLDLIBS) $(SPINNAKERLDFLAGS)

