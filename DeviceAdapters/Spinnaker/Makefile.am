
SPINNAKERLDFLAGS =
SPINLIBADD =
SPINLDFLAGS =
SPINNAKERCPPFLAGS =
AM_CXXFLAGS=$(MMDEVAPI_CXXFLAGS) 
libmmgr_dal_SpinnakerCamera_la_SOURCES=SpinnakerCamera.cpp SpinnakerCamera.h
deviceadapter_LTLIBRARIES=libmmgr_dal_SpinnakerCamera.la


if BUILD_SPINNAKER_LINUX
   SPINNAKER_ROOT=/opt/spinnaker

   SPINNAKERLDFLAGS +=-Wl,--enable-new-dtags -Wl,-rpath,$(SPINNAKER_ROOT)/lib,-L$(SPINNAKER_ROOT)/lib

   SPINNAKERCPPFLAGS +=-I$(SPINNAKER_ROOT)/include
   SPINNAKERLDLIBS =-lSpinnaker

   SPINLIBADD+=$(MMDEVAPI_LIBADD) $(SPINNAKERLDLIBS)
   SPINLDFLAGS+=$(MMDEVAPI_LDFLAGS) $(SPINNAKERLDLIBS) $(SPINNAKERLDFLAGS)
endif


libmmgr_dal_SpinnakerCamera_la_CPPFLAGS=$(SPINNAKERCPPFLAGS) 
libmmgr_dal_SpinnakerCamera_la_LIBADD=$(SPINLIBADD)  
libmmgr_dal_SpinnakerCamera_la_LDFLAGS=$(SPINLDFLAGS)

EXTRA_DIST = 
