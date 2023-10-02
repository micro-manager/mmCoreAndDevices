
AUTOMAKE_OPTIONS = subdir-objects
AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) -I$(VIMBA_X_HOME)/api/include
deviceadapter_LTLIBRARIES = libmmgr_dal_AlliedVisionCamera.la
libmmgr_dal_AlliedVisionCamera_la_SOURCES = AlliedVisionHub.h AlliedVisionHub.cpp AlliedVisionDeviceBase.h AlliedVisionDeviceBase.cpp AlliedVisionCamera.h AlliedVisionCamera.cpp Loader/Constants.h Loader/LibLoader.h Loader/LibLoader.cpp $(VIMBA_X_HOME)/api/include/VmbC/VmbC.h $(VIMBA_X_HOME)/api/include/VmbC/VmbCommonTypes.h $(VIMBA_X_HOME)/api/include/VmbC/VmbConstants.h $(VIMBA_X_HOME)/api/include/VmbC/VmbCTypeDefinitions.h $(VIMBA_X_HOME)/api/include/VmbImageTransform/VmbTransform.h $(VIMBA_X_HOME)/api/include/VmbImageTransform/VmbTransformTypes.h
libmmgr_dal_AlliedVisionCamera_la_LIBADD = $(MMDEVAPI_LIBADD)
libmmgr_dal_AlliedVisionCamera_la_LDFLAGS = $(MMDEVAPI_LDFLAGS)

EXTRA_DIST = AlliedVisionCamera.vcproj AlliedVisionCamera.vcproj.filters AlliedVisionCamera.vcproj.user
