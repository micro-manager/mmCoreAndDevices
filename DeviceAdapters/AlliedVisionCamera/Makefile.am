
AUTOMAKE_OPTIONS = subdir-objects
AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) -ISDK
deviceadapter_LTLIBRARIES = libmmgr_dal_AlliedVisionCamera.la
libmmgr_dal_AlliedVisionCamera_la_SOURCES = AlliedVisionHub.h AlliedVisionHub.cpp AlliedVisionDeviceBase.h AlliedVisionDeviceBase.cpp AlliedVisionCamera.h AlliedVisionCamera.cpp SDK/Loader/Constants.h SDK/Loader/LibLoader.h SDK/Loader/LibLoader.cpp SDK/VmbC/VmbC.h SDK/VmbC/VmbCommonTypes.h SDK/VmbC/VmbConstants.h SDK/Vmbc/VmbCTypeDefinitions.h SDK/VmbImageTransform/VmbTransform.h SDK/VmbImageTransform/VmbTransformTypes.h
libmmgr_dal_AlliedVisionCamera_la_LIBADD = $(MMDEVAPI_LIBADD)
libmmgr_dal_AlliedVisionCamera_la_LDFLAGS = $(MMDEVAPI_LDFLAGS)

EXTRA_DIST = AlliedVisionCamera.vcproj AlliedVisionCamera.vcproj.filters AlliedVisionCamera.vcproj.user
