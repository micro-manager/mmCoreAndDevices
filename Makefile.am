AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4


# if BUILD_MMCORE
# MMCORE_DIR = MMCore
# endif

if BUILD_SECRETDEVICEADAPTERS
SECRETDEVICEADAPTERS = SecretDeviceAdapters
endif


# CLEANFILES = $(launcher_SCRIPTS)


# TODO: Building of DeviceAdapters could be made optional.
SUBDIRS = \
	MMDevice \
	DeviceAdapters \
	$(SECRETDEVICEADAPTERS)

dox:
	-rm -rf doxygen/out
	$(MKDIR_P) doxygen/out/MMDevice
	$(MKDIR_P) doxygen/out/MMCore
	doxygen doxygen/MMDevice
	doxygen doxygen/MMCore
