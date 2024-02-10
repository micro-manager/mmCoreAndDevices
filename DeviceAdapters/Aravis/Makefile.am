
AM_CXXFLAGS=$(MMDEVAPI_CXXFLAGS)

# Linux, use output from from pkg-config (aravis-0.10).
ARAVISCPPFLAGS = -I/usr/local/include/aravis-0.10 -I/usr/include/libxml2 -I/usr/include/libusb-1.0 -I/usr/include/libmount -I/usr/include/blkid -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include
ARAVISLDFLAGS = -Wl,--enable-new-dtags -Wl,-rpath,/usr/local/lib/x86_64-linux-gnu,-L/usr/local/lib/x86_64-linux-gnu
ARAVISLDLIBS = -laravis-0.10 -lgio-2.0 -lgobject-2.0 -lglib-2.0

deviceadapter_LTLIBRARIES=libmmgr_dal_AravisCamera.la
libmmgr_dal_AravisCamera_la_SOURCES=AravisCamera.cpp AravisCamera.h
libmmgr_dal_AravisCamera_la_CPPFLAGS=$(ARAVISCPPFLAGS)
libmmgr_dal_AravisCamera_la_LIBADD=$(MMDEVAPI_LIBADD) $(ARAVISLDLIBS)
libmmgr_dal_AravisCamera_la_LDFLAGS=$(MMDEVAPI_LDFLAGS) $(ARAVISLDLIBS) $(ARAVISLDFLAGS)

