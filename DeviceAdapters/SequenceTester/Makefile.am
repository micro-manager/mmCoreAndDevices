AM_CPPFLAGS = $(BOOST_CPPFLAGS) $(MSGPACK_CPPFLAGS)

AM_CXXFLAGS = $(MMDEVAPI_CXXFLAGS) $(MSGPACK_CXXFLAGS)

deviceadapter_LTLIBRARIES = libmmgr_dal_SequenceTester.la

libmmgr_dal_SequenceTester_la_SOURCES = \
					InterDevice.cpp \
					InterDevice.h \
					LoggedSetting.cpp \
					LoggedSetting.h \
					SequenceTester.cpp \
					SequenceTester.h \
					SequenceTesterImpl.h \
					SettingLogger.cpp \
					SettingLogger.h \
					TextImage.cpp \
					TextImage.h \
					TriggerInput.cpp \
					TriggerInput.h

libmmgr_dal_SequenceTester_la_LIBADD = $(MMDEVAPI_LIBADD) \
				       $(MSGPACK_LIBS)
libmmgr_dal_SequenceTester_la_LDFLAGS = $(MMDEVAPI_LDFLAGS) \
					$(MSGPACK_LDFLAGS)
