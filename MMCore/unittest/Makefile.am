check_PROGRAMS = \
	APIError-Tests \
	CoreSanity-Tests \
	LoggingSplitEntryIntoLines-Tests \
	Logger-Tests
AM_DEFAULT_SOURCE_EXT = .cpp
AM_CPPFLAGS = $(GMOCK_CPPFLAGS) -I..
LDADD = ../../../testing/libgmock.la ../libMMCore.la
TESTS = $(check_PROGRAMS)
