
m4_define([MM_CPP_DIR], [.])  # Define a macro substitution to be used in this file.
AC_PREREQ([2.64])
AC_INIT([Micro-Manager], [1.4], [info@micro-manager.org])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR(MM_CPP_DIR[/MMCore/MMCore.cpp])
AC_CANONICAL_BUILD
AC_CANONICAL_HOST
AM_INIT_AUTOMAKE([foreign 1.11])
LT_PREREQ([2.2.6])
LT_INIT([disable-static])
AC_PROG_MKDIR_P

AC_PROG_CXX([clang++ llvm-g++ g++ c++])
AC_PROG_CC([clang llvm-gcc gcc cc])


# Testing (googletest, googlemock)
# See testing/setuptesting.sh
MM_GMOCK([$srcdir/testing/googletest], [\$(top_srcdir)/testing/googletest])
AM_CONDITIONAL([BUILD_CPP_TESTS], [test "x$have_gmock" = xyes])


# Boost
# TODO Reflect results in configuration
AX_BOOST_BASE([1.48.0])
AX_BOOST_DATE_TIME
AX_BOOST_FILESYSTEM
AX_BOOST_SYSTEM
AX_BOOST_THREAD


# AppleHost dependencies
case $host in
   *apple-darwin*) MMCORE_APPLEHOST_LDFLAGS="-framework CoreFoundation -framework IOKit" ;;
   *) MMCORE_APPLEHOST_LDFLAGS="" ;;
esac
AC_SUBST([MMCORE_APPLEHOST_LDFLAGS])


# TODO Make conditional
can_build_mmcore=yes


##
## MMCoreJ
##

# Define a shell variable based on the earlier m4_define
[MM_CPP_DIR]=MM_CPP_DIR
AC_ARG_VAR([MM_CPP_DIR], [The directory containing the c++ code])

# SWIG
AC_ARG_VAR([SWIG], [Simple Wrapper and Interface Generator])
AC_CHECK_PROGS([SWIG], [swig])
# TODO Provide SWIGFLAGS as precious variable




##
## Leftover stuff (needs cleanup)
##

AC_HEADER_STDC
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_CHECK_FUNCS([memset])
AC_CHECK_LIB(dl, dlopen)


# Install Device Adapter API library and headers
install_mmdevapi=false
# TODO reinstate this flag in some better form
#AC_ARG_ENABLE(inst-devapi,
#    [  --enable-inst-devapi    Install the Device Adapter API library and headers ],
#    [ install_mmdevapi=true ])
AM_CONDITIONAL([INSTALL_MMDEVAPI], [test x$install_mmdevapi = xtrue])


##
## Subdirectory configuration
##

AC_MSG_CHECKING([for proprietary device adapter source])
AS_IF([test -f "$srcdir/$MM_CPP_DIR/SecretDeviceAdapters/configure"],
   [build_secretdeviceadapters=yes], [build_secretdeviceadapters=no])
AM_CONDITIONAL([BUILD_SECRETDEVICEADAPTERS],
   [test "x$build_secretdeviceadapters" = xyes])

AC_CONFIG_SUBDIRS(MM_CPP_DIR[/DeviceAdapters])
AS_IF([test "x$build_secretdeviceadapters" = xyes], [
   AC_CONFIG_SUBDIRS(MM_CPP_DIR[/SecretDeviceAdapters])
])


##
## Output generation
##

AC_CONFIG_FILES(m4_strip([
   Makefile
   ]MM_CPP_DIR[/MMDevice/Makefile
   ]MM_CPP_DIR[/MMDevice/unittest/Makefile
   ]MM_CPP_DIR[/MMCore/Makefile
   ]MM_CPP_DIR[/MMCore/unittest/Makefile
]))

AC_OUTPUT

echo ""
echo "m4_text_box([Micro-Manager configuration])"
echo ""

echo "   Build device adapters:                  yes"
echo "   Build closed-source device adapters:    $build_secretdeviceadapters"
echo ""

echo "   Installation prefix:                    $prefix"

echo "Type 'make' to build."
echo "Then type 'make install' to install."
echo ""
