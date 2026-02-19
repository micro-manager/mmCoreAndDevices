///////////////////////////////////////////////////////////////////////////////
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Declaration helper macros
//
// COPYRIGHT:     2026 Board of Regents of the University of Wisconsin System
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#pragma once

// At the moment we rely on C++ dynamic exception specifiers (deprecated in
// C++11, removed in C++17) to tell SWIG-Java which exceptions may be thrown by
// API functions. But to avoid warnings and errors from the C++ compiler, we
// hide them behind a macro.
// Also hide 'noexcept' from SWIG 3.x, which doesn't support that keyword.
#ifdef SWIG
#   define MMCORE_LEGACY_THROW(ex) throw (ex)
#   define MMCORE_NOEXCEPT throw ()
#   define MMCORE_DEPRECATED
#else
#   define MMCORE_LEGACY_THROW(ex)
#   define MMCORE_NOEXCEPT noexcept
#   define MMCORE_DEPRECATED [[deprecated]]
#endif
