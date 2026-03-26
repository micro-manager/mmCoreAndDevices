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
//
// AUTHOR:        Mark Tsuchida

#pragma once

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif


namespace mmcore {
namespace internal {
namespace logging {
namespace internal {


namespace {

inline bool
DetectStderrColor()
{
   if (std::getenv("NO_COLOR") != nullptr)
      return false;

   const char* term = std::getenv("TERM");
   if (term != nullptr && std::strcmp(term, "dumb") == 0)
      return false;

#ifdef _WIN32
   if (!_isatty(_fileno(stderr)))
      return false;

   HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
   if (h == INVALID_HANDLE_VALUE)
      return false;

   DWORD mode = 0;
   if (!GetConsoleMode(h, &mode))
      return false;

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
   return (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
   return isatty(STDERR_FILENO) != 0;
#endif
}

} // anonymous namespace


inline bool
ShouldColorStderr()
{
   static const bool result = DetectStderrColor();
   return result;
}


} // namespace internal
} // namespace logging
} // namespace internal
} // namespace mmcore
