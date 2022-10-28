// COPYRIGHT:     University of California, San Francisco, 2014,
//                All Rights reserved
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

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <ostream>
#include <sstream>
#include <string>


namespace mm
{
namespace logging
{
namespace internal
{


inline const char*
LevelString(LogLevel logLevel)
{
   switch (logLevel)
   {
      case LogLevelTrace: return "trc";
      case LogLevelDebug: return "dbg";
      case LogLevelInfo: return "IFO";
      case LogLevelWarning: return "WRN";
      case LogLevelError: return "ERR";
      case LogLevelFatal: return "FTL";
      default: return "???";
   }
}


// A stateful formatter for the metadata prefix and corresponding
// continuation-line prefix. Intended for single-threaded use only.
class MetadataFormatter
{
   // Reuse buffers for efficiency
   std::string buf_;
   std::ostringstream sstrm_;
   size_t openBracketCol_;
   size_t closeBracketCol_;

public:
   MetadataFormatter() : openBracketCol_(0), closeBracketCol_(0) {}

   // Format the line prefix for the first line of an entry
   void FormatLinePrefix(std::ostream& stream, const Metadata& metadata);

   // Format the line prefix for subsequent lines of an entry
   void FormatContinuationPrefix(std::ostream& stream);
};


inline std::string
FormatLocalTime(std::chrono::time_point<std::chrono::system_clock> tp)
{
   using namespace std::chrono;
   auto us = duration_cast<microseconds>(tp.time_since_epoch());
   auto secs = duration_cast<seconds>(us);
   auto whole = duration_cast<microseconds>(secs);
   auto frac = static_cast<int>((us - whole).count());

   // As of C++14/17, it is simpler (and probably faster) to use C functions for
   // date-time formatting

   std::time_t t(secs.count()); // time_t is seconds on platforms we support
   std::tm *ptm;
#ifdef _WIN32 // Windows localtime() is documented thread-safe
   ptm = std::localtime(&t);
#else // POSIX has localtime_r()
   std::tm tmstruct;
   ptm = localtime_r(&t, &tmstruct);
#endif

   // Format as "yyyy-mm-dd hh:mm:ss.uuuuuu" (26 chars)
   const char *timeFmt = "%Y-%m-%dT%H:%M:%S";
   char buf[32];
   std::size_t len = std::strftime(buf, sizeof(buf), timeFmt, ptm);
   std::snprintf(buf + len, sizeof(buf) - len, ".%06d", frac);
   return buf;
}


inline void
MetadataFormatter::FormatLinePrefix(std::ostream& stream,
      const Metadata& metadata)
{
   // Pre-forming string is more efficient than writing bit by bit to stream.

   buf_ = FormatLocalTime(metadata.GetStampData().GetTimestamp());
   buf_ += " tid";
   sstrm_.str(std::string());
   sstrm_ << metadata.GetStampData().GetThreadId();
   buf_ += sstrm_.str();
   buf_ += ' ';

   openBracketCol_ = buf_.size();
   buf_ += '[';

   buf_ += LevelString(metadata.GetEntryData().GetLevel());
   buf_ += ',';
   buf_ += metadata.GetLoggerData().GetComponentLabel();

   closeBracketCol_ = buf_.size();
   buf_ += ']';

   stream << buf_;
}


inline void
MetadataFormatter::FormatContinuationPrefix(std::ostream& stream)
{
   buf_.assign(closeBracketCol_ + 1, ' ');
   buf_[openBracketCol_] = '[';
   buf_[closeBracketCol_] = ']';
   stream << buf_;
}


} // namespace internal
} // namespace logging
} // namespace mm
