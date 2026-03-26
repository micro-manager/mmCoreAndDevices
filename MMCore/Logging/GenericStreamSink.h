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

#include "FileRotation.h"
#include "GenericSink.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>


namespace mmcore {
namespace internal {
namespace logging {


class CannotOpenFileException : public std::exception
{
public:
   virtual const char* what() const noexcept { return "Cannot open log file"; }
};


namespace internal {


class CountingStreambuf : public std::streambuf
{
   std::streambuf* wrapped_;
   std::size_t count_;

public:
   explicit CountingStreambuf(std::streambuf* wrapped) :
      wrapped_(wrapped),
      count_(0)
   {}

   std::size_t Count() const { return count_; }
   void ResetCount() { count_ = 0; }

   void SetWrapped(std::streambuf* wrapped) { wrapped_ = wrapped; }

protected:
   int_type overflow(int_type c) override
   {
      if (!traits_type::eq_int_type(c, traits_type::eof()))
      {
         auto result = wrapped_->sputc(traits_type::to_char_type(c));
         if (!traits_type::eq_int_type(result, traits_type::eof()))
            ++count_;
         return result;
      }
      return traits_type::not_eof(c);
   }

   std::streamsize xsputn(const char_type* s, std::streamsize n) override
   {
      auto written = wrapped_->sputn(s, n);
      count_ += static_cast<std::size_t>(written);
      return written;
   }

   int sync() override
   {
      return wrapped_->pubsync();
   }
};


template <class TFormatter, class UMetadata, typename VPacketIter>
void
WritePacketsToStream(std::ostream& stream,
      VPacketIter first, VPacketIter last,
      std::shared_ptr< GenericEntryFilter<UMetadata> > filter)
{
   TFormatter formatter;

   bool beforeFirst = true;
   for (VPacketIter it = first; it != last; ++it)
   {
      // Apply filter if present
      if (filter && !filter->Filter(it->GetMetadataConstRef()))
         continue;

      // If line continuation (broken up just to fit into LinePacket buffer),
      // splice the packets.
      if (it->GetPacketState() == PacketStateLineContinuation)
      {
         stream << it->GetText();
         continue;
      }

      // Close the previous output line.
      if (!beforeFirst)
         stream << '\n';

      // Write metadata on first line of entry; write empty prefix of same
      // width on subsequent lines.
      if (it->GetPacketState() == PacketStateEntryFirstLine)
         formatter.FormatLinePrefix(stream, it->GetMetadataConstRef());
      else // PacketStateNewLine
         formatter.FormatContinuationPrefix(stream);

      stream << ' ' << it->GetText();
      beforeFirst = false;
   }

   // Close the last output line
   if (!beforeFirst)
      stream << '\n';
}


template <class TMetadata, class UFormatter>
class GenericStdErrLogSink : public GenericSink<TMetadata>
{
   bool hadError_;

public:
   typedef GenericSink<TMetadata> Super;
   typedef typename Super::PacketArrayType PacketArrayType;

   GenericStdErrLogSink() : hadError_(false) {}

   virtual void Consume(const PacketArrayType& packets)
   {
      std::clog.clear();
      WritePacketsToStream<UFormatter>(std::clog,
            packets.Begin(), packets.End(), this->GetFilter());
      std::clog.flush();
      if (std::clog.fail())
      {
         std::clog.clear();
         if (!hadError_)
         {
            hadError_ = true;
            // Probably futile but try anyway:
            std::cerr << "Logging: cannot write to stderr\n";
         }
      }
      else
      {
         hadError_ = false;
      }
   }
};


template <class TMetadata, class UFormatter>
class GenericFileLogSink : public GenericSink<TMetadata>
{
   std::string filename_;
   std::ofstream fileStream_;
   CountingStreambuf countingBuf_;
   std::ostream countingStream_;
   bool hadError_;
   std::size_t maxFileSize_;
   int maxBackupFiles_;
   std::size_t initialFileSize_;

public:
   typedef GenericSink<TMetadata> Super;
   typedef typename Super::PacketArrayType PacketArrayType;

   GenericFileLogSink(const GenericFileLogSink&) = delete;
   GenericFileLogSink& operator=(const GenericFileLogSink&) = delete;

   GenericFileLogSink(const std::string& filename, bool append = false,
         std::size_t maxFileSize = 0, int maxBackupFiles = 0) :
      filename_(filename),
      countingBuf_(nullptr),
      countingStream_(nullptr),
      hadError_(false),
      maxFileSize_(maxFileSize),
      maxBackupFiles_(maxBackupFiles),
      initialFileSize_(0)
   {
      std::ios_base::openmode mode = std::ios_base::out;
      mode |= (append ? std::ios_base::app : std::ios_base::trunc);

      fileStream_.open(filename_.c_str(), mode);
      if (!fileStream_)
         throw CannotOpenFileException();

      if (append)
      {
         auto pos = fileStream_.tellp();
         if (pos > 0)
            initialFileSize_ = static_cast<std::size_t>(pos);
      }

      countingBuf_.SetWrapped(fileStream_.rdbuf());
      countingStream_.rdbuf(&countingBuf_);
   }

   virtual void Consume(const PacketArrayType& packets)
   {
      countingStream_.clear();
      WritePacketsToStream<UFormatter>(countingStream_,
            packets.Begin(), packets.End(), this->GetFilter());
      countingStream_.flush();
      if (countingStream_.fail())
      {
         countingStream_.clear();
         if (!hadError_)
         {
            hadError_ = true;
            std::cerr << "Logging: cannot write to file " << filename_ << '\n';
         }
      }
      else
      {
         hadError_ = false;
      }

      if (maxFileSize_ > 0 &&
            initialFileSize_ + countingBuf_.Count() > maxFileSize_)
      {
         RotateFile();
      }
   }

private:
   void RotateFile()
   {
      countingStream_.flush();
      fileStream_.close();

      std::string rotatedName = MakeRotatedFilename(filename_);
      if (std::rename(filename_.c_str(), rotatedName.c_str()) != 0)
      {
         if (!hadError_)
         {
            hadError_ = true;
            std::cerr << "Logging: cannot rotate file " << filename_ << '\n';
         }
         fileStream_.open(filename_.c_str(),
               std::ios_base::out | std::ios_base::app);
         return;
      }

      DeleteExcessRotatedFiles(filename_, maxBackupFiles_);

      fileStream_.open(filename_.c_str(),
            std::ios_base::out | std::ios_base::trunc);
      if (fileStream_)
      {
         countingBuf_.SetWrapped(fileStream_.rdbuf());
         countingBuf_.ResetCount();
         initialFileSize_ = 0;
         hadError_ = false;
      }
      else
      {
         std::cerr << "Logging: cannot reopen file " << filename_
               << " after rotation\n";
      }
   }
};


} // namespace internal
} // namespace logging
} // namespace internal
} // namespace mmcore
