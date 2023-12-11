#include "LogManager.h"

#include "CoreUtils.h"
#include "Error.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace mm
{

namespace
{

const char* StringForLogLevel(logging::LogLevel level)
{
   switch (level)
   {
      case logging::LogLevelTrace: return "trace";
      case logging::LogLevelDebug: return "debug";
      case logging::LogLevelInfo: return "info";
      case logging::LogLevelWarning: return "warning";
      case logging::LogLevelError: return "error";
      case logging::LogLevelFatal: return "fatal";
      default: return "(unknown)";
   }
}

} // anonymous namespace

const logging::SinkMode LogManager::PrimarySinkMode = logging::SinkModeAsynchronous;

LogManager::LogManager() :
   loggingCore_(std::make_shared<logging::LoggingCore>()),
   internalLogger_(loggingCore_->NewLogger("LogManager")),
   primaryLogLevel_(logging::LogLevelInfo),
   usingStdErr_(false),
   nextSecondaryHandle_(0)
{}


void
LogManager::SetUseStdErr(bool flag)
{
   std::lock_guard<std::mutex> lock(mutex_);

   if (flag == usingStdErr_)
      return;

   usingStdErr_ = flag;
   if (flag)
   {
      if (!stdErrSink_)
      {
         stdErrSink_ = std::make_shared<logging::StdErrLogSink>();
         stdErrSink_->SetFilter(
               std::make_shared<logging::LevelFilter>(primaryLogLevel_));
      }
      loggingCore_->AddSink(stdErrSink_, PrimarySinkMode);

      LOG_INFO(internalLogger_) << "Enabled logging to stderr";
   }
   else
   {
      LOG_INFO(internalLogger_) << "Disabling logging to stderr";

      loggingCore_->RemoveSink(stdErrSink_, PrimarySinkMode);
   }
}


bool
LogManager::IsUsingStdErr() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return usingStdErr_;
}


void
LogManager::SetPrimaryLogFilename(const std::string& filename, bool truncate)
{
   std::lock_guard<std::mutex> lock(mutex_);

   if (filename == primaryFilename_)
      return;

   primaryFilename_ = filename;

   if (primaryFilename_.empty())
   {
      if (primaryFileSink_)
      {
         LOG_INFO(internalLogger_) << "Disabling primary log file";
         loggingCore_->RemoveSink(primaryFileSink_, PrimarySinkMode);
         primaryFileSink_.reset();
      }
      return;
   }

   std::shared_ptr<logging::LogSink> newSink;
   try
   {
      newSink = std::make_shared<logging::FileLogSink>(primaryFilename_, !truncate);
   }
   catch (const logging::CannotOpenFileException&)
   {
      LOG_ERROR(internalLogger_) << "Failed to open file " <<
         filename << " as primary log file";
      if (primaryFileSink_)
      {
         LOG_INFO(internalLogger_) << "Disabling primary log file";
         loggingCore_->RemoveSink(primaryFileSink_, PrimarySinkMode);
      }
      primaryFileSink_.reset();
      primaryFilename_.clear();
      throw CMMError("Cannot open file " + ToQuotedString(filename));
   }

   newSink->SetFilter(std::make_shared<logging::LevelFilter>(primaryLogLevel_));

   if (!primaryFileSink_)
   {
      loggingCore_->AddSink(newSink, PrimarySinkMode);
      primaryFileSink_ = newSink;
      LOG_INFO(internalLogger_) << "Enabled primary log file " <<
         primaryFilename_;
   }
   else
   {
      // We will use atomic swapping so that no entries get lost between the
      // two files. This makes it possible to use this function for log
      // rotation.

      LOG_INFO(internalLogger_) << "Switching primary log file";
      std::vector<std::pair<std::shared_ptr<logging::LogSink>, logging::SinkMode>> toRemove;
      std::vector<std::pair<std::shared_ptr<logging::LogSink>, logging::SinkMode>> toAdd;
      toRemove.push_back(
            std::make_pair(primaryFileSink_, PrimarySinkMode));
      toAdd.push_back(std::make_pair(newSink, PrimarySinkMode));

      loggingCore_->AtomicSwapSinks(toRemove.begin(), toRemove.end(),
            toAdd.begin(), toAdd.end());
      primaryFileSink_ = newSink;
      LOG_INFO(internalLogger_) << "Switched primary log file to " <<
         primaryFilename_;
   }
}


std::string
LogManager::GetPrimaryLogFilename() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return primaryFilename_;
}


bool
LogManager::IsUsingPrimaryLogFile() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return !primaryFilename_.empty();
}


void
LogManager::SetPrimaryLogLevel(logging::LogLevel level)
{
   std::lock_guard<std::mutex> lock(mutex_);

   if (level == primaryLogLevel_)
      return;

   logging::LogLevel oldLevel = primaryLogLevel_;
   primaryLogLevel_ = level;

   LOG_INFO(internalLogger_) << "Switching primary log level from " <<
      StringForLogLevel(oldLevel) << " to " << StringForLogLevel(level);

   std::shared_ptr<logging::EntryFilter> filter =
      std::make_shared<logging::LevelFilter>(level);

   std::vector<
      std::pair<
         std::pair<std::shared_ptr<logging::LogSink>, logging::SinkMode>,
         std::shared_ptr<logging::EntryFilter>
      >
   > changes;
   if (stdErrSink_)
   {
      changes.push_back(
            std::make_pair(std::make_pair(stdErrSink_, PrimarySinkMode),
               filter));
   }
   if (primaryFileSink_)
   {
      changes.push_back(
            std::make_pair(std::make_pair(primaryFileSink_, PrimarySinkMode),
               filter));
   }

   loggingCore_->AtomicSetSinkFilters(changes.begin(), changes.end());

   LOG_INFO(internalLogger_) << "Switched primary log level from " <<
      StringForLogLevel(oldLevel) << " to " << StringForLogLevel(level);
}


logging::LogLevel
LogManager::GetPrimaryLogLevel() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return primaryLogLevel_;
}


LogManager::LogFileHandle
LogManager::AddSecondaryLogFile(logging::LogLevel level,
      const std::string& filename, bool truncate, logging::SinkMode mode)
{
   std::lock_guard<std::mutex> lock(mutex_);

   std::shared_ptr<logging::LogSink> sink;
   try
   {
      sink = std::make_shared<logging::FileLogSink>(filename, !truncate);
   }
   catch (const logging::CannotOpenFileException&)
   {
      LOG_ERROR(internalLogger_) << "Failed to open file " <<
         filename << " as secondary log file";
      throw CMMError("Cannot open file " + ToQuotedString(filename));
   }

   sink->SetFilter(std::make_shared<logging::LevelFilter>(level));

   LogFileHandle handle = nextSecondaryHandle_++;
   secondaryLogFiles_.insert(std::make_pair(handle,
            LogFileInfo(filename, sink, mode)));

   loggingCore_->AddSink(sink, mode);

   LOG_INFO(internalLogger_) << "Added secondary log file " << filename <<
      " with log level " << StringForLogLevel(level);

   return handle;
}


void
LogManager::RemoveSecondaryLogFile(LogManager::LogFileHandle handle)
{
   std::lock_guard<std::mutex> lock(mutex_);

   std::map<LogFileHandle, LogFileInfo>::iterator foundIt =
      secondaryLogFiles_.find(handle);
   if (foundIt == secondaryLogFiles_.end())
   {
      LOG_ERROR(internalLogger_) << "Cannot remove secondary log file (" <<
         handle << ": no such handle)";
      return;
   }

   LOG_INFO(internalLogger_) << "Removing secondary log file " <<
      foundIt->second.filename_;
   loggingCore_->RemoveSink(foundIt->second.sink_, foundIt->second.mode_);
   secondaryLogFiles_.erase(foundIt);
}


logging::Logger
LogManager::NewLogger(const std::string& label)
{
   return loggingCore_->NewLogger(label);
}

} // namespace mm
