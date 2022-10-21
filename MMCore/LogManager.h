#pragma once

#include "Logging/Logging.h"

#include <map>
#include <mutex>
#include <string>

namespace mm
{

/**
 * Facade to the logging subsystem.
 */
class LogManager
{
public:
   typedef int LogFileHandle;

private:
   std::shared_ptr<logging::LoggingCore> loggingCore_;
   logging::Logger internalLogger_;

   mutable std::mutex mutex_;

   logging::LogLevel primaryLogLevel_;

   bool usingStdErr_;
   std::shared_ptr<logging::LogSink> stdErrSink_;

   std::string primaryFilename_;
   std::shared_ptr<logging::LogSink> primaryFileSink_;

   LogFileHandle nextSecondaryHandle_;
   struct LogFileInfo
   {
      std::string filename_;
      std::shared_ptr<logging::LogSink> sink_;
      logging::SinkMode mode_;

      LogFileInfo(const std::string& filename,
            std::shared_ptr<logging::LogSink> sink,
            logging::SinkMode mode) :
         filename_(filename),
         sink_(sink),
         mode_(mode)
      {}
   };
   std::map<LogFileHandle, LogFileInfo> secondaryLogFiles_;

   static const logging::SinkMode PrimarySinkMode;

public:
   LogManager();

   void SetUseStdErr(bool flag);
   bool IsUsingStdErr() const;

   void SetPrimaryLogFilename(const std::string& filename, bool truncate);
   std::string GetPrimaryLogFilename() const;
   bool IsUsingPrimaryLogFile() const;

   void SetPrimaryLogLevel(logging::LogLevel level);
   logging::LogLevel GetPrimaryLogLevel() const;

   LogFileHandle AddSecondaryLogFile(logging::LogLevel level,
         const std::string& filename, bool truncate = true,
         logging::SinkMode mode = logging::SinkModeAsynchronous);
   void RemoveSecondaryLogFile(LogFileHandle handle);
   // We could add an atomic SwapSecondaryLogFile(handle, filename, truncate),
   // nice for log rotation, but we don't need it now.

   logging::Logger NewLogger(const std::string& label);
};

} // namespace mm
