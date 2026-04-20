#include <catch2/catch_all.hpp>

#include "LogManager.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace mmcore {
namespace internal {

static std::string ReadFileContents(const std::string& path)
{
   std::ifstream in(path);
   return std::string(std::istreambuf_iterator<char>(in),
                      std::istreambuf_iterator<char>());
}


class TempFile
{
   std::string path_;

public:
   TempFile()
   {
      std::random_device rd;
      auto dir = std::filesystem::temp_directory_path();
      for (;;)
      {
         auto p = dir / ("mmcore-test-" + std::to_string(rd()));
         if (!std::filesystem::exists(p))
         {
            path_ = p.string();
            std::ofstream ofs(path_);
            break;
         }
      }
   }

   ~TempFile() { std::filesystem::remove(path_); }

   const std::string& Path() const { return path_; }
};


TEST_CASE("SetLogLevels independence", "[LogManager]")
{
   SECTION("setting primary does not affect stderr")
   {
      LogManager mgr;
      mgr.SetLogLevels(LogLevelWarning, true, false);
      REQUIRE(mgr.GetPrimaryLogLevel() == LogLevelWarning);
      REQUIRE(mgr.GetStderrLogLevel() == LogLevelInfo);
   }

   SECTION("setting stderr does not affect primary")
   {
      LogManager mgr;
      mgr.SetLogLevels(LogLevelError, false, true);
      REQUIRE(mgr.GetStderrLogLevel() == LogLevelError);
      REQUIRE(mgr.GetPrimaryLogLevel() == LogLevelInfo);
   }

   SECTION("setting both updates both")
   {
      LogManager mgr;
      mgr.SetLogLevels(LogLevelTrace, true, true);
      REQUIRE(mgr.GetPrimaryLogLevel() == LogLevelTrace);
      REQUIRE(mgr.GetStderrLogLevel() == LogLevelTrace);
   }
}


TEST_CASE("level filtering on primary log file", "[LogManager]")
{
   TempFile tmp;
   std::string contents;

   {
      LogManager mgr;
      mgr.SetPrimaryLogFilename(tmp.Path(), true);
      mgr.SetLogLevels(LogLevelWarning, true, false);

      logging::Logger lgr = mgr.NewLogger("test");
      lgr(LogLevelTrace, "msg-trace");
      lgr(LogLevelDebug, "msg-debug");
      lgr(LogLevelInfo, "msg-info");
      lgr(LogLevelWarning, "msg-warning");
      lgr(LogLevelError, "msg-error");
      lgr(LogLevelCritical, "msg-critical");
   }

   contents = ReadFileContents(tmp.Path());

   REQUIRE(contents.find("msg-trace") == std::string::npos);
   REQUIRE(contents.find("msg-debug") == std::string::npos);
   REQUIRE(contents.find("msg-info") == std::string::npos);
   REQUIRE(contents.find("msg-warning") != std::string::npos);
   REQUIRE(contents.find("msg-error") != std::string::npos);
   REQUIRE(contents.find("msg-critical") != std::string::npos);
}


TEST_CASE("stderr level persists across disable/enable", "[LogManager]")
{
   SECTION("level set while enabled persists through disable/enable")
   {
      LogManager mgr;
      mgr.SetUseStdErr(true);
      mgr.SetLogLevels(LogLevelError, false, true);
      mgr.SetUseStdErr(false);
      mgr.SetUseStdErr(true);
      REQUIRE(mgr.GetStderrLogLevel() == LogLevelError);
   }

   SECTION("level set while disabled takes effect on enable")
   {
      LogManager mgr;
      mgr.SetLogLevels(LogLevelError, false, true);
      mgr.SetUseStdErr(true);
      REQUIRE(mgr.GetStderrLogLevel() == LogLevelError);
   }
}

} // namespace internal
} // namespace mmcore
