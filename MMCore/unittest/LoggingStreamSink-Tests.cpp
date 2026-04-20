#include <catch2/catch_all.hpp>

#include "Logging/Logging.h"

#include "Logging/FileRotation.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace mmcore {
namespace internal {
namespace logging {


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
      // Keep trying random names until we find one that doesn't exist.
      for (;;)
      {
         auto p = dir / ("mmcore-test-" + std::to_string(rd()));
         if (!std::filesystem::exists(p))
         {
            path_ = p.string();
            // Create the file so it exists for the sink to truncate.
            std::ofstream ofs(path_);
            REQUIRE(ofs.good());
            break;
         }
      }
   }

   ~TempFile() { std::filesystem::remove(path_); }

   const std::string& Path() const { return path_; }
};


TEST_CASE("file log sink writes to file", "[LoggingStreamSink]")
{
   TempFile tmp;

   auto core = std::make_shared<LoggingCore>();
   auto sink = std::make_shared<FileLogSink>(tmp.Path());
   core->AddSink(sink, SinkModeSynchronous);

   Logger lgr = core->NewLogger("test");
   lgr(LogLevelInfo, "hello from test");

   std::string contents = ReadFileContents(tmp.Path());
   REQUIRE(contents.find("hello from test") != std::string::npos);
}


TEST_CASE("file log sink throws on bad path", "[LoggingStreamSink]")
{
   REQUIRE_THROWS_AS(
      FileLogSink("/nonexistent/dir/file.log"),
      CannotOpenFileException);
}


TEST_CASE("file log sink append mode", "[LoggingStreamSink]")
{
   TempFile tmp;

   {
      auto core = std::make_shared<LoggingCore>();
      auto sink = std::make_shared<FileLogSink>(tmp.Path());
      core->AddSink(sink, SinkModeSynchronous);
      Logger lgr = core->NewLogger("test");
      lgr(LogLevelInfo, "first entry");
   }

   {
      auto core = std::make_shared<LoggingCore>();
      auto sink = std::make_shared<FileLogSink>(tmp.Path(), true);
      core->AddSink(sink, SinkModeSynchronous);
      Logger lgr = core->NewLogger("test");
      lgr(LogLevelInfo, "second entry");
   }

   std::string contents = ReadFileContents(tmp.Path());
   REQUIRE(contents.find("first entry") != std::string::npos);
   REQUIRE(contents.find("second entry") != std::string::npos);
}


// A streambuf that can be made to fail on demand.
class FailableStreambuf : public std::stringbuf
{
   bool shouldFail_ = false;

public:
   void SetShouldFail(bool fail) { shouldFail_ = fail; }

   std::string Contents() const { return str(); }

protected:
   int overflow(int c) override
   {
      if (shouldFail_)
         return traits_type::eof();
      return std::stringbuf::overflow(c);
   }

   int sync() override
   {
      if (shouldFail_)
         return -1;
      return std::stringbuf::sync();
   }
};


TEST_CASE("stream error recovery pattern", "[LoggingStreamSink]")
{
   FailableStreambuf buf;
   std::ostream stream(&buf);
   bool hadError = false;

   auto consume = [&](const char* text) {
      stream.clear();
      stream << text;
      stream.flush();
      if (stream.fail())
      {
         stream.clear();
         if (!hadError)
            hadError = true;
      }
      else
      {
         hadError = false;
      }
   };

   SECTION("normal write succeeds")
   {
      consume("hello");
      REQUIRE_FALSE(stream.fail());
      REQUIRE_FALSE(hadError);
      REQUIRE(buf.Contents().find("hello") != std::string::npos);
   }

   SECTION("clear before write enables retry after failure")
   {
      buf.SetShouldFail(true);
      consume("fail");
      REQUIRE(hadError);

      buf.SetShouldFail(false);
      consume("retry");
      REQUIRE_FALSE(hadError);
      REQUIRE(buf.Contents().find("retry") != std::string::npos);
   }

   SECTION("hadError resets on success allowing re-report")
   {
      // First failure sets hadError
      buf.SetShouldFail(true);
      consume("fail1");
      REQUIRE(hadError);

      // Success resets hadError
      buf.SetShouldFail(false);
      consume("ok");
      REQUIRE_FALSE(hadError);

      // Second failure sets hadError again
      buf.SetShouldFail(true);
      consume("fail2");
      REQUIRE(hadError);
   }

   SECTION("continuous failure does not reset hadError")
   {
      buf.SetShouldFail(true);

      consume("fail1");
      REQUIRE(hadError);

      // A second failure should leave hadError true (not reset it).
      consume("fail2");
      REQUIRE(hadError);
   }
}


class TempDir
{
   std::filesystem::path path_;

public:
   TempDir()
   {
      std::random_device rd;
      auto dir = std::filesystem::temp_directory_path();
      for (;;)
      {
         auto p = dir / ("mmcore-test-" + std::to_string(rd()));
         if (!std::filesystem::exists(p))
         {
            std::filesystem::create_directory(p);
            path_ = p;
            break;
         }
      }
   }

   ~TempDir() { std::filesystem::remove_all(path_); }

   std::string Path() const { return path_.string(); }

   std::string FilePath(const std::string& name) const
   {
      return (path_ / name).string();
   }

   std::vector<std::string> ListFiles() const
   {
      std::vector<std::string> result;
      for (const auto& entry : std::filesystem::directory_iterator(path_))
         result.push_back(entry.path().filename().string());
      std::sort(result.begin(), result.end());
      return result;
   }
};


TEST_CASE("MakeRotatedFilename format with extension",
      "[LoggingStreamSink][rotation]")
{
   std::string rotated = MakeRotatedFilename("/tmp/CoreLog.log");
   std::string filename = std::filesystem::path(rotated).filename().string();

   // Should match CoreLog_YYYYMMDDTHHMMSS.log
   std::regex pattern(R"(CoreLog_\d{8}T\d{6}\.log)");
   REQUIRE(std::regex_match(filename, pattern));
}


TEST_CASE("MakeRotatedFilename format without extension",
      "[LoggingStreamSink][rotation]")
{
   std::string rotated = MakeRotatedFilename("/tmp/CoreLog");
   std::string filename = std::filesystem::path(rotated).filename().string();

   std::regex pattern(R"(CoreLog_\d{8}T\d{6})");
   REQUIRE(std::regex_match(filename, pattern));
}


TEST_CASE("rotation triggers at size threshold",
      "[LoggingStreamSink][rotation]")
{
   TempDir dir;
   std::string logPath = dir.FilePath("test.log");

   auto core = std::make_shared<LoggingCore>();
   auto sink = std::make_shared<FileLogSink>(logPath, false, 200, 0);
   core->AddSink(sink, SinkModeSynchronous);

   Logger lgr = core->NewLogger("test");

   // Write enough entries to exceed 200 bytes
   for (int i = 0; i < 20; ++i)
      lgr(LogLevelInfo, "This is a log entry for rotation testing");

   auto files = dir.ListFiles();

   // Should have the current log file and at least one rotated file
   REQUIRE(files.size() >= 2);

   // Current file should still exist
   bool hasCurrentFile = std::find(files.begin(), files.end(), "test.log")
         != files.end();
   REQUIRE(hasCurrentFile);

   // Rotated files should match the pattern
   std::regex pattern(R"(test_\d{8}T\d{6}\.log)");
   int rotatedCount = 0;
   for (const auto& f : files)
   {
      if (f != "test.log")
      {
         REQUIRE(std::regex_match(f, pattern));
         ++rotatedCount;
      }
   }
   REQUIRE(rotatedCount >= 1);
}


TEST_CASE("excess rotated files are deleted",
      "[LoggingStreamSink][rotation]")
{
   TempDir dir;
   std::string logPath = dir.FilePath("test.log");

   // Pre-create some "old" rotated files
   {
      std::ofstream(dir.FilePath("test_20240101T000000.log")) << "old1";
      std::ofstream(dir.FilePath("test_20240102T000000.log")) << "old2";
      std::ofstream(dir.FilePath("test_20240103T000000.log")) << "old3";
   }

   auto core = std::make_shared<LoggingCore>();
   auto sink = std::make_shared<FileLogSink>(logPath, false, 200, 2);
   core->AddSink(sink, SinkModeSynchronous);

   Logger lgr = core->NewLogger("test");

   // Write enough to trigger rotation
   for (int i = 0; i < 20; ++i)
      lgr(LogLevelInfo, "This is a log entry for rotation testing");

   auto files = dir.ListFiles();

   // Count rotated files (everything except test.log)
   int rotatedCount = 0;
   for (const auto& f : files)
   {
      if (f != "test.log")
         ++rotatedCount;
   }
   REQUIRE(rotatedCount <= 2);
}


TEST_CASE("no rotation when disabled", "[LoggingStreamSink][rotation]")
{
   TempDir dir;
   std::string logPath = dir.FilePath("test.log");

   auto core = std::make_shared<LoggingCore>();
   auto sink = std::make_shared<FileLogSink>(logPath, false, 0, 0);
   core->AddSink(sink, SinkModeSynchronous);

   Logger lgr = core->NewLogger("test");

   for (int i = 0; i < 20; ++i)
      lgr(LogLevelInfo, "This is a log entry that should not trigger rotation");

   auto files = dir.ListFiles();

   // Should only have the one log file
   REQUIRE(files.size() == 1);
   REQUIRE(files[0] == "test.log");
}


TEST_CASE("DeleteExcessRotatedFiles removes oldest",
      "[LoggingStreamSink][rotation]")
{
   TempDir dir;
   std::string logPath = dir.FilePath("app.log");

   // Create rotated files
   std::ofstream(dir.FilePath("app_20240101T000000.log")) << "a";
   std::ofstream(dir.FilePath("app_20240201T000000.log")) << "b";
   std::ofstream(dir.FilePath("app_20240301T000000.log")) << "c";
   std::ofstream(dir.FilePath("app_20240401T000000.log")) << "d";

   DeleteExcessRotatedFiles(logPath, 2);

   auto files = dir.ListFiles();
   REQUIRE(files.size() == 2);
   REQUIRE(files[0] == "app_20240301T000000.log");
   REQUIRE(files[1] == "app_20240401T000000.log");
}


} // namespace logging
} // namespace internal
} // namespace mmcore
