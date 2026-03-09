#include <catch2/catch_all.hpp>

#include "Logging/Logging.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <streambuf>
#include <string>

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


} // namespace logging
} // namespace internal
} // namespace mmcore
