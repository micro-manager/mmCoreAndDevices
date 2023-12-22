#include <catch2/catch_all.hpp>

#include "Logging/GenericLinePacket.h"
#include "Logging/Logging.h"
#include "Logging/GenericPacketArray.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace mm {
namespace logging {

TEST_CASE("split entry into lines", "[Logging]")
{
   Metadata::LoggerDataType loggerData("component");
   Metadata::EntryDataType entryData(LogLevelInfo);
   Metadata::StampDataType stampData;
   stampData.Stamp();

   SECTION("empty result")
   {
      const char *testStr = GENERATE(
         "", "\r", "\n", "\r\r", "\r\n", "\n\n",
         "\r\r\r", "\r\r\n", "\r\n\r", "\r\n\n",
         "\n\r\r", "\n\r\n", "\n\n\r", "\n\n\n");
      internal::GenericPacketArray<Metadata> array;
      array.AppendEntry(loggerData, entryData, stampData, testStr);
      std::vector<internal::GenericLinePacket<Metadata>> result;
      std::copy(array.Begin(), array.End(), std::back_inserter(result));
      CHECK(result.size() == 1);
      CHECK_THAT(result[0].GetText(), Catch::Matchers::Equals(""));
   }

   SECTION("single-char result")
   {
      const char *testStr = GENERATE(
         "X", "X\r", "X\n", "X\r\r", "X\r\n", "X\n\n",
         "X\r\r\r", "X\r\r\n", "X\r\n\r", "X\r\n\n",
         "X\n\r\r", "X\n\r\n", "X\n\n\r", "X\n\n\n");
      internal::GenericPacketArray<Metadata> array;
      array.AppendEntry(loggerData, entryData, stampData, testStr);
      std::vector<internal::GenericLinePacket<Metadata>> result;
      std::copy(array.Begin(), array.End(), std::back_inserter(result));
      CHECK(result.size() == 1);
      CHECK_THAT(result[0].GetText(), Catch::Matchers::Equals("X"));
   }

   SECTION("two-line result")
   {
      const char *testStr = GENERATE(
         "X\rY", "X\nY", "X\r\nY",
         "X\nY\r", "X\nY\n", "X\nY\r\r", "X\nY\r\n", "X\nY\n\n",
         "X\nY\r\r\r", "X\nY\r\r\n", "X\nY\r\n\r", "X\nY\r\n\n",
         "X\nY\n\r\r", "X\nY\n\r\n", "X\nY\n\n\r", "X\nY\n\n\n");
      internal::GenericPacketArray<Metadata> array;
      array.AppendEntry(loggerData, entryData, stampData, testStr);
      std::vector<internal::GenericLinePacket<Metadata>> result;
      std::copy(array.Begin(), array.End(), std::back_inserter(result));
      CHECK(result.size() == 2);
      CHECK(result[0].GetPacketState() == internal::PacketStateEntryFirstLine);
      CHECK_THAT(result[0].GetText(), Catch::Matchers::Equals("X"));
      CHECK(result[1].GetPacketState() == internal::PacketStateNewLine);
      CHECK_THAT(result[1].GetText(), Catch::Matchers::Equals("Y"));
   }

   SECTION("three-line result with empty middle line")
   {
      const char *testStr = GENERATE(
         "X\r\rY", "X\n\nY", "X\n\rY", "X\r\n\rY", "X\r\n\nY",
         "X\r\r\nY", "X\n\r\nY", "X\r\n\r\nY");
      internal::GenericPacketArray<Metadata> array;
      array.AppendEntry(loggerData, entryData, stampData, testStr);
      std::vector<internal::GenericLinePacket<Metadata>> result;
      std::copy(array.Begin(), array.End(), std::back_inserter(result));
      CHECK(result.size() == 3);
      CHECK(result[0].GetPacketState() == internal::PacketStateEntryFirstLine);
      CHECK_THAT(result[0].GetText(), Catch::Matchers::Equals("X"));
      CHECK(result[1].GetPacketState() == internal::PacketStateNewLine);
      CHECK_THAT(result[1].GetText(), Catch::Matchers::Equals(""));
      CHECK(result[2].GetPacketState() == internal::PacketStateNewLine);
      CHECK_THAT(result[2].GetText(), Catch::Matchers::Equals("Y"));
   }

   SECTION("one leading newline")
   {
      const char *testStr = GENERATE("\rX", "\nX", "\r\nX");
      internal::GenericPacketArray<Metadata> array;
      array.AppendEntry(loggerData, entryData, stampData, testStr);
      std::vector<internal::GenericLinePacket<Metadata>> result;
      std::copy(array.Begin(), array.End(), std::back_inserter(result));
      CHECK(result.size() == 2);
      CHECK(result[0].GetPacketState() == internal::PacketStateEntryFirstLine);
      CHECK_THAT(result[0].GetText(), Catch::Matchers::Equals(""));
      CHECK(result[1].GetPacketState() == internal::PacketStateNewLine);
      CHECK_THAT(result[1].GetText(), Catch::Matchers::Equals("X"));

   }

   SECTION("two leading newlines")
   {
      const char *testStr = GENERATE(
         "\r\rX", "\n\rX", "\n\nX", "\r\n\rX", "\r\n\nX",
         "\r\r\nX", "\n\r\nX");
      internal::GenericPacketArray<Metadata> array;
      array.AppendEntry(loggerData, entryData, stampData, testStr);
      std::vector<internal::GenericLinePacket<Metadata>> result;
      std::copy(array.Begin(), array.End(), std::back_inserter(result));
      CHECK(result.size() == 3);
      CHECK(result[0].GetPacketState() == internal::PacketStateEntryFirstLine);
      CHECK_THAT(result[0].GetText(), Catch::Matchers::Equals(""));
      CHECK(result[1].GetPacketState() == internal::PacketStateNewLine);
      CHECK_THAT(result[1].GetText(), Catch::Matchers::Equals(""));
      CHECK(result[2].GetPacketState() == internal::PacketStateNewLine);
      CHECK_THAT(result[2].GetText(), Catch::Matchers::Equals("X"));
   }

   SECTION("soft newlines")
   {
      static const std::size_t MaxLogLineLen =
          internal::GenericLinePacket<Metadata>::PacketTextLen;
      SECTION("no soft split")
      {
         std::size_t repeatCount = GENERATE(MaxLogLineLen - 1, MaxLogLineLen);
         std::string testStr(repeatCount, 'x');
         internal::GenericPacketArray<Metadata> array;
         array.AppendEntry(loggerData, entryData, stampData, testStr.c_str());
         std::vector<internal::GenericLinePacket<Metadata>> result;
         std::copy(array.Begin(), array.End(), std::back_inserter(result));
         CHECK(result.size() == 1);
         CHECK(result[0].GetPacketState() == internal::PacketStateEntryFirstLine);
         CHECK(result[0].GetText() == testStr);
      }

      SECTION("one soft split")
      {
         std::size_t charCount = GENERATE(
            MaxLogLineLen + 1,
            2 * MaxLogLineLen - 1,
            2 * MaxLogLineLen);
         std::string testStr(charCount, 'x');
         internal::GenericPacketArray<Metadata> array;
         array.AppendEntry(loggerData, entryData, stampData, testStr.c_str());
         std::vector<internal::GenericLinePacket<Metadata>> result;
         std::copy(array.Begin(), array.End(), std::back_inserter(result));
         CHECK(result.size() == 2);
         CHECK(result[0].GetPacketState() == internal::PacketStateEntryFirstLine);
         CHECK(result[0].GetText() == std::string(MaxLogLineLen, 'x'));
         CHECK(result[1].GetPacketState() == internal::PacketStateLineContinuation);
         CHECK(result[1].GetText() ==
            std::string(testStr.size() - MaxLogLineLen, 'x'));
      }

      SECTION("two soft splits")
      {
         std::size_t charCount = GENERATE(
            2 * MaxLogLineLen + 1,
            3 * MaxLogLineLen - 1,
            3 * MaxLogLineLen);
         std::string testStr(charCount, 'x');
         internal::GenericPacketArray<Metadata> array;
         array.AppendEntry(loggerData, entryData, stampData, testStr.c_str());
         std::vector<internal::GenericLinePacket<Metadata>> result;
         std::copy(array.Begin(), array.End(), std::back_inserter(result));
         CHECK(result.size() == 3);
         CHECK(result[0].GetPacketState() == internal::PacketStateEntryFirstLine);
         CHECK(result[0].GetText() == std::string(MaxLogLineLen, 'x'));
         CHECK(result[1].GetPacketState() == internal::PacketStateLineContinuation);
         CHECK(result[1].GetText() == std::string(MaxLogLineLen, 'x'));
         CHECK(result[2].GetPacketState() == internal::PacketStateLineContinuation);
         CHECK(result[2].GetText() ==
            std::string(testStr.size() - 2 * MaxLogLineLen, 'x'));
      }
   }
}

} // namespace logging
} // namespace mm