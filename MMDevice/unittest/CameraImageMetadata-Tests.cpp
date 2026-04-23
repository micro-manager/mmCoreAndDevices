#include <catch2/catch_all.hpp>

#include "CameraImageMetadata.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

// --- CameraImageMetadata serialization ---

// Split the serialized output into (count, body). The count header's exact
// padding is an implementation detail (the wire spec permits any amount of
// whitespace around the count), so tests should not pin it down.
static std::pair<std::size_t, std::string> SplitSerialized(const char* s) {
   const char* nl = std::strchr(s, '\n');
   REQUIRE(nl != nullptr);
   const std::size_t count = static_cast<std::size_t>(
      std::atol(std::string(s, nl).c_str()));
   return std::make_pair(count, std::string(nl + 1));
}

TEST_CASE("CameraImageMetadata serialize empty", "[Metadata]") {
   MM::CameraImageMetadata m;
   auto r = SplitSerialized(m.Serialize());
   CHECK(r.first == 0);
   CHECK(r.second.empty());
}

TEST_CASE("CameraImageMetadata serialize single string tag",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   auto r = SplitSerialized(m.Serialize());
   CHECK(r.first == 1);
   CHECK(r.second == "s\nExposure\n_\n1\n10.0\n");
}

TEST_CASE("CameraImageMetadata serialize single int tag", "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Binning", 2);
   auto r = SplitSerialized(m.Serialize());
   CHECK(r.first == 1);
   CHECK(r.second == "s\nBinning\n_\n1\n2\n");
}

TEST_CASE("CameraImageMetadata serialize single double tag",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("X", 1.5);
   auto r = SplitSerialized(m.Serialize());
   CHECK(r.first == 1);
   CHECK(r.second == "s\nX\n_\n1\n1.5\n");
}

TEST_CASE("CameraImageMetadata serialize multiple tags",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   m.AddTag("Binning", 2);
   m.AddTag("X", 1.5);
   auto r = SplitSerialized(m.Serialize());
   CHECK(r.first == 3);
   CHECK(r.second ==
       "s\nExposure\n_\n1\n10.0\n"
       "s\nBinning\n_\n1\n2\n"
       "s\nX\n_\n1\n1.5\n");
}

TEST_CASE("CameraImageMetadata duplicate key appended, last value wins on deserialize",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Key", "first");
   m.AddTag("Key", "second");
   auto r = SplitSerialized(m.Serialize());
   CHECK(r.first == 2);
   CHECK(r.second ==
       "s\nKey\n_\n1\nfirst\n"
       "s\nKey\n_\n1\nsecond\n");
}

TEST_CASE("CameraImageMetadata AddTag with std::string key",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag(std::string("Key"), "val");
   auto r = SplitSerialized(m.Serialize());
   CHECK(r.first == 1);
   CHECK(r.second == "s\nKey\n_\n1\nval\n");
}

TEST_CASE("CameraImageMetadata Clear resets to empty", "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   m.AddTag("Binning", 2);
   m.Clear();
   auto r = SplitSerialized(m.Serialize());
   CHECK(r.first == 0);
   CHECK(r.second.empty());
}

TEST_CASE("CameraImageMetadata Serialize twice returns same result",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   m.AddTag("Binning", 2);
   std::string first(m.Serialize());
   std::string second(m.Serialize());
   CHECK(first == second);
}
