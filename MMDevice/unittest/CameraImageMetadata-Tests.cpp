#include <catch2/catch_all.hpp>

#include "CameraImageMetadata.h"

#include <string>

// --- CameraImageMetadata serialization ---

TEST_CASE("CameraImageMetadata serialize empty", "[Metadata]") {
   MM::CameraImageMetadata m;
   CHECK(std::string(m.Serialize()) == "0\n");
}

TEST_CASE("CameraImageMetadata serialize single string tag",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   CHECK(std::string(m.Serialize()) == "1\ns\nExposure\n_\n1\n10.0\n");
}

TEST_CASE("CameraImageMetadata serialize single int tag", "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Binning", 2);
   CHECK(std::string(m.Serialize()) == "1\ns\nBinning\n_\n1\n2\n");
}

TEST_CASE("CameraImageMetadata serialize single double tag",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("X", 1.5);
   CHECK(std::string(m.Serialize()) == "1\ns\nX\n_\n1\n1.5\n");
}

TEST_CASE("CameraImageMetadata serialize multiple tags in key order",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   m.AddTag("Binning", 2);
   m.AddTag("X", 1.5);
   CHECK(std::string(m.Serialize()) ==
       "3\n"
       "s\nBinning\n_\n1\n2\n"
       "s\nExposure\n_\n1\n10.0\n"
       "s\nX\n_\n1\n1.5\n");
}

TEST_CASE("CameraImageMetadata duplicate key keeps last value",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Key", "first");
   m.AddTag("Key", "second");
   CHECK(std::string(m.Serialize()) == "1\ns\nKey\n_\n1\nsecond\n");
}

TEST_CASE("CameraImageMetadata AddTag with std::string key",
          "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag(std::string("Key"), "val");
   CHECK(std::string(m.Serialize()) == "1\ns\nKey\n_\n1\nval\n");
}

TEST_CASE("CameraImageMetadata Clear resets to empty", "[Metadata]") {
   MM::CameraImageMetadata m;
   m.AddTag("Exposure", "10.0");
   m.AddTag("Binning", 2);
   m.Clear();
   CHECK(std::string(m.Serialize()) == "0\n");
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
