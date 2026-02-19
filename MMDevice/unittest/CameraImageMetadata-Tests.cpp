#include <catch2/catch_all.hpp>

#include "CameraImageMetadata.h"

#include <string>

// --- CameraImageMetadata serialization ---

TEST_CASE("CameraImageMetadata serialize empty", "[Metadata]") {
   MM::CameraImageMetadata m;
   CHECK(std::string(m.Serialize()) == "0\n");
}
