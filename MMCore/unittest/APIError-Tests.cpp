#include <catch2/catch_all.hpp>

#include "MMCore.h"

TEST_CASE("setFocusDirection with invalid device", "[APIError]")
{
   CMMCore c;
   // Must not throw or crash
   c.setFocusDirection(nullptr, 0);
   c.setFocusDirection("", 0);
   c.setFocusDirection("Blah", 0);
   c.setFocusDirection("Core", 0);
   c.setFocusDirection(nullptr, +1);
   c.setFocusDirection("", +1);
   c.setFocusDirection("Blah", +1);
   c.setFocusDirection("Core", +1);
   c.setFocusDirection(nullptr, -1);
   c.setFocusDirection("", -1);
   c.setFocusDirection("Blah", -1);
   c.setFocusDirection("Core", -1);
}

TEST_CASE("getNumberOfStates with invalid device", "[APIError]")
{
   CMMCore c;
   // Must not throw or crash
   CHECK(c.getNumberOfStates(nullptr) == -1);
   CHECK(c.getNumberOfStates("") == -1);
   CHECK(c.getNumberOfStates("Blah") == -1);
   CHECK(c.getNumberOfStates("Core") == -1);
}

TEST_CASE("getAvailableConfigs with invalid group", "[APIError]")
{
   CMMCore c;
   CHECK(c.getAvailableConfigs(nullptr).empty());
   CHECK(c.getAvailableConfigs("").empty());
   CHECK(c.getAvailableConfigs("Blah").empty());
}

TEST_CASE("getPixelSizeUm with no config", "[APIError]")
{
   CMMCore c;
   CHECK(c.getPixelSizeUm(false) == 0.0);
   CHECK(c.getPixelSizeUm(true) == 0.0);
}

TEST_CASE("isConfigDefined with null args", "[APIError]")
{
   CMMCore c;
   CHECK_FALSE(c.isConfigDefined(nullptr, "Blah"));
   CHECK_FALSE(c.isConfigDefined("Blah", nullptr));
   CHECK_FALSE(c.isConfigDefined(nullptr, nullptr));
   CHECK_FALSE(c.isConfigDefined("Blah", "Blah"));
   CHECK_FALSE(c.isConfigDefined(nullptr, ""));
   CHECK_FALSE(c.isConfigDefined("", nullptr));
   CHECK_FALSE(c.isConfigDefined(nullptr, nullptr));
   CHECK_FALSE(c.isConfigDefined("", ""));
   CHECK_FALSE(c.isConfigDefined("", "Blah"));
   CHECK_FALSE(c.isConfigDefined("Blah", ""));
}

TEST_CASE("isGroupDefined with null arg", "[APIError]")
{
   CMMCore c;
   CHECK_FALSE(c.isGroupDefined(nullptr));
   CHECK_FALSE(c.isGroupDefined(""));
   CHECK_FALSE(c.isGroupDefined("Blah"));
}

TEST_CASE("supportsDeviceDetection with invalid device", "[APIError]")
{
   CMMCore c;
   CHECK_FALSE(c.supportsDeviceDetection(nullptr));
   CHECK_FALSE(c.supportsDeviceDetection(""));
   CHECK_FALSE(c.supportsDeviceDetection("Blah"));
   CHECK_FALSE(c.supportsDeviceDetection("Core"));
}

TEST_CASE("detectDevice with invalid device", "[APIError]")
{
   CMMCore c;
   CHECK(c.detectDevice(nullptr) == MM::Unimplemented);
   CHECK(c.detectDevice("") == MM::Unimplemented);
   CHECK(c.detectDevice("Blah") == MM::Unimplemented);
   CHECK(c.detectDevice("Core") == MM::Unimplemented);
}