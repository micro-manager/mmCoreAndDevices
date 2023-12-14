#include <gtest/gtest.h>

#include "MMCore.h"

TEST(APIErrorTests, SetFocusDirectionWithInvalidDevice)
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

TEST(APIErrorTests, GetNumberOfStatesWithInvalidDevice)
{
   CMMCore c;
   // Must not throw or crash
   EXPECT_EQ(-1, c.getNumberOfStates(nullptr));
   EXPECT_EQ(-1, c.getNumberOfStates(""));
   EXPECT_EQ(-1, c.getNumberOfStates("Blah"));
   EXPECT_EQ(-1, c.getNumberOfStates("Core"));
}

TEST(APIErrorTests, GetAvailableConfigsWithInvalidGroup)
{
   CMMCore c;
   EXPECT_TRUE(c.getAvailableConfigs(nullptr).empty());
   EXPECT_TRUE(c.getAvailableConfigs("").empty());
   EXPECT_TRUE(c.getAvailableConfigs("Blah").empty());
}

TEST(APIErrorTests, GetPixelSizeUmWithNoConfig)
{
   CMMCore c;
   EXPECT_EQ(0.0, c.getPixelSizeUm(false));
   EXPECT_EQ(0.0, c.getPixelSizeUm(true));
}

TEST(APIErrorTests, IsConfigDefinedWithNullArgs)
{
   CMMCore c;
   EXPECT_FALSE(c.isConfigDefined(nullptr, "Blah"));
   EXPECT_FALSE(c.isConfigDefined("Blah", nullptr));
   EXPECT_FALSE(c.isConfigDefined(nullptr, nullptr));
   EXPECT_FALSE(c.isConfigDefined("Blah", "Blah"));
   EXPECT_FALSE(c.isConfigDefined(nullptr, ""));
   EXPECT_FALSE(c.isConfigDefined("", nullptr));
   EXPECT_FALSE(c.isConfigDefined(nullptr, nullptr));
   EXPECT_FALSE(c.isConfigDefined("", ""));
   EXPECT_FALSE(c.isConfigDefined("", "Blah"));
   EXPECT_FALSE(c.isConfigDefined("Blah", ""));
}

TEST(APIErrorTests, IsGroupDefinedWithNullArg)
{
   CMMCore c;
   EXPECT_FALSE(c.isGroupDefined(nullptr));
   EXPECT_FALSE(c.isGroupDefined(""));
   EXPECT_FALSE(c.isGroupDefined("Blah"));
}

TEST(APIErrorTests, SupportsDeviceDetectionWithInvalidDevice)
{
   CMMCore c;
   EXPECT_FALSE(c.supportsDeviceDetection(nullptr));
   EXPECT_FALSE(c.supportsDeviceDetection(""));
   EXPECT_FALSE(c.supportsDeviceDetection("Blah"));
   EXPECT_FALSE(c.supportsDeviceDetection("Core"));
}

TEST(APIErrorTests, DetectDeviceWithInvalidDevice)
{
   CMMCore c;
   EXPECT_EQ(MM::Unimplemented, c.detectDevice(nullptr));
   EXPECT_EQ(MM::Unimplemented, c.detectDevice(""));
   EXPECT_EQ(MM::Unimplemented, c.detectDevice("Blah"));
   EXPECT_EQ(MM::Unimplemented, c.detectDevice("Core"));
}