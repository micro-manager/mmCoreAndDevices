#include <catch2/catch_all.hpp>

#include "DeviceUtils.h"

#include <string>
#include <vector>

TEST_CASE("CopyLimitedString truncates", "[CopyLimitedString]")
{
   std::vector<char> dest(MM::MaxStrLength, '.');
   const std::string src(MM::MaxStrLength + 10, '*');
   CHECK_FALSE(CDeviceUtils::CopyLimitedString(dest.data(), src.c_str()));
   CHECK(dest[0] == '*');
   CHECK(dest[MM::MaxStrLength - 2] == '*');
   CHECK(dest[MM::MaxStrLength - 1] == '\0');
}

TEST_CASE("CopyLimitedString max untruncated len", "[CopyLimitedString]")
{
   std::vector<char> dest(MM::MaxStrLength, '.');
   const std::string src(MM::MaxStrLength - 1, '*');
   CHECK(CDeviceUtils::CopyLimitedString(dest.data(), src.c_str()));
   CHECK(dest[0] == '*');
   CHECK(dest[MM::MaxStrLength - 2] == '*');
   CHECK(dest[MM::MaxStrLength - 1] == '\0');
}
