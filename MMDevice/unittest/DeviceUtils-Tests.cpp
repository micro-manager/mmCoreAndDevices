#include <catch2/catch_all.hpp>

#include "DeviceUtils.h"

#include <string>
#include <thread>
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

TEST_CASE("ConvertToString result is not clobbered by other threads",
   "[ConvertToString]")
{
   const char* mine = CDeviceUtils::ConvertToString(1L);
   const char* theirs = nullptr;
   std::thread t([&] { theirs = CDeviceUtils::ConvertToString(2L); });
   t.join();
   CHECK(std::string(mine) == "1");
   CHECK(mine != theirs);
}
