#include <catch2/catch_all.hpp>

#include "RegisteredDeviceCollection.h"

namespace MM {
namespace internal {

TEST_CASE("RegisteredDeviceCollection empty") {
   RegisteredDeviceCollection c;
   CHECK(c.GetNumberOfDevices() == 0);
   // Size-zero buffer is not touched
   CHECK_FALSE(c.GetDeviceName(0, nullptr, 0));
   CHECK_FALSE(c.GetDeviceDescription("nonexistent", nullptr, 0));
   char buf[256];
   CHECK_FALSE(c.GetDeviceName(0, buf, sizeof(buf)));
   CHECK_FALSE(c.GetDeviceDescription("nonexistent", buf, sizeof(buf)));
   int typ = 42;
   CHECK_FALSE(c.GetDeviceType("nonexistent", &typ));
   CHECK(typ == MM::UnknownType);
}

TEST_CASE("RegisteredDeviceCollection single device returns correct info") {
   RegisteredDeviceCollection c;
   c.RegisterDevice("name", MM::ShutterDevice, "description");
   CHECK(c.GetNumberOfDevices() == 1);

   // Size-zero buffer is not touched
   CHECK_FALSE(c.GetDeviceName(0, nullptr, 0));
   // Description truncation is not an error
   CHECK(c.GetDeviceDescription("name", nullptr, 0));

   char buf[12];
   CHECK_FALSE(c.GetDeviceName(0, buf, 4));
   CHECK(c.GetDeviceName(0, buf, 5));
   CHECK(std::string(buf) == "name");
   CHECK_FALSE(c.GetDeviceName(1, buf, 11));

   CHECK(c.GetDeviceDescription("name", buf, 5));
   CHECK(std::string(buf) == "desc");
   CHECK(c.GetDeviceDescription("name", buf, 12));
   CHECK(std::string(buf) == "description");
   CHECK_FALSE(c.GetDeviceDescription("nonexistent", buf, 12));

   int typ = 42;
   CHECK(c.GetDeviceType("name", &typ));
   CHECK(typ == MM::ShutterDevice);
   CHECK_FALSE(c.GetDeviceType("nonexistent", &typ));
   CHECK(typ == MM::UnknownType);
}

TEST_CASE("RegisteredDeviceCollection multiple devices return correct info") {
   RegisteredDeviceCollection c;
   c.RegisterDevice("lucky-ferret", MM::ShutterDevice, "A shutter");
   c.RegisterDevice("bright-gnu", MM::CameraDevice, "The Camera");
   c.RegisterDevice("tender-dodo", MM::StageDevice, "Let me focus");
   CHECK(c.GetNumberOfDevices() == 3);

   char buf[256];
   CHECK(c.GetDeviceName(0, buf, sizeof(buf)));
   CHECK(std::string(buf) == "lucky-ferret");
   CHECK(c.GetDeviceName(1, buf, sizeof(buf)));
   CHECK(std::string(buf) == "bright-gnu");
   CHECK(c.GetDeviceName(2, buf, sizeof(buf)));
   CHECK(std::string(buf) == "tender-dodo");

   CHECK(c.GetDeviceDescription("lucky-ferret", buf, sizeof(buf)));
   CHECK(std::string(buf) == "A shutter");
   CHECK(c.GetDeviceDescription("bright-gnu", buf, sizeof(buf)));
   CHECK(std::string(buf) == "The Camera");
   CHECK(c.GetDeviceDescription("tender-dodo", buf, sizeof(buf)));
   CHECK(std::string(buf) == "Let me focus");

   int typ;
   CHECK(c.GetDeviceType("lucky-ferret", &typ));
   CHECK(typ == MM::ShutterDevice);
   CHECK(c.GetDeviceType("bright-gnu", &typ));
   CHECK(typ == MM::CameraDevice);
   CHECK(c.GetDeviceType("tender-dodo", &typ));
   CHECK(typ == MM::StageDevice);
}

} // namespace internal
} // namespace MM