#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

TEST_CASE("Per-device timeout set/get/has/unset") {
   StubGeneric dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   SECTION("no override by default; get returns global") {
      CHECK_FALSE(c.hasDeviceTimeout("dev"));
      CHECK(c.getDeviceTimeoutMs("dev") == c.getTimeoutMs());
   }

   SECTION("set, get, has, unset happy path") {
      c.setDeviceTimeoutMs("dev", 123);
      CHECK(c.hasDeviceTimeout("dev"));
      CHECK(c.getDeviceTimeoutMs("dev") == 123);
      // Global is untouched
      CHECK(c.getTimeoutMs() == 5000);

      c.unsetDeviceTimeout("dev");
      CHECK_FALSE(c.hasDeviceTimeout("dev"));
      CHECK(c.getDeviceTimeoutMs("dev") == c.getTimeoutMs());
   }

   SECTION("unset without an override is a no-op") {
      CHECK_NOTHROW(c.unsetDeviceTimeout("dev"));
      CHECK_FALSE(c.hasDeviceTimeout("dev"));
   }

   SECTION("set with zero throws") {
      CHECK_THROWS(c.setDeviceTimeoutMs("dev", 0));
      CHECK_FALSE(c.hasDeviceTimeout("dev"));
   }

   SECTION("set with negative throws") {
      CHECK_THROWS(c.setDeviceTimeoutMs("dev", -1));
      CHECK_FALSE(c.hasDeviceTimeout("dev"));
   }

   SECTION("set overwrites existing override") {
      c.setDeviceTimeoutMs("dev", 100);
      c.setDeviceTimeoutMs("dev", 200);
      CHECK(c.getDeviceTimeoutMs("dev") == 200);
   }
}

TEST_CASE("Per-device timeout: unknown label throws") {
   CMMCore c;

   CHECK_THROWS(c.setDeviceTimeoutMs("nosuchdevice", 100));
   CHECK_THROWS(c.unsetDeviceTimeout("nosuchdevice"));
   CHECK_THROWS(c.getDeviceTimeoutMs("nosuchdevice"));
   CHECK_THROWS(c.hasDeviceTimeout("nosuchdevice"));
}

TEST_CASE("Per-device timeout: Core label handling") {
   CMMCore c;

   SECTION("set on Core throws") {
      CHECK_THROWS(c.setDeviceTimeoutMs("Core", 100));
   }

   SECTION("unset on Core throws") {
      CHECK_THROWS(c.unsetDeviceTimeout("Core"));
   }

   SECTION("get on Core returns the global timeout") {
      CHECK(c.getDeviceTimeoutMs("Core") == c.getTimeoutMs());
      c.setTimeoutMs(1234);
      CHECK(c.getDeviceTimeoutMs("Core") == 1234);
   }

   SECTION("has on Core returns false") {
      CHECK_FALSE(c.hasDeviceTimeout("Core"));
   }
}

TEST_CASE("Per-device timeout survives across waitForDevice calls") {
   StubGeneric dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setDeviceTimeoutMs("dev", 777);
   c.waitForDevice("dev");
   CHECK(c.hasDeviceTimeout("dev"));
   CHECK(c.getDeviceTimeoutMs("dev") == 777);

   c.waitForDevice("dev");
   CHECK(c.hasDeviceTimeout("dev"));
   CHECK(c.getDeviceTimeoutMs("dev") == 777);
}

TEST_CASE("Per-device timeout is forgotten after unloadDevice") {
   StubGeneric dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);

   c.setDeviceTimeoutMs("dev", 999);
   CHECK(c.hasDeviceTimeout("dev"));

   c.unloadDevice("dev");
   CHECK_THROWS(c.hasDeviceTimeout("dev"));

   // Reload under the same label; override must be gone.
   c.loadDevice("dev", "mock_adapter", "dev");
   c.initializeDevice("dev");
   CHECK_FALSE(c.hasDeviceTimeout("dev"));
   CHECK(c.getDeviceTimeoutMs("dev") == c.getTimeoutMs());
}
