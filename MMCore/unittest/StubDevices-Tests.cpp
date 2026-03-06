#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

TEST_CASE("StubGeneric can be default-constructed") {
   StubGeneric dev;
   CHECK(dev.name == "StubGeneric");
}

TEST_CASE("StubCamera can be default-constructed") {
   StubCamera dev;
   CHECK(dev.width == 512);
}

TEST_CASE("StubStage can be default-constructed") {
   StubStage dev;
   CHECK(dev.positionUm == 0.0);
}

TEST_CASE("StubXYStage can be default-constructed") {
   StubXYStage dev;
   CHECK(dev.posXSteps == 0);
}

TEST_CASE("StubStateDevice can be default-constructed") {
   StubStateDevice dev;
   CHECK(dev.numPositions == 10);
}

TEST_CASE("StubShutter can be default-constructed") {
   StubShutter dev;
   CHECK(dev.open == false);
}

TEST_CASE("StubMagnifier can be default-constructed") {
   StubMagnifier dev;
   CHECK(dev.magnification == 1.0);
}

TEST_CASE("StubAutoFocus can be default-constructed") {
   StubAutoFocus dev;
   CHECK(dev.offset == 0.0);
}

TEST_CASE("StubImageProcessor can be default-constructed") {
   StubImageProcessor dev;
   CHECK(dev.name == "StubImageProcessor");
}

TEST_CASE("StubSLM can be default-constructed") {
   StubSLM dev;
   CHECK(dev.width == 64);
}

TEST_CASE("StubGalvo can be default-constructed") {
   StubGalvo dev;
   CHECK(dev.posX == 0.0);
}

TEST_CASE("StubGeneric can be loaded into CMMCore") {
   StubGeneric dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubGeneric");
}

TEST_CASE("StubCamera can be loaded into CMMCore") {
   StubCamera dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubCamera");
}

TEST_CASE("StubStage can be loaded into CMMCore") {
   StubStage dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubStage");
}

TEST_CASE("StubXYStage can be loaded into CMMCore") {
   StubXYStage dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubXYStage");
}

TEST_CASE("StubStateDevice can be loaded into CMMCore") {
   StubStateDevice dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubStateDevice");
}

TEST_CASE("StubShutter can be loaded into CMMCore") {
   StubShutter dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubShutter");
}

TEST_CASE("StubMagnifier can be loaded into CMMCore") {
   StubMagnifier dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubMagnifier");
}

TEST_CASE("StubAutoFocus can be loaded into CMMCore") {
   StubAutoFocus dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubAutoFocus");
}

TEST_CASE("StubImageProcessor can be loaded into CMMCore") {
   StubImageProcessor dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubImageProcessor");
}

TEST_CASE("StubSLM can be loaded into CMMCore") {
   StubSLM dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubSLM");
}

TEST_CASE("StubGalvo can be loaded into CMMCore") {
   StubGalvo dev;
   MockAdapterWithDevices adapter{{"dev", &dev}};
   CMMCore c;
   adapter.LoadIntoCore(c);
   CHECK(c.getDeviceName("dev") == "StubGalvo");
}
