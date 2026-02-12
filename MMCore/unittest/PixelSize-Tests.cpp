#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "MMCore.h"
#include "MockDeviceUtils.h"
#include "StubDevices.h"

TEST_CASE("Pixel size and affine matrix are unscaled with no magnifier or camera") {
   const char* resID = "res0";
   std::vector<double> matrix = {
      0.9, 0.1, 0.5,
      0.2, 1.1, -0.5,
      // 0.0, 0.0, 1.0,
   };

   CMMCore c;
   c.definePixelSizeConfig(resID);
   c.setPixelSizeUm(resID, 0.95);
   c.setPixelSizeAffine(resID, matrix);
   c.setPixelSizeConfig(resID);

   CHECK(c.getPixelSizeUm(false) == 0.95);
   CHECK(c.getPixelSizeAffine(false) == matrix);
}

TEST_CASE("Pixel size and affine matrix are scaled by magnification") {
   StubMagnifier mag0;
   mag0.magnification = 2.0;
   StubMagnifier mag1;
   mag1.magnification = 3.0;
   MockAdapterWithDevices adapter{
      {"mag0", &mag0},
      {"mag1", &mag1},
   };

   CMMCore c;
   adapter.LoadIntoCore(c);

   const char* resID = "res0";
   std::vector<double> matrix = {
      0.9, 0.1, 0.5,
      0.2, 1.1, -0.5,
   };

   c.definePixelSizeConfig(resID);
   c.setPixelSizeUm(resID, 0.95);
   c.setPixelSizeAffine(resID, matrix);
   c.setPixelSizeConfig(resID);

   // Scaling the affine matrix just applies the scaling to the linear part and
   // the translation part -- so all elements are scaled equally.
   auto scaledMatrix = matrix;
   for (auto& v : scaledMatrix) {
      v /= 6.0;
   }

   CHECK_THAT(c.getPixelSizeUm(false),
      Catch::Matchers::WithinAbs(0.95 / 6.0, 1e-9));
   CHECK_THAT(c.getPixelSizeAffine(false),
      Catch::Matchers::Approx(scaledMatrix).margin(1e-9));
}

TEST_CASE("Pixel size and affine matrix are scaled by camera binning") {
   StubCamera cam;
   cam.binning = 4;
   MockAdapterWithDevices adapter{
      {"cam", &cam},
   };

   CMMCore c;
   adapter.LoadIntoCore(c);
   c.setCameraDevice("cam");

   const char* resID = "res0";
   std::vector<double> matrix = {
      0.9, 0.1, 0.5,
      0.2, 1.1, -0.5,
   };

   c.definePixelSizeConfig(resID);
   c.setPixelSizeUm(resID, 0.95);
   c.setPixelSizeAffine(resID, matrix);
   c.setPixelSizeConfig(resID);

   auto scaledMatrix = matrix;
   for (auto& v : scaledMatrix) {
      v *= 4.0;
   }

   CHECK_THAT(c.getPixelSizeUm(false),
      Catch::Matchers::WithinAbs(0.95 * 4.0, 1e-9));
   CHECK_THAT(c.getPixelSizeAffine(false),
      Catch::Matchers::Approx(scaledMatrix).margin(1e-9));
}
