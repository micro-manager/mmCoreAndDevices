#include <catch2/catch_all.hpp>

#include "DeviceBase.h"

// Test that the um <-> steps conversions in CXYStageBase are correct.

namespace {

class MockXYStage : public CXYStageBase<MockXYStage> {
    long stepsX_ = 0;
    long stepsY_ = 0;

public:
    int Initialize() override { return DEVICE_OK; }
    int Shutdown() override { return DEVICE_OK; }
    bool Busy() override { return false; }
    void GetName(char*) const override {}

    int GetLimitsUm(double &, double &, double &, double &) override {
        return DEVICE_ERR;
    }
    int GetStepLimits(long &, long &, long &, long &) override {
        return DEVICE_ERR;
    }
    int Home() override { return DEVICE_ERR; }
    int Stop() override { return DEVICE_ERR; }
    int SetOrigin() override { return DEVICE_ERR; }
    int IsXYStageSequenceable(bool &) const override { return DEVICE_ERR; }

    double GetStepSizeXUm() override { return 7.0; }
    double GetStepSizeYUm() override { return 5.0; }

    int SetPositionSteps(long xSteps, long ySteps) override {
        stepsX_ = xSteps;
        stepsY_ = ySteps;
        return DEVICE_OK;
    }

    int GetPositionSteps(long& xSteps, long& ySteps) override {
        xSteps = stepsX_;
        ySteps = stepsY_;
        return DEVICE_OK;
    }

};

}

TEST_CASE("CXYStageBase-GetPositionUm") {
    const int flipX = GENERATE(0, 1);
    const int flipY = GENERATE(0, 1);
    CAPTURE(flipX, flipY);

    auto mxy = MockXYStage();
    REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorX,
                            std::to_string(flipX).c_str()) == DEVICE_OK);
    REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorY,
                            std::to_string(flipY).c_str()) == DEVICE_OK);

    double xUm{-1}, yUm{-1};
    REQUIRE(mxy.GetPositionUm(xUm, yUm) == DEVICE_OK);
    CHECK(xUm == 0.0);
    CHECK(yUm == 0.0);

    REQUIRE(mxy.SetPositionSteps(2, 3) == DEVICE_OK);

    REQUIRE(mxy.GetPositionUm(xUm, yUm) == DEVICE_OK);
    CHECK(xUm == (flipX ? -14.0 : 14.0));
    CHECK(yUm == (flipY ? -15.0 : 15.0));
}

TEST_CASE("CXYStageBase-SetAdapterOriginUm") {
    const auto adapterOriginX_um = GENERATE(0.0, 13.0);
    const auto adapterOriginY_um = GENERATE(0.0, 17.0);
    const int flipX = GENERATE(0, 1);
    const int flipY = GENERATE(0, 1);
    CAPTURE(adapterOriginX_um, adapterOriginY_um, flipX, flipY);

    auto mxy = MockXYStage();
    REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorX,
                            std::to_string(flipX).c_str()) == DEVICE_OK);
    REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorY,
                            std::to_string(flipY).c_str()) == DEVICE_OK);

    REQUIRE(mxy.SetAdapterOriginUm(adapterOriginX_um, adapterOriginY_um) == DEVICE_OK);

    // Current position in steps does not change on setting adapter origin
    // (i.e., stage does not move):
    long xSteps{-1}, ySteps{-1};
    REQUIRE(mxy.GetPositionSteps(xSteps, ySteps) == DEVICE_OK);
    CHECK(xSteps == 0);
    CHECK(ySteps == 0);

    // Current position becomes the given um coordinates:
    double xUm{}, yUm{};
    REQUIRE(mxy.GetPositionUm(xUm, yUm) == DEVICE_OK);
    CHECK(xUm == std::round(adapterOriginX_um / 7.0) * 7.0);
    CHECK(yUm == std::round(adapterOriginY_um / 5.0) * 5.0);
}

TEST_CASE("CXYStageBase-SetPositionUm") {
    const auto adapterOriginX_um = GENERATE(0.0, 13.0);
    const auto adapterOriginY_um = GENERATE(0.0, 17.0);
    const int flipX = GENERATE(0, 1);
    const int flipY = GENERATE(0, 1);
    CAPTURE(adapterOriginX_um, adapterOriginY_um, flipX, flipY);

    auto mxy = MockXYStage();

    SECTION("translate-then-flip") {
        REQUIRE(mxy.SetAdapterOriginUm(adapterOriginX_um, adapterOriginY_um)
                == DEVICE_OK);
        REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorX,
                                std::to_string(flipX).c_str()) == DEVICE_OK);
        REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorY,
                                std::to_string(flipY).c_str()) == DEVICE_OK);

        REQUIRE(mxy.SetPositionUm(42.0, 55.0) == DEVICE_OK);

        long xSteps{}, ySteps{};
        REQUIRE(mxy.GetPositionSteps(xSteps, ySteps) == DEVICE_OK);
        CHECK(xSteps == std::lround(((flipX ? -1 : 1) * 42.0 - adapterOriginX_um) / 7.0));
        CHECK(ySteps == std::lround(((flipY ? -1 : 1) * 55.0 - adapterOriginY_um) / 5.0));
    }

    SECTION("flip-then-translate") {
        REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorX,
                                std::to_string(flipX).c_str()) == DEVICE_OK);
        REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorY,
                                std::to_string(flipY).c_str()) == DEVICE_OK);
        REQUIRE(mxy.SetAdapterOriginUm(adapterOriginX_um, adapterOriginY_um)
                == DEVICE_OK);

        REQUIRE(mxy.SetPositionUm(42.0, 55.0) == DEVICE_OK);

        long xSteps{}, ySteps{};
        REQUIRE(mxy.GetPositionSteps(xSteps, ySteps) == DEVICE_OK);
        CHECK(xSteps == std::lround(((flipX ? -1 : 1) * (42.0 - adapterOriginX_um)) / 7.0));
        CHECK(ySteps == std::lround(((flipY ? -1 : 1) * (55.0 - adapterOriginY_um)) / 5.0));
    }
}

TEST_CASE("CXYStageBase-SetRelativePositionUm") {
    const auto adapterOriginX_um = GENERATE(0.0, 13.0);
    const auto adapterOriginY_um = GENERATE(0.0, 17.0);
    const int flipX = GENERATE(0, 1);
    const int flipY = GENERATE(0, 1);
    CAPTURE(adapterOriginX_um, adapterOriginY_um, flipX, flipY);

    auto mxy = MockXYStage();

    SECTION("translate-then-flip") {
        REQUIRE(mxy.SetAdapterOriginUm(adapterOriginX_um, adapterOriginY_um)
                == DEVICE_OK);
        REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorX,
                                std::to_string(flipX).c_str()) == DEVICE_OK);
        REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorY,
                                std::to_string(flipY).c_str()) == DEVICE_OK);
    }

    SECTION("flip-then-translate") {
        REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorX,
                                std::to_string(flipX).c_str()) == DEVICE_OK);
        REQUIRE(mxy.SetProperty(MM::g_Keyword_Transpose_MirrorY,
                                std::to_string(flipY).c_str()) == DEVICE_OK);
        REQUIRE(mxy.SetAdapterOriginUm(adapterOriginX_um, adapterOriginY_um)
                == DEVICE_OK);
    }

    REQUIRE(mxy.SetRelativePositionUm(42.0, 55.0) == DEVICE_OK);

    // We moved relative to steps (0, 0), so the adapter (microns) origin
    // does not play a role in the steps position we end up in.
    long xSteps{}, ySteps{};
    REQUIRE(mxy.GetPositionSteps(xSteps, ySteps) == DEVICE_OK);
    CHECK(xSteps == std::lround(((flipX ? -1 : 1) * 42.0) / 7.0));
    CHECK(ySteps == std::lround(((flipY ? -1 : 1) * 55.0) / 5.0));
}