#include <catch2/catch_all.hpp>

#include "MMDevice.h"

namespace MM {

TEST_CASE("MMTime round trip negative values", "[MMTime]")
{
    CHECK(MMTime(-0.4).getUsec() == 0.0);
    CHECK(MMTime(-1.0).getUsec() == -1.0);
    CHECK(MMTime(-1'000'000.0).getUsec() == -1'000'000.0);

    CHECK(MMTime(0, -1).getUsec() == -1.0);
    CHECK(MMTime(-1, 0).getUsec() == -1'000'000.0);
    CHECK(MMTime(-1, 1).getUsec() == -999'999.0);
    CHECK(MMTime(-1, -1).getUsec() == -1'000'001.0);
}

TEST_CASE("MMTime to string", "[MMTime]")
{
    using namespace std::string_literals;
    CHECK(MMTime{}.toString() == "0.000000"s);
    CHECK(MMTime(0, 1).toString() == "0.000001"s);
    CHECK(MMTime(0, -1).toString() == "-0.000001"s);
    CHECK(MMTime(-1, 999'999).toString() == "-0.000001"s);
    CHECK(MMTime(1, 0).toString() == "1.000000"s);
    CHECK(MMTime(-1, 0).toString() == "-1.000000"s);
    CHECK(MMTime(-1, -1).toString() == "-1.000001"s);
    CHECK(MMTime(-1, 1).toString() == "-0.999999"s);
}

TEST_CASE("MMTime arithmetic", "[MMTime]")
{
    CHECK(MMTime(3.0) + MMTime(2.0) == MMTime(5.0));
    CHECK(MMTime(3.0) - MMTime(2.0) == MMTime(1.0));
    CHECK(MMTime(-3.0) + MMTime(-2.0) == MMTime(-5.0));
    CHECK(MMTime(-3.0) - MMTime(-2.0) == MMTime(-1.0));
}

TEST_CASE("MMTime comparison", "[MMTime]")
{
    CHECK(MMTime(5.0) == MMTime(5.0));
    CHECK(MMTime(5.0) >= MMTime(5.0));
    CHECK(MMTime(5.0) <= MMTime(5.0));
    CHECK_FALSE(MMTime(5.0) != MMTime(5.0));
    CHECK(MMTime(3.0) > MMTime(2.0));
    CHECK(MMTime(3.0) >= MMTime(2.0));
    CHECK(MMTime(2.0) < MMTime(3.0));
    CHECK(MMTime(2.0) <= MMTime(3.0));
}

TEST_CASE("MMTime from numbers", "[MMTime]")
{
    CHECK(MMTime::fromUs(1) == MMTime(1.0));
    CHECK(MMTime::fromMs(1) == MMTime(1'000.0));
    CHECK(MMTime::fromSeconds(1) == MMTime(1'000'000.0));
}

TEST_CASE("MMTime to umbers", "[MMTime]")
{
    CHECK(MMTime(1.0).getUsec() == 1.0);
    CHECK(MMTime(1000.0).getMsec() == 1.0);
}

} // namespace MM