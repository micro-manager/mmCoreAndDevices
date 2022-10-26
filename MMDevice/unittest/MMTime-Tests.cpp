#include <gtest/gtest.h>

#include "MMDevice.h"

using namespace MM;


TEST(MMTimeTests, RoundTripNegativeValues)
{
    ASSERT_DOUBLE_EQ(0.0, MMTime(-0.4).getUsec());
    ASSERT_DOUBLE_EQ(-1.0, MMTime(-1.0).getUsec());
    ASSERT_DOUBLE_EQ(-1'000'000.0, MMTime(-1'000'000.0).getUsec());

    ASSERT_DOUBLE_EQ(-1.0, MMTime(0, -1).getUsec());
    ASSERT_DOUBLE_EQ(-1'000'000.0, MMTime(-1, 0).getUsec());
    ASSERT_DOUBLE_EQ(-999'999.0, MMTime(-1, 1).getUsec());
    ASSERT_DOUBLE_EQ(-1'000'001.0, MMTime(-1, -1).getUsec());
}


TEST(MMTimeTests, ToString)
{
    using namespace std::literals::string_literals;
    ASSERT_EQ("0.000000"s, MMTime{}.toString());
    ASSERT_EQ("0.000001"s, MMTime(0, 1).toString());
    ASSERT_EQ("-0.000001"s, MMTime(0, -1).toString());
    ASSERT_EQ("-0.000001"s, MMTime(-1, 999'999).toString());
    ASSERT_EQ("1.000000"s, MMTime(1, 0).toString());
    ASSERT_EQ("-1.000000"s, MMTime(-1, 0).toString());
    ASSERT_EQ("-1.000001"s, MMTime(-1, -1).toString());
    ASSERT_EQ("-0.999999"s, MMTime(-1, 1).toString());
}


TEST(MMTimeTests, Arithmetic)
{
    ASSERT_EQ(MMTime(5.0), MMTime(3.0) + MMTime(2.0));
    ASSERT_EQ(MMTime(1.0), MMTime(3.0) - MMTime(2.0));
    ASSERT_EQ(MMTime(-5.0), MMTime(-3.0) + MMTime(-2.0));
    ASSERT_EQ(MMTime(-1.0), MMTime(-3.0) - MMTime(-2.0));
}


TEST(MMTimeTests, Comparison)
{
    ASSERT_TRUE(MMTime(5.0) == MMTime(5.0));
    ASSERT_TRUE(MMTime(5.0) >= MMTime(5.0));
    ASSERT_TRUE(MMTime(5.0) <= MMTime(5.0));
    ASSERT_FALSE(MMTime(5.0) != MMTime(5.0));
    ASSERT_TRUE(MMTime(3.0) > MMTime(2.0));
    ASSERT_TRUE(MMTime(3.0) >= MMTime(2.0));
    ASSERT_TRUE(MMTime(2.0) < MMTime(3.0));
    ASSERT_TRUE(MMTime(2.0) <= MMTime(3.0));
}


TEST(MMTimeTests, FromNumbers)
{
    ASSERT_EQ(MMTime(1.0), MMTime::fromUs(1.0));
    ASSERT_EQ(MMTime(1'000.0), MMTime::fromMs(1.0));
    ASSERT_EQ(MMTime(1'000'000.0), MMTime::fromSeconds(1));
}


TEST(MMTimeTests, ToNumbers)
{
    ASSERT_DOUBLE_EQ(1.0, MMTime(1.0).getUsec());
    ASSERT_DOUBLE_EQ(1.0, MMTime(1000.0).getMsec());
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
