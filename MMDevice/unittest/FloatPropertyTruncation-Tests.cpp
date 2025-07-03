#include <catch2/catch_all.hpp>

#include "Property.h"

namespace MM {

TEST_CASE("Set value is truncated to 4 digits", "[FloatPropertyTruncation]")
{
   FloatProperty fp("TestProp");
   double v;

   CHECK(fp.Set(0.00004));
   CHECK(fp.Get(v));
   CHECK(v == 0.0);

   CHECK(fp.Set(0.00005));
   CHECK(fp.Get(v));
   CHECK(v == 0.0001);

   CHECK(fp.Set(-0.00004));
   CHECK(fp.Get(v));
   CHECK(v == 0.0);

   CHECK(fp.Set(-0.00005));
   CHECK(fp.Get(v));
   CHECK(v == -0.0001);
}

TEST_CASE("Lower limit is truncated up", "[FloatPropertyTruncation]")
{
   FloatProperty fp("TestProp");

   CHECK(fp.SetLimits(0.0, 1000.0));
   CHECK(fp.GetLowerLimit() == 0.0);
   CHECK(fp.SetLimits(0.00001, 1000.0));
   CHECK(fp.GetLowerLimit() == 0.0001);
   CHECK(fp.SetLimits(0.00011, 1000.0));
   CHECK(fp.GetLowerLimit() == 0.0002);

   CHECK(fp.SetLimits(-0.0, 1000.0));
   CHECK(fp.GetLowerLimit() == 0.0);
   CHECK(fp.SetLimits(-0.00001, 1000.0));
   CHECK(fp.GetLowerLimit() == 0.0);
   CHECK(fp.SetLimits(-0.00011, 1000.0));
   CHECK(fp.GetLowerLimit() == -0.0001);
}

TEST_CASE("Upper limit is truncated down", "[FloatPropertyTruncation]")
{
   FloatProperty fp("TestProp");

   CHECK(fp.SetLimits(-1000.0, 0.0));
   CHECK(fp.GetUpperLimit() == 0.0);
   CHECK(fp.SetLimits(-1000.0, 0.00001));
   CHECK(fp.GetUpperLimit() == 0.0);
   CHECK(fp.SetLimits(-1000.0, 0.00011));
   CHECK(fp.GetUpperLimit() == 0.0001);

   CHECK(fp.SetLimits(-1000.0, -0.0));
   CHECK(fp.GetUpperLimit() == 0.0);
   CHECK(fp.SetLimits(-1000.0, -0.00001));
   CHECK(fp.GetUpperLimit() == -0.0001);
   CHECK(fp.SetLimits(-1000.0, -0.00011));
   CHECK(fp.GetUpperLimit() == -0.0002);
}

} // namespace MM