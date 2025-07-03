#include <catch2/catch_all.hpp>

#include "MMCore.h"

TEST_CASE("CMMCore create and destroy twice", "[CoreCreateDestroy]")
{
   {
      CMMCore c1;
   }
   {
      CMMCore c2;
   }
}

TEST_CASE("CMMCore create two at once", "[CoreCreateDestroy]")
{
   CMMCore c1;
   CMMCore c2;
}

TEST_CASE("CMMCore create and reset", "[CoreCreateDestroy]")
{
   CMMCore c;
   c.reset();
}