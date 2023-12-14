#include <gtest/gtest.h>

#include "MMCore.h"

TEST(CoreSanityTests, CreateAndDestroyTwice)
{
   {
      CMMCore c1;
   }
   {
      CMMCore c2;
   }
}

TEST(CoreSanityTests, CreateTwoAtOnce)
{
   CMMCore c1;
   CMMCore c2;
}

TEST(CoreSanityTests, CreateAndReset)
{
   CMMCore c;
   c.reset();
}