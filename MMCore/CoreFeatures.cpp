// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     2023, Board of Regents of the University of Wisconsin System
//                All Rights reserved
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Mark Tsuchida

#include "CoreFeatures.h"

#include "Error.h"

#include <map>
#include <stdexcept>
#include <utility>

// Core Features (inspired by chrome://flags)
//
// Core features are named boolean flags that control API behavior. They can be
// used for these purposes:
// - Providing a migration path for API changes, especially related to error
//   handling (stricter checks or reporting previously ignored errors).
// - Providing a way to develop new features (especially complex ones) without
//   forking or branching MMCore. The new feature can be disabled by default
//   until its API is well-vetted and stable, preventing casual users from
//   accidentally using the experimental feature without realizing. Trying to
//   access the new feature without enabling it would generally result in an
//   exception.
// - Possibly other purposes, provided that care is taken to ensure that
//   switching the feature does _not_ result in effectively two mutually
//   incompatible variants of MMCore.
//
// Importantly, feature switches are a migration strategy, _not_ a permanent
// configuration mechanism. Every feature must have the property: one or the
// other setting of the feature allows old and new user code to run. (When used
// for the introduction of better error checking, disabling the feature should
// not break user code that is compatible with the new error checking. When
// used for the introduction of new functionality, enabling the feature should
// not break compatibility with existing user code.)
//
// Typically, switching features is done manually and locally (for testing new
// behavior), or is done once during initialization by the application/library
// providing the overall environment (such as MMStudio or its analogue).
//
// How to add a new feature:
// - Add a bool flag to struct mm::features::Flags (in the .h file), with its
//   default value (usually false for a brand-new feature)
// - Add the feature name and getter/setter lambdas in the map inside
//   featureMap() (below)
// - Document the feature in the Doxygen comment for CMMCore::enableFeature()
//   (internal notes about the feature that are not useful to the user should
//   be documented in featureMap())
// - In Core code, query the feature state with: mm::features::flags().name
//
// Lifecycle of a feature:
// - Features should generally be disabled by default when first added. When
//   the feature represents experimental functionality, breaking changes can be
//   made to the new functionality while it remains in this stage.
// - When the feature is ready for widespread use, it should be enabled by
//   default. If this causes a backward-incompatible change in default Core API
//   behavior, this change requires MMCore's major version to be incremented;
//   disabling the feature should usually be deprecated. If it causes new
//   functions to be available by default, MMCore's minor version should be
//   incremented; disabling the feature may be forbidden.
// - When the old behavior (i.e., feature disabled) is no longer needed, the
//   feature should be permanently enabled: the getter should then always
//   return true, the setter should throw an exception, and the corresponding
//   flag in mm::features::Flags should be removed. However, the feature name
//   should never be removed.
// - There may be cases where a feature is abandoned before becoming enabled by
//   default. In this case, it should be permanently disabled.

namespace mm {
namespace features {

namespace internal {

Flags g_flags{};

}

namespace {

const auto& featureMap() {
   // Here we define the mapping from feature names to what they do.
   // Use functions (lambdas) to get/set the flags, so that we have the
   // possibility of having feature names that enable sets of features.
   using GetFunc = bool(*)();
   using SetFunc = void(*)(bool);
   using internal::g_flags;
   static const std::map<std::string, std::pair<GetFunc, SetFunc>> map = {
      {
         "StrictInitializationChecks", {
            [] { return g_flags.strictInitializationChecks; },
            [](bool e) { g_flags.strictInitializationChecks = e; }
            // This is made switchable to give user code (mainly the MMStudio
            // Hardware Configuration Wizard) time to be fixed and tested,
            // while allowing other environments to benefit from safer
            // behavior. It should be enabled by default when we no longer have
            // prominent users that require the old behavior, and permanently
            // enabled perhaps a few years later.
         }
      },
      {
         "ParallelDeviceInitialization", {
            [] { return g_flags.ParallelDeviceInitialization; },
            [](bool e) { g_flags.ParallelDeviceInitialization = e; }
         }
      },
      // How to add a new Core feature: see the comment at the top of this file.
      // Features (the string names) must never be removed once added!
   };
   return map;
}

}

void enableFeature(const std::string& name, bool enable) {
   try {
      featureMap().at(name).second(enable);
   } catch (const std::out_of_range&) {
      throw CMMError("No such feature: " + name);
   }
}

bool isFeatureEnabled(const std::string& name) {
   try {
      return featureMap().at(name).first();
   } catch (const std::out_of_range&) {
      throw CMMError("No such feature: " + name);
   }
}

} // namespace features
} // namespace mm
