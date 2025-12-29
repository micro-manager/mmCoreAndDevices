///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentObjectiveSetup.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 objective setup device - sends full objective
//                specifications to SDK using S_SOB command
//
// COPYRIGHT:     University of California, San Francisco, 2025
//
// LICENSE:       This file is distributed under the BSD license.
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
// AUTHOR:        Nico Stuurman, 2025

#pragma once

#include "DeviceBase.h"
#include "EvidentHubWin.h"
#include "EvidentModelWin.h"
#include "EvidentProtocolWin.h"
#include "EvidentLensDatabase.h"

//////////////////////////////////////////////////////////////////////////////
// Objective Setup Device (for one-time microscope configuration)
//////////////////////////////////////////////////////////////////////////////

class EvidentObjectiveSetup : public CGenericBase<EvidentObjectiveSetup>
{
public:
   EvidentObjectiveSetup();
   ~EvidentObjectiveSetup();

   // MMDevice API
   int Initialize();
   int Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   // Action interface - Parameterized handlers
   int OnPosDetectedName(MM::PropertyBase* pProp, MM::ActionType eAct, long position);
   int OnPosDetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct, long position);
   int OnPosDatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct, long position);
   int OnPosSendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct, long position);

   // Action interface - Special objective handlers (parameterized)
   int OnPosSpecialNA(MM::PropertyBase* pProp, MM::ActionType eAct, long position);
   int OnPosSpecialMagnification(MM::PropertyBase* pProp, MM::ActionType eAct, long position);
   int OnPosSpecialImmersion(MM::PropertyBase* pProp, MM::ActionType eAct, long position);
   int OnPosSpecialSendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct, long position);

   // Action interface - Global controls
   int OnFilterMagnification(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFilterImmersion(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   EvidentHubWin* GetHub();

   // Helper functions
   int QueryObjectiveAtPosition(int position);
   int SendObjectiveToSDK(int position);
   int SendSpecialObjectiveToSDK(int position);
   int ConvertImmersionToMediumCode(EvidentLens::ImmersionType immersion);
   int ConvertImmersionStringToMediumCode(const std::string& immersion);
   std::string FormatLensForDropdown(const EvidentLens::LensInfo* lens);
   void GetEffectiveObjectiveSpecs(int position, double& na, double& mag, int& medium);
   int UpdateDatabaseDropdown(int position);
   std::string FormatSpecsString(double na, double mag, int medium);

   // Member variables
   bool initialized_;
   std::string name_;

   // Detected objective information from GOB queries (6 positions)
   struct DetectedObjective
   {
      std::string name;
      double na;
      double magnification;
      int medium;  // 1=Dry, 2=Water, 3=Oil, 4=Silicon, 5=Gel
      bool detected;
   };
   DetectedObjective detectedObjectives_[6];

   // User override selections (6 positions)
   std::string selectedLensModel_[6];  // Model name from database, or "NONE" to clear position

   // Special/custom objective specifications (6 positions)
   double specialNA_[6];                // NA value (0.04-2.00)
   double specialMagnification_[6];     // Magnification (0.01-150)
   std::string specialImmersion_[6];    // Immersion type string

   // Filter settings for database dropdown
   std::string filterMagnification_;  // "All" or specific mag value
   std::string filterImmersion_;      // "All" or specific immersion type
};
