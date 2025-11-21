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

   // Action interface - Position 1
   int OnPos1DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos1DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos1FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos1DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos1SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Action interface - Position 2
   int OnPos2DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos2DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos2FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos2DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos2SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Action interface - Position 3
   int OnPos3DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos3DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos3FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos3DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos3SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Action interface - Position 4
   int OnPos4DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos4DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos4FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos4DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos4SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Action interface - Position 5
   int OnPos5DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos5DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos5FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos5DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos5SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Action interface - Position 6
   int OnPos6DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos6DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos6FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos6DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos6SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Action interface - Global controls
   int OnFilterMagnification(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFilterImmersion(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   EvidentHubWin* GetHub();

   // Helper functions
   int QueryObjectiveAtPosition(int position);
   int SendObjectiveToSDK(int position, double na, double mag, int medium);
   int ConvertImmersionToMediumCode(EvidentLens::ImmersionType immersion);
   std::string FormatLensForDropdown(const EvidentLens::LensInfo* lens);
   void GetEffectiveObjectiveSpecs(int position, double& na, double& mag, int& medium);
   int UpdateFinalSpecsDisplay(int position);
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

   // Filter settings for database dropdown
   std::string filterMagnification_;  // "All" or specific mag value
   std::string filterImmersion_;      // "All" or specific immersion type
};
