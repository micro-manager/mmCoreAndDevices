///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentObjectiveSetup.cpp
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

#include "EvidentObjectiveSetup.h"
#include <sstream>
#include <iomanip>

using namespace EvidentIX85Win;

///////////////////////////////////////////////////////////////////////////////
// EvidentObjectiveSetup implementation
///////////////////////////////////////////////////////////////////////////////

EvidentObjectiveSetup::EvidentObjectiveSetup() :
   initialized_(false),
   name_("IX85-ObjectiveSetup"),
   filterMagnification_("All"),
   filterImmersion_("All")
{
   // Initialize arrays
   for (int i = 0; i < 6; i++)
   {
      selectedLensModel_[i] = "NONE";  // Default to NONE
      detectedObjectives_[i].name = "";
      detectedObjectives_[i].na = 0.0;
      detectedObjectives_[i].magnification = 0.0;
      detectedObjectives_[i].medium = 1;  // Default to Dry
      detectedObjectives_[i].detected = false;
   }

   InitializeDefaultErrorMessages();

   // Custom error messages
   SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Hub device not found. Please initialize EvidentHubWin first.");
   SetErrorText(ERR_NEGATIVE_ACK, "Microscope rejected command (negative acknowledgment).");
   SetErrorText(ERR_INVALID_RESPONSE, "Invalid response from microscope.");

   // Hub property (pre-initialization)
   CreateHubIDProperty();
}

EvidentObjectiveSetup::~EvidentObjectiveSetup()
{
   Shutdown();
}

void EvidentObjectiveSetup::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

bool EvidentObjectiveSetup::Busy()
{
   return false;  // This is a setup device, not busy during normal operation
}

EvidentHubWin* EvidentObjectiveSetup::GetHub()
{
   MM::Device* device = GetParentHub();
   if (device == 0)
      return 0;

   EvidentHubWin* hub = dynamic_cast<EvidentHubWin*>(device);
   return hub;
}

int EvidentObjectiveSetup::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Initialization
///////////////////////////////////////////////////////////////////////////////

int EvidentObjectiveSetup::Initialize()
{
   EvidentHubWin* hub = GetHub();
   if (hub == 0)
      return ERR_DEVICE_NOT_AVAILABLE;

   if (initialized_)
      return DEVICE_OK;

   // Query all 6 nosepiece positions to detect installed objectives
   LogMessage("Querying installed objectives...");
   for (int pos = 1; pos <= 6; pos++)
   {
      int ret = QueryObjectiveAtPosition(pos);
      if (ret != DEVICE_OK)
      {
         std::ostringstream msg;
         msg << "Warning: Failed to query objective at position " << pos;
         LogMessage(msg.str().c_str());
      }
   }

   // Create filter properties
   CPropertyAction* pAct = new CPropertyAction(this, &EvidentObjectiveSetup::OnFilterMagnification);
   int ret = CreateProperty("Filter-By-Magnification", "All", MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue("Filter-By-Magnification", "All");
   AddAllowedValue("Filter-By-Magnification", "1.25x");
   AddAllowedValue("Filter-By-Magnification", "2.0x");
   AddAllowedValue("Filter-By-Magnification", "2.5x");
   AddAllowedValue("Filter-By-Magnification", "4.0x");
   AddAllowedValue("Filter-By-Magnification", "5.0x");
   AddAllowedValue("Filter-By-Magnification", "10.0x");
   AddAllowedValue("Filter-By-Magnification", "20.0x");
   AddAllowedValue("Filter-By-Magnification", "25.0x");
   AddAllowedValue("Filter-By-Magnification", "30.0x");
   AddAllowedValue("Filter-By-Magnification", "40.0x");
   AddAllowedValue("Filter-By-Magnification", "50.0x");
   AddAllowedValue("Filter-By-Magnification", "60.0x");
   AddAllowedValue("Filter-By-Magnification", "100.0x");
   AddAllowedValue("Filter-By-Magnification", "150.0x");

   pAct = new CPropertyAction(this, &EvidentObjectiveSetup::OnFilterImmersion);
   ret = CreateProperty("Filter-By-Immersion", "All", MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue("Filter-By-Immersion", "All");
   AddAllowedValue("Filter-By-Immersion", "Dry");
   AddAllowedValue("Filter-By-Immersion", "Water");
   AddAllowedValue("Filter-By-Immersion", "Oil");
   AddAllowedValue("Filter-By-Immersion", "Silicon");
   AddAllowedValue("Filter-By-Immersion", "Gel");

   // Create properties for each of the 6 nosepiece positions
   for (int pos = 1; pos <= 6; pos++)
   {
      std::ostringstream propName;
      CPropertyAction* pActName = 0;
      CPropertyAction* pActSpecs = 0;
      CPropertyAction* pActFinal = 0;

      // Detected name (read-only with action handler)
      propName.str("");
      propName << "Position-" << pos << "-Detected-Name";
      switch (pos)
      {
         case 1: pActName = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos1DetectedName); break;
         case 2: pActName = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos2DetectedName); break;
         case 3: pActName = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos3DetectedName); break;
         case 4: pActName = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos4DetectedName); break;
         case 5: pActName = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos5DetectedName); break;
         case 6: pActName = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos6DetectedName); break;
      }
      ret = CreateProperty(propName.str().c_str(), detectedObjectives_[pos-1].name.c_str(), MM::String, true, pActName);
      if (ret != DEVICE_OK)
         return ret;

      // Detected specs (read-only with action handler)
      propName.str("");
      propName << "Position-" << pos << "-Detected-Specs";
      switch (pos)
      {
         case 1: pActSpecs = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos1DetectedSpecs); break;
         case 2: pActSpecs = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos2DetectedSpecs); break;
         case 3: pActSpecs = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos3DetectedSpecs); break;
         case 4: pActSpecs = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos4DetectedSpecs); break;
         case 5: pActSpecs = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos5DetectedSpecs); break;
         case 6: pActSpecs = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos6DetectedSpecs); break;
      }
      std::string detectedSpecs = FormatSpecsString(
         detectedObjectives_[pos-1].na,
         detectedObjectives_[pos-1].magnification,
         detectedObjectives_[pos-1].medium);
      ret = CreateProperty(propName.str().c_str(), detectedSpecs.c_str(), MM::String, true, pActSpecs);
      if (ret != DEVICE_OK)
         return ret;

      // Database selection dropdown
      propName.str("");
      propName << "Position-" << pos << "-Database-Selection";
      CPropertyAction* pActSel = 0;
      switch (pos)
      {
         case 1: pActSel = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos1DatabaseSelection); break;
         case 2: pActSel = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos2DatabaseSelection); break;
         case 3: pActSel = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos3DatabaseSelection); break;
         case 4: pActSel = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos4DatabaseSelection); break;
         case 5: pActSel = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos5DatabaseSelection); break;
         case 6: pActSel = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos6DatabaseSelection); break;
      }
      ret = CreateProperty(propName.str().c_str(), "NONE", MM::String, false, pActSel);
      if (ret != DEVICE_OK)
         return ret;

      // Populate database dropdown with all lenses
      ret = UpdateDatabaseDropdown(pos);
      if (ret != DEVICE_OK)
         return ret;

      // Final specs that will be sent (read-only with action handler)
      propName.str("");
      propName << "Position-" << pos << "-Final-Specs";
      switch (pos)
      {
         case 1: pActFinal = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos1FinalSpecs); break;
         case 2: pActFinal = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos2FinalSpecs); break;
         case 3: pActFinal = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos3FinalSpecs); break;
         case 4: pActFinal = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos4FinalSpecs); break;
         case 5: pActFinal = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos5FinalSpecs); break;
         case 6: pActFinal = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos6FinalSpecs); break;
      }
      ret = CreateProperty(propName.str().c_str(), detectedSpecs.c_str(), MM::String, true, pActFinal);
      if (ret != DEVICE_OK)
         return ret;

      // Send to SDK action button for this position
      propName.str("");
      propName << "Position-" << pos << "-Send-To-SDK";
      CPropertyAction* pActSend = 0;
      switch (pos)
      {
         case 1: pActSend = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos1SendToSDK); break;
         case 2: pActSend = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos2SendToSDK); break;
         case 3: pActSend = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos3SendToSDK); break;
         case 4: pActSend = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos4SendToSDK); break;
         case 5: pActSend = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos5SendToSDK); break;
         case 6: pActSend = new CPropertyAction(this, &EvidentObjectiveSetup::OnPos6SendToSDK); break;
      }
      ret = CreateProperty(propName.str().c_str(), "Press to send", MM::String, false, pActSend);
      if (ret != DEVICE_OK)
         return ret;
      AddAllowedValue(propName.str().c_str(), "Press to send");
      AddAllowedValue(propName.str().c_str(), "Sending...");
   }

   // Status message (read-only)
   CreateProperty("Last-Status", "Ready", MM::String, true);

   initialized_ = true;
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////////////////

int EvidentObjectiveSetup::QueryObjectiveAtPosition(int position)
{
   EvidentHubWin* hub = GetHub();
   if (hub == 0)
      return ERR_DEVICE_NOT_AVAILABLE;

   // Send GOB command to query objective information
   std::string cmd = BuildCommand(CMD_GET_OBJECTIVE, position);
   std::string response;
   int ret = hub->ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
      return ret;

   // Parse response: GOB p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11
   // p1: Position (1-6)
   // p2: Name (string)
   // p3: NA (0.00-2.00 or "N")
   // p4: Magnification (0-200 or "N")
   // p5: Medium (1=dry, 2=water, 3=oil, 4=silicone oil, 5=silicone gel, or "N")
   // p6-p11: Additional parameters (not needed for S_SOB)

   std::vector<std::string> params = ParseParameters(response);
   if (params.size() < 5)
   {
      std::ostringstream msg;
      msg << "GOB response has insufficient parameters: " << response;
      LogMessage(msg.str().c_str());
      return ERR_INVALID_RESPONSE;
   }

   int idx = position - 1;  // Array index
   detectedObjectives_[idx].detected = true;
   detectedObjectives_[idx].name = params[1];  // p2

   // Parse NA (p3)
   if (params[2] == "N" || params[2] == "n")
   {
      detectedObjectives_[idx].na = 0.0;
   }
   else
   {
      try
      {
         detectedObjectives_[idx].na = std::stod(params[2]);
      }
      catch (...)
      {
         detectedObjectives_[idx].na = 0.0;
      }
   }

   // Parse Magnification (p4)
   if (params[3] == "N" || params[3] == "n")
   {
      detectedObjectives_[idx].magnification = 0.0;
   }
   else
   {
      try
      {
         detectedObjectives_[idx].magnification = std::stod(params[3]);
      }
      catch (...)
      {
         detectedObjectives_[idx].magnification = 0.0;
      }
   }

   // Parse Medium (p5)
   if (params[4] == "N" || params[4] == "n")
   {
      detectedObjectives_[idx].medium = 1;  // Default to Dry
   }
   else
   {
      detectedObjectives_[idx].medium = ParseIntParameter(params[4]);
      if (detectedObjectives_[idx].medium < 1 || detectedObjectives_[idx].medium > 5)
         detectedObjectives_[idx].medium = 1;  // Default to Dry
   }

   std::ostringstream msg;
   msg << "Position " << position << ": " << detectedObjectives_[idx].name
       << " (Mag=" << detectedObjectives_[idx].magnification
       << ", NA=" << detectedObjectives_[idx].na
       << ", Medium=" << detectedObjectives_[idx].medium << ")";
   LogMessage(msg.str().c_str());

   return DEVICE_OK;
}

int EvidentObjectiveSetup::SendObjectiveToSDK(int position, double na, double mag, int medium)
{
   EvidentHubWin* hub = GetHub();
   if (hub == 0)
      return ERR_DEVICE_NOT_AVAILABLE;

   int idx = position - 1;

   // Get the model name for this position
   // selectedLensModel_[idx] is always set to either "NONE" or a specific model name
   std::string modelName = selectedLensModel_[idx];

   std::ostringstream msg;
   msg << "Sending to SDK - Position " << position << ": " << modelName;
   LogMessage(msg.str().c_str());

   // Enter Setting mode
   std::string cmd = BuildCommand(CMD_OPERATION_MODE, 1);
   std::string response;
   int ret = hub->ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
   {
      LogMessage("Failed to enter Setting mode");
      return ret;
   }

   // Send S_OB command: S_OB position,name
   std::ostringstream sobCmd;
   sobCmd << CMD_AF_SET_OBJECTIVE << " " << position << "," << modelName;

   ret = hub->ExecuteCommand(sobCmd.str(), response);

   // Exit Setting mode (always do this, even if S_OB failed)
   cmd = BuildCommand(CMD_OPERATION_MODE, 0);
   std::string exitResponse;
   int exitRet = hub->ExecuteCommand(cmd, exitResponse);

   if (ret != DEVICE_OK)
   {
      std::ostringstream errMsg;
      errMsg << "S_OB command failed for position " << position << ": " << response;
      LogMessage(errMsg.str().c_str());
      return ret;
   }

   if (exitRet != DEVICE_OK)
   {
      LogMessage("Warning: Failed to exit Setting mode");
      return exitRet;
   }

   msg.str("");
   msg << "Successfully sent objective " << modelName << " for position " << position;
   LogMessage(msg.str().c_str());

   // Re-query the objective to update detected properties
   ret = QueryObjectiveAtPosition(position);
   if (ret == DEVICE_OK)
   {
      // Update the detected name and specs properties
      std::ostringstream propName;

      propName.str("");
      propName << "Position-" << position << "-Detected-Name";
      SetProperty(propName.str().c_str(), detectedObjectives_[idx].name.c_str());

      propName.str("");
      propName << "Position-" << position << "-Detected-Specs";
      std::string detectedSpecs = FormatSpecsString(
         detectedObjectives_[idx].na,
         detectedObjectives_[idx].magnification,
         detectedObjectives_[idx].medium);
      SetProperty(propName.str().c_str(), detectedSpecs.c_str());

      // Update final specs display
      UpdateFinalSpecsDisplay(position);
   }
   else
   {
      LogMessage("Warning: Failed to re-query objective after sending");
   }

   return DEVICE_OK;
}

int EvidentObjectiveSetup::ConvertImmersionToMediumCode(EvidentLens::ImmersionType immersion)
{
   switch (immersion)
   {
      case EvidentLens::Immersion_Dry:      return 1;
      case EvidentLens::Immersion_Water:    return 2;
      case EvidentLens::Immersion_Oil:      return 3;
      case EvidentLens::Immersion_Silicon:  return 4;
      case EvidentLens::Immersion_Gel:      return 5;
      default:                               return 1;  // Default to Dry
   }
}

std::string EvidentObjectiveSetup::FormatLensForDropdown(const EvidentLens::LensInfo* lens)
{
   if (lens == nullptr)
      return "(None)";

   // Just return model name to avoid comma issues with MM property system
   return lens->model;
}

void EvidentObjectiveSetup::GetEffectiveObjectiveSpecs(int position, double& na, double& mag, int& medium)
{
   int idx = position - 1;

   // Check if a specific database objective is selected (not "NONE")
   if (selectedLensModel_[idx] != "NONE")
   {
      // Try to use database selection
      const EvidentLens::LensInfo* lens = EvidentLens::GetLensByModel(selectedLensModel_[idx].c_str());
      if (lens != nullptr)
      {
         na = lens->na;
         mag = lens->magnification;
         medium = ConvertImmersionToMediumCode(lens->immersion);
         return;
      }
   }

   // Use detected objective (default for "NONE" or if database lookup fails)
   na = detectedObjectives_[idx].na;
   mag = detectedObjectives_[idx].magnification;
   medium = detectedObjectives_[idx].medium;
}

int EvidentObjectiveSetup::UpdateFinalSpecsDisplay(int position)
{
   double na, mag;
   int medium;
   GetEffectiveObjectiveSpecs(position, na, mag, medium);

   std::ostringstream propName;
   propName << "Position-" << position << "-Final-Specs";
   return SetProperty(propName.str().c_str(), FormatSpecsString(na, mag, medium).c_str());
}

int EvidentObjectiveSetup::UpdateDatabaseDropdown(int position)
{
   std::ostringstream propName;
   propName << "Position-" << position << "-Database-Selection";

   // Clear existing allowed values
   ClearAllowedValues(propName.str().c_str());

   // Add "NONE" option (clear position in SDK)
   AddAllowedValue(propName.str().c_str(), "NONE");

   // Get filter criteria
   double filterMag = 0.0;
   if (filterMagnification_ != "All")
   {
      // Remove 'x' suffix and parse
      std::string magStr = filterMagnification_;
      size_t xPos = magStr.find('x');
      if (xPos != std::string::npos)
         magStr = magStr.substr(0, xPos);
      try
      {
         filterMag = std::stod(magStr);
      }
      catch (...)
      {
         filterMag = 0.0;
      }
   }

   EvidentLens::ImmersionType filterImm = EvidentLens::Immersion_Dry;
   bool filterByImmersion = (filterImmersion_ != "All");
   if (filterByImmersion)
   {
      if (filterImmersion_ == "Dry")
         filterImm = EvidentLens::Immersion_Dry;
      else if (filterImmersion_ == "Water")
         filterImm = EvidentLens::Immersion_Water;
      else if (filterImmersion_ == "Oil")
         filterImm = EvidentLens::Immersion_Oil;
      else if (filterImmersion_ == "Silicon")
         filterImm = EvidentLens::Immersion_Silicon;
      else if (filterImmersion_ == "Gel")
         filterImm = EvidentLens::Immersion_Gel;
   }

   // Add filtered lenses from database
   for (int i = 0; i < EvidentLens::LENS_DATABASE_SIZE; i++)
   {
      const EvidentLens::LensInfo* lens = &EvidentLens::LENS_DATABASE[i];

      // Apply filters
      if (filterMag > 0.0 && lens->magnification != filterMag)
         continue;
      if (filterByImmersion && lens->immersion != filterImm)
         continue;

      std::string formatted = FormatLensForDropdown(lens);
      AddAllowedValue(propName.str().c_str(), formatted.c_str());
   }

   return DEVICE_OK;
}

std::string EvidentObjectiveSetup::FormatSpecsString(double na, double mag, int medium)
{
   std::ostringstream specs;
   specs << std::fixed << std::setprecision(1) << mag << "x, "
         << "NA=" << std::fixed << std::setprecision(2) << na << ", ";

   switch (medium)
   {
      case 1: specs << "Dry"; break;
      case 2: specs << "Water"; break;
      case 3: specs << "Oil"; break;
      case 4: specs << "Silicon"; break;
      case 5: specs << "Gel"; break;
      default: specs << "Unknown"; break;
   }

   return specs.str();
}

///////////////////////////////////////////////////////////////////////////////
// Property Handlers
///////////////////////////////////////////////////////////////////////////////

int EvidentObjectiveSetup::OnFilterMagnification(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(filterMagnification_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(filterMagnification_);

      // Update all database dropdowns with new filter
      for (int pos = 1; pos <= 6; pos++)
      {
         UpdateDatabaseDropdown(pos);
      }
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnFilterImmersion(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(filterImmersion_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(filterImmersion_);

      // Update all database dropdowns with new filter
      for (int pos = 1; pos <= 6; pos++)
      {
         UpdateDatabaseDropdown(pos);
      }
   }
   return DEVICE_OK;
}

// Position 1 - Detected properties
int EvidentObjectiveSetup::OnPos1DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(detectedObjectives_[0].name.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos1DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string specs = FormatSpecsString(
         detectedObjectives_[0].na,
         detectedObjectives_[0].magnification,
         detectedObjectives_[0].medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos1FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(1, na, mag, medium);
      std::string specs = FormatSpecsString(na, mag, medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos1DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(selectedLensModel_[0].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      selectedLensModel_[0] = value;
      UpdateFinalSpecsDisplay(1);
   }
   return DEVICE_OK;
}

// Position 2 - Detected properties
int EvidentObjectiveSetup::OnPos2DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(detectedObjectives_[1].name.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos2DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string specs = FormatSpecsString(
         detectedObjectives_[1].na,
         detectedObjectives_[1].magnification,
         detectedObjectives_[1].medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos2FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(2, na, mag, medium);
      std::string specs = FormatSpecsString(na, mag, medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos2DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(selectedLensModel_[1].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      selectedLensModel_[1] = value;
      UpdateFinalSpecsDisplay(2);
   }
   return DEVICE_OK;
}

// Position 3 - Detected properties
int EvidentObjectiveSetup::OnPos3DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(detectedObjectives_[2].name.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos3DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string specs = FormatSpecsString(
         detectedObjectives_[2].na,
         detectedObjectives_[2].magnification,
         detectedObjectives_[2].medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos3FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(3, na, mag, medium);
      std::string specs = FormatSpecsString(na, mag, medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos3DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(selectedLensModel_[2].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      selectedLensModel_[2] = value;
      UpdateFinalSpecsDisplay(3);
   }
   return DEVICE_OK;
}

// Position 4 - Detected properties
int EvidentObjectiveSetup::OnPos4DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(detectedObjectives_[3].name.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos4DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string specs = FormatSpecsString(
         detectedObjectives_[3].na,
         detectedObjectives_[3].magnification,
         detectedObjectives_[3].medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos4FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(4, na, mag, medium);
      std::string specs = FormatSpecsString(na, mag, medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos4DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(selectedLensModel_[3].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      selectedLensModel_[3] = value;
      UpdateFinalSpecsDisplay(4);
   }
   return DEVICE_OK;
}

// Position 5 - Detected properties
int EvidentObjectiveSetup::OnPos5DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(detectedObjectives_[4].name.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos5DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string specs = FormatSpecsString(
         detectedObjectives_[4].na,
         detectedObjectives_[4].magnification,
         detectedObjectives_[4].medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos5FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(5, na, mag, medium);
      std::string specs = FormatSpecsString(na, mag, medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos5DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(selectedLensModel_[4].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      selectedLensModel_[4] = value;
      UpdateFinalSpecsDisplay(5);
   }
   return DEVICE_OK;
}

// Position 6 - Detected properties
int EvidentObjectiveSetup::OnPos6DetectedName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(detectedObjectives_[5].name.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos6DetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string specs = FormatSpecsString(
         detectedObjectives_[5].na,
         detectedObjectives_[5].magnification,
         detectedObjectives_[5].medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos6FinalSpecs(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(6, na, mag, medium);
      std::string specs = FormatSpecsString(na, mag, medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos6DatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(selectedLensModel_[5].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      selectedLensModel_[5] = value;
      UpdateFinalSpecsDisplay(6);
   }
   return DEVICE_OK;
}

// Send to SDK - Position 1
int EvidentObjectiveSetup::OnPos1SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Press to send");
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set("Sending...");

      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(1, na, mag, medium);

      int ret = SendObjectiveToSDK(1, na, mag, medium);

      std::ostringstream statusMsg;
      if (ret == DEVICE_OK)
         statusMsg << "Position 1: Sent successfully";
      else
         statusMsg << "Position 1: Failed to send";

      SetProperty("Last-Status", statusMsg.str().c_str());
      LogMessage(statusMsg.str().c_str());

      pProp->Set("Press to send");
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos2SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Press to send");
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set("Sending...");

      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(2, na, mag, medium);

      int ret = SendObjectiveToSDK(2, na, mag, medium);

      std::ostringstream statusMsg;
      if (ret == DEVICE_OK)
         statusMsg << "Position 2: Sent successfully";
      else
         statusMsg << "Position 2: Failed to send";

      SetProperty("Last-Status", statusMsg.str().c_str());
      LogMessage(statusMsg.str().c_str());

      pProp->Set("Press to send");
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos3SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Press to send");
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set("Sending...");

      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(3, na, mag, medium);

      int ret = SendObjectiveToSDK(3, na, mag, medium);

      std::ostringstream statusMsg;
      if (ret == DEVICE_OK)
         statusMsg << "Position 3: Sent successfully";
      else
         statusMsg << "Position 3: Failed to send";

      SetProperty("Last-Status", statusMsg.str().c_str());
      LogMessage(statusMsg.str().c_str());

      pProp->Set("Press to send");
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos4SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Press to send");
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set("Sending...");

      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(4, na, mag, medium);

      int ret = SendObjectiveToSDK(4, na, mag, medium);

      std::ostringstream statusMsg;
      if (ret == DEVICE_OK)
         statusMsg << "Position 4: Sent successfully";
      else
         statusMsg << "Position 4: Failed to send";

      SetProperty("Last-Status", statusMsg.str().c_str());
      LogMessage(statusMsg.str().c_str());

      pProp->Set("Press to send");
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos5SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Press to send");
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set("Sending...");

      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(5, na, mag, medium);

      int ret = SendObjectiveToSDK(5, na, mag, medium);

      std::ostringstream statusMsg;
      if (ret == DEVICE_OK)
         statusMsg << "Position 5: Sent successfully";
      else
         statusMsg << "Position 5: Failed to send";

      SetProperty("Last-Status", statusMsg.str().c_str());
      LogMessage(statusMsg.str().c_str());

      pProp->Set("Press to send");
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPos6SendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Press to send");
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set("Sending...");

      double na, mag;
      int medium;
      GetEffectiveObjectiveSpecs(6, na, mag, medium);

      int ret = SendObjectiveToSDK(6, na, mag, medium);

      std::ostringstream statusMsg;
      if (ret == DEVICE_OK)
         statusMsg << "Position 6: Sent successfully";
      else
         statusMsg << "Position 6: Failed to send";

      SetProperty("Last-Status", statusMsg.str().c_str());
      LogMessage(statusMsg.str().c_str());

      pProp->Set("Press to send");
   }
   return DEVICE_OK;
}
