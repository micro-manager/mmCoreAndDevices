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

      // Initialize special objective specs
      specialNA_[i] = 0.16;           // Default NA
      specialMagnification_[i] = 10.0; // Default magnification
      specialImmersion_[i] = "Dry";    // Default immersion
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

      // Detected name (read-only with action handler)
      propName.str("");
      propName << "Position-" << pos << "-Detected-Name";
      CPropertyActionEx* pActNameEX = new CPropertyActionEx(this, &EvidentObjectiveSetup::OnPosDetectedName, pos);
      ret = CreateProperty(propName.str().c_str(), detectedObjectives_[pos-1].name.c_str(), MM::String, true, pActNameEX);
      if (ret != DEVICE_OK)
         return ret;

      // Detected specs (read-only with action handler)
      propName.str("");
      propName << "Position-" << pos << "-Detected-Specs";
      CPropertyActionEx* pActSpecsEX = new CPropertyActionEx(this, &EvidentObjectiveSetup::OnPosDetectedSpecs, pos);
      std::string detectedSpecs = FormatSpecsString(
         detectedObjectives_[pos-1].na,
         detectedObjectives_[pos-1].magnification,
         detectedObjectives_[pos-1].medium);
      ret = CreateProperty(propName.str().c_str(), detectedSpecs.c_str(), MM::String, true, pActSpecsEX);
      if (ret != DEVICE_OK)
         return ret;

      // Database selection dropdown
      propName.str("");
      propName << "Position-" << pos << "-Database-Selection";
      CPropertyActionEx* pActSelEX = new CPropertyActionEx(this, &EvidentObjectiveSetup::OnPosDatabaseSelection, pos);
      ret = CreateProperty(propName.str().c_str(), "NONE", MM::String, false, pActSelEX);
      if (ret != DEVICE_OK)
         return ret;

      // Populate database dropdown with all lenses
      ret = UpdateDatabaseDropdown(pos);
      if (ret != DEVICE_OK)
         return ret;

      // Send to SDK action button for this position
      propName.str("");
      propName << "Position-" << pos << "-Send-To-SDK";
      CPropertyActionEx* pActSendEX = new CPropertyActionEx(this, &EvidentObjectiveSetup::OnPosSendToSDK, pos);
      ret = CreateProperty(propName.str().c_str(), "Press to send", MM::String, false, pActSendEX);
      if (ret != DEVICE_OK)
         return ret;
      AddAllowedValue(propName.str().c_str(), "Press to send");
      AddAllowedValue(propName.str().c_str(), "Sending...");

      // Special objective properties
      // Special NA
      propName.str("");
      propName << "Position-" << pos << "-Special-NA";
      CPropertyActionEx* pActNAEX = new CPropertyActionEx(this, &EvidentObjectiveSetup::OnPosSpecialNA, pos); 
      ret = CreateProperty(propName.str().c_str(), "0.16", MM::Float, false, pActNAEX);
      if (ret != DEVICE_OK)
         return ret;
      SetPropertyLimits(propName.str().c_str(), 0.04, 2.00);

      // Special Magnification
      propName.str("");
      propName << "Position-" << pos << "-Special-Magnification";
      CPropertyActionEx* pActMagEX = new CPropertyActionEx(this, &EvidentObjectiveSetup::OnPosSpecialMagnification, pos);
      ret = CreateProperty(propName.str().c_str(), "10.0", MM::Float, false, pActMagEX);
      if (ret != DEVICE_OK)
         return ret;
      SetPropertyLimits(propName.str().c_str(), 0.01, 150.0);

      // Special Immersion
      propName.str("");
      propName << "Position-" << pos << "-Special-Immersion";
      CPropertyActionEx* pActImmEX = new CPropertyActionEx(this, &EvidentObjectiveSetup::OnPosSpecialImmersion, pos);
      ret = CreateProperty(propName.str().c_str(), "Dry", MM::String, false, pActImmEX);
      if (ret != DEVICE_OK)
         return ret;
      AddAllowedValue(propName.str().c_str(), "Dry");
      AddAllowedValue(propName.str().c_str(), "Water");
      AddAllowedValue(propName.str().c_str(), "Oil");
      AddAllowedValue(propName.str().c_str(), "Silicon oil");
      AddAllowedValue(propName.str().c_str(), "Silicon gel");

      // Special Send to SDK button
      propName.str("");
      propName << "Position-" << pos << "-Special-Send-To-SDK";
      CPropertyActionEx* pActSpecialSendEX = new CPropertyActionEx(this, &EvidentObjectiveSetup::OnPosSpecialSendToSDK, pos);
      ret = CreateProperty(propName.str().c_str(), "Press to send", MM::String, false, pActSpecialSendEX);
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
      catch (const std::logic_error&)
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
      catch (const std::logic_error&)
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

int EvidentObjectiveSetup::SendObjectiveToSDK(int position)
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
   }
   else
   {
      LogMessage("Warning: Failed to re-query objective after sending");
   }

   return DEVICE_OK;
}

int EvidentObjectiveSetup::SendSpecialObjectiveToSDK(int position)
{
   EvidentHubWin* hub = GetHub();
   if (hub == 0)
      return ERR_DEVICE_NOT_AVAILABLE;

   int idx = position - 1;

   // Get special objective specs for this position
   double na = specialNA_[idx];
   double magnification = specialMagnification_[idx];
   int medium = ConvertImmersionStringToMediumCode(specialImmersion_[idx]);

   std::ostringstream msg;
   msg << "Sending special objective to SDK - Position " << position
       << ": NA=" << na << ", Mag=" << magnification << ", Medium=" << medium;
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

   // Send S_SOB command: S_SOB position,NA,magnification,medium
   std::ostringstream sobCmd;
   sobCmd << CMD_AF_SET_OBJECTIVE_FULL << " " << position << ","
          << std::fixed << std::setprecision(2) << na << ","
          << std::fixed << std::setprecision(2) << magnification << ","
          << medium;

   ret = hub->ExecuteCommand(sobCmd.str(), response);
   if (ret != DEVICE_OK)
   {
      LogMessage("S_SOB command failed");
      std::string exitCmd = BuildCommand(CMD_OPERATION_MODE, 0);
      hub->ExecuteCommand(exitCmd, response);
      return ret;
   }

   // Exit Setting mode
   cmd = BuildCommand(CMD_OPERATION_MODE, 0);
   std::string exitResponse;
   hub->ExecuteCommand(cmd, exitResponse);

   // Re-query the objective to update detected properties
   ret = QueryObjectiveAtPosition(position);
   if (ret == DEVICE_OK)
   {
      // Update detected properties
      std::ostringstream propName;
      propName << "Position-" << position << "-Detected-Name";
      SetProperty(propName.str().c_str(), detectedObjectives_[idx].name.c_str());

      propName.str("");
      propName << "Position-" << position << "-Detected-Specs";
      std::string detectedSpecs = FormatSpecsString(
         detectedObjectives_[idx].na,
         detectedObjectives_[idx].magnification,
         detectedObjectives_[idx].medium);
      SetProperty(propName.str().c_str(), detectedSpecs.c_str());
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

int EvidentObjectiveSetup::ConvertImmersionStringToMediumCode(const std::string& immersion)
{
   if (immersion == "Dry")
      return 1;
   else if (immersion == "Water")
      return 2;
   else if (immersion == "Oil")
      return 3;
   else if (immersion == "Silicon oil")
      return 4;
   else if (immersion == "Silicon gel")
      return 5;
   else
      return 1;  // Default to Dry
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
      catch (const std::logic_error&)
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

// Position handlers (parameterized)
int EvidentObjectiveSetup::OnPosDetectedName(MM::PropertyBase* pProp, MM::ActionType eAct, long position)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(detectedObjectives_[position - 1].name.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPosDetectedSpecs(MM::PropertyBase* pProp, MM::ActionType eAct, long position)
{
   if (eAct == MM::BeforeGet)
   {
      std::string specs = FormatSpecsString(
         detectedObjectives_[position - 1].na,
         detectedObjectives_[position - 1].magnification,
         detectedObjectives_[position - 1].medium);
      pProp->Set(specs.c_str());
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPosDatabaseSelection(MM::PropertyBase* pProp, MM::ActionType eAct, long position)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(selectedLensModel_[position - 1].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      selectedLensModel_[position - 1] = value;
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPosSendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct, long position)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Press to send");
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set("Sending...");

      int ret = SendObjectiveToSDK(position);

      std::ostringstream statusMsg;
      if (ret == DEVICE_OK)
         statusMsg << "Position " << position << ": Sent successfully";
      else
         statusMsg << "Position " << position << ": Failed to send";

      SetProperty("Last-Status", statusMsg.str().c_str());
      LogMessage(statusMsg.str().c_str());

      pProp->Set("Press to send");
   }
   return DEVICE_OK;
}


// Special Objective handlers (parameterized)
int EvidentObjectiveSetup::OnPosSpecialNA(MM::PropertyBase* pProp, MM::ActionType eAct, long position)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(specialNA_[position - 1]);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(specialNA_[position - 1]);
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPosSpecialMagnification(MM::PropertyBase* pProp, MM::ActionType eAct, long position)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(specialMagnification_[position - 1]);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(specialMagnification_[position - 1]);
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPosSpecialImmersion(MM::PropertyBase* pProp, MM::ActionType eAct, long position)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(specialImmersion_[position - 1].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      specialImmersion_[position - 1] = value;
   }
   return DEVICE_OK;
}

int EvidentObjectiveSetup::OnPosSpecialSendToSDK(MM::PropertyBase* pProp, MM::ActionType eAct, long position)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Press to send");
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Set("Sending...");

      int ret = SendSpecialObjectiveToSDK(position);

      std::ostringstream statusMsg;
      if (ret == DEVICE_OK)
         statusMsg << "Position " << position << ": Special objective sent successfully";
      else
         statusMsg << "Position " << position << ": Failed to send special objective";

      SetProperty("Last-Status", statusMsg.str().c_str());
      LogMessage(statusMsg.str().c_str());

      pProp->Set("Press to send");
   }
   return DEVICE_OK;
}
