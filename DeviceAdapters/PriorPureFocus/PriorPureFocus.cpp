///////////////////////////////////////////////////////////////////////////////
// FILE:          PriorPureFocus.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Prior PureFocus Autofocus System
//                
//                
// AUTHOR:        Nico Stuurman
//
// COPYRIGHT:     Regents of the University of California, 2025
//
// LICENSE:       This file is distributed under the BSD license.
//

#include "PriorPureFocus.h"
#include <cstdio>
#include <string>
#include <math.h>
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include <sstream>

using namespace std;

// External names used by the rest of the system
// to load particular device from the "PriorPureFocus.dll" library
const char* g_PureFocusDevice = "Prior PureFocus";

// Device name variables
const char* g_PureFocusDeviceName = "PureFocus";
const char* g_PureFocusDeviceDescription = "Prior Scientific PureFocus Autofocus System";

const char* g_Stepper = "Stepper Drive";
const char* g_Piezo = "Piezo Drive";
const char* g_Measure = "Measure";


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_PureFocusDevice, MM::AutoFocusDevice, g_PureFocusDeviceDescription);
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_PureFocusDevice) == 0)
   {
      return new CPureFocus();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// CPureFocus implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

CPureFocus::CPureFocus() :
   initialized_(false),
   name_(g_PureFocusDeviceName),
   port_("Undefined"),
   locked_(false),
   busy_(false),
   answerTimeoutMs_(1000),
   objective_(0),
   piezoRange_(0),
   statusMode_(0),
   pinholeColumns_(1),
   pinholeWidth_(1),
   laserPower_(2048),
   dateString_(""),
   date_(""),
   version_("")
{
   InitializeDefaultErrorMessages();

   // create pre-initialization properties
   // ------------------------------------

   // Port
   CPropertyAction* pAct = new CPropertyAction(this, &CPureFocus::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

CPureFocus::~CPureFocus()
{
   Shutdown();
}

void CPureFocus::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}

int CPureFocus::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   // Set property list
   // -----------------

   // Name
   int ret = CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);
   if (DEVICE_OK != ret)
      return ret;

   // Description
   ret = CreateProperty(MM::g_Keyword_Description, g_PureFocusDeviceDescription, MM::String, true);
   if (DEVICE_OK != ret)
      return ret;

   // Check if we talk to the Prior and get the build date and version info
   ret = SendSerialCommand(port_.c_str(), "DATE", "\r"); // Query device
   if (ret != DEVICE_OK)
      return ret;

   // Check if we're talking to the right device
   std::string signature;
   ret = GetSerialAnswer(port_.c_str(), "\r", signature);
   if (ret != DEVICE_OK)
      return ret;

   if (signature.length() < 6 || signature.substr(0, 5) != "Prior")
   {
      return ERR_DEVICE_NOT_FOUND;
   }

   ret = GetSerialAnswer(port_.c_str(), "\r", dateString_);
   if (ret != DEVICE_OK)
      return ret;
   // this relies on Prior not changing their date_version string format
   size_t pos1 = findNthChar(dateString_, ' ', 1);
   size_t pos2 = findNthChar(dateString_, ' ', 2);
   size_t pos3 = findNthChar(dateString_, ' ', 3);
   if (pos3 != std::string::npos) {
      version_ = dateString_.substr(pos1 + 1, pos2 - pos1);
      date_ = dateString_.substr(pos3 + 1);
   }
   else {
      return ERR_DEVICE_NOT_FOUND;
   }

   // Status
   CPropertyAction* pAct = new CPropertyAction(this, &CPureFocus::OnStatus);
   ret = CreateProperty("Status", "Unknown", MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // version 
   pAct = new CPropertyAction(this, &CPureFocus::OnVersion);
   ret = CreateProperty("Version", version_.c_str(), MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // build date
   pAct = new CPropertyAction(this, &CPureFocus::OnBuildDate);
   ret = CreateProperty("BuildDate", date_.c_str(), MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Piezo or Stepper (or Measure???)
   pAct = new CPropertyAction(this, &CPureFocus::OnFocusControl);
   ret = CreateProperty("FocusControl", g_Piezo, MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue("FocusControl", g_Stepper);
   AddAllowedValue("FocusControl", g_Piezo);
   AddAllowedValue("FocusControl", g_Measure);
  
   char driveType[MM::MaxStrLength];
   GetProperty("FocusControl", driveType);
   if (strcmp(driveType, g_Piezo) == 0) {
      // First get the Piezo range
      ret = SendSerialCommand(port_.c_str(), "UPR", "\r"); // Query device
      if (ret != DEVICE_OK)
         return ret;

      // Check if we're talking to the right device
      std::string answer;
      ret = GetSerialAnswer(port_.c_str(), "\r", answer);
      if (ret != DEVICE_OK)
         return ret;

      piezoRange_ = std::stoi(answer);

      pAct = new CPropertyAction(this, &CPureFocus::OnPiezoPosition);
      ret = CreateProperty("PiezoPosition", "0.0", MM::Float, false, pAct);
      if (ret != DEVICE_OK)
         return ret;

      SetPropertyLimits("PiezoPosition", 0.0, piezoRange_);
   }

   // Lock
   pAct = new CPropertyAction(this, &CPureFocus::OnLock);
   ret = CreateProperty("Lock", "Unlocked", MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   AddAllowedValue("Lock", "Locked");
   AddAllowedValue("Lock", "Unlocked");

   // Focus score
   pAct = new CPropertyAction(this, &CPureFocus::OnFocusScore);
   ret = CreateProperty("Focus Score", "0", MM::Integer, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Offset
   pAct = new CPropertyAction(this, &CPureFocus::OnOffset);
   ret = CreateProperty("Offset", "0", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Pinhole Columns
   pAct = new CPropertyAction(this, &CPureFocus::OnPinholeColumns);
   ret = CreateProperty("Pinhole Columns", "1", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Pinhole Width
   pAct = new CPropertyAction(this, &CPureFocus::OnPinholeWidth);
   ret = CreateProperty("Pinhole Width", "1", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Laser Power
   pAct = new CPropertyAction(this, &CPureFocus::OnLaserPower);
   ret = CreateProperty("Laser Power", "2048", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Set allowable ranges
   SetPropertyLimits("Laser Power", 0, 4095);

   // Query device for actual values
   ret = UpdatePinholeProperties();
   if (ret != DEVICE_OK)
      return ret;

   ret = UpdateLaserPower();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;
   return DEVICE_OK;
}

int CPureFocus::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool CPureFocus::Busy()
{
   // For more complex implementations, you might want to query the device status
   // to determine if it's busy, rather than relying on a flag
   MMThreadGuard guard(lock_);
   return busy_;
}

int CPureFocus::SetContinuousFocusing(bool state)
{
   if (state == locked_)
      return DEVICE_OK;

   MMThreadGuard guard(lock_);

   ostringstream cmd;
   cmd << "SERVO," << (state ? "1" : "0");
   int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r");

   if (ret != DEVICE_OK)
      return ret;

   // Get response to confirm command was executed
   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   locked_ = state;
   return DEVICE_OK;
}

int CPureFocus::GetContinuousFocusing(bool& state)
{
   // Query the actual status from the device
   MMThreadGuard guard(lock_);
   int ret = SendSerialCommand(port_.c_str(), "SERVO", "\r");
   if (ret != DEVICE_OK)
      return ret;

   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // Parse response (should be "0" or "1")
   if (resp == "0")
      locked_ = false;
   else if (resp == "1")
      locked_ = true;
   else
      return ERR_UNEXPECTED_RESPONSE;

   state = locked_;
   return DEVICE_OK;
}

bool CPureFocus::IsContinuousFocusLocked()
{
   MMThreadGuard guard(lock_);
   int ret = SendSerialCommand(port_.c_str(), "FOCUS", "\r");
   if (ret != DEVICE_OK)
      return false;

   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // Parse response (should be "0" or "1")
   if (resp == "0")
      return false; // not in focus
   else if (resp == "1")
      return true; // in focus

   return false; // error
}

int CPureFocus::FullFocus()
{
   // Check if locked - in some implementations this might be allowed even when locked{
   bool lockState;
   int ret = GetContinuousFocusing(lockState);
   if (ret != DEVICE_OK)
      return ret;

   if (lockState)
      return ERR_AUTOFOCUS_LOCKED;

   MMThreadGuard guard(lock_);
   ret = SendSerialCommand(port_.c_str(), "FOCUS", "\r");
   if (ret != DEVICE_OK)
      return ret;

   // Get response
   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // For implementations where AF is not an immediate operation, you would poll the status
   // until the operation is complete. This is device-specific.
   busy_ = true;

   // For this adapter, we're assuming the operation is completed when we get a response
   // In a real implementation, you would poll status or receive a completion notification
   busy_ = false;

   return DEVICE_OK;
}

int CPureFocus::IncrementalFocus()
{
   // Check if locked - in some implementations this might be allowed even when locked
   bool lockState;
   int ret = GetContinuousFocusing(lockState);
   if (ret != DEVICE_OK)
      return ret;

   if (lockState)
      return ERR_AUTOFOCUS_LOCKED;

   MMThreadGuard guard(lock_);
   ret = SendSerialCommand(port_.c_str(), "IF", "\r");
   if (ret != DEVICE_OK)
      return ret;

   // Get response
   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // For implementations where IF is not an immediate operation, you would poll the status
   // until the operation is complete. This is device-specific.{
   busy_ = true;

   // For this adapter, we're assuming the operation is completed when we get a response
   // In a real implementation, you would poll status or receive a completion notification
   busy_ = false;

   return DEVICE_OK;
}

int CPureFocus::GetOffset(double& offset)
{
   MMThreadGuard guard(lock_);
   int ret = SendSerialCommand(port_.c_str(), "LENSP", "\r");
   if (ret != DEVICE_OK)
      return ret;

   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // Try to parse the response as a number
   try {
      offset = std::stod(resp);
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }

   return DEVICE_OK;
}

int CPureFocus::SetOffset(double offset)
{
   MMThreadGuard guard(lock_);

   ostringstream cmd;
   cmd << "LENSG," << (long) offset; // Set offset command with parameter
   int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r");
   if (ret != DEVICE_OK)
      return ret;

   // Get response to confirm the command was executed
   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   if (resp != "0")
      return ERR_UNEXPECTED_RESPONSE;

   return DEVICE_OK;
}


int CPureFocus::GetCurrentFocusScore(double& score)
{
   MMThreadGuard guard(lock_);
   int ret = SendSerialCommand(port_.c_str(), "ERROR", "\r");
   if (ret != DEVICE_OK)
      return ret;

   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // Try to parse the response as a number
   try {
      score = std::stod(resp);
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }

   return DEVICE_OK;
}

int CPureFocus::GetLastFocusScore(double& score) {
   // Unclear what we should do.
   return GetCurrentFocusScore(score);
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CPureFocus::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         // revert
         pProp->Set(port_.c_str());
         return ERR_PORT_CHANGE_FORBIDDEN;
      }

      pProp->Get(port_);
   }

   return DEVICE_OK;
}

int CPureFocus::OnStatus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      MMThreadGuard guard(lock_);
      int ret = SendSerialCommand(port_.c_str(), "STATUS", "\r");
      if (ret != DEVICE_OK)
         return ret;

      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set(resp.c_str());
   }

   return DEVICE_OK;
}

int CPureFocus::OnLock(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // Query actual device status
      bool state;
      int ret = GetContinuousFocusing(state);
      if (ret != DEVICE_OK)
         return ret;

      if (state)
         pProp->Set("Locked");
      else
         pProp->Set("Unlocked");
   }
   else if (eAct == MM::AfterSet)
   {
      string state;
      pProp->Get(state);

      bool lockState = (state == "Locked");
      int ret = SetContinuousFocusing(lockState);
      if (ret != DEVICE_OK)
         return ret;
   }

   return DEVICE_OK;
}

int CPureFocus::OnFocusScore(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double score;
      int ret = GetCurrentFocusScore(score);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set(score);
   }

   return DEVICE_OK;
}

int CPureFocus::OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double offset;
      int ret = GetOffset(offset);
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set(offset);
   }
   else if (eAct == MM::AfterSet)
   {
      double offset;
      pProp->Get(offset);

      int ret = SetOffset(offset);
      if (ret != DEVICE_OK)
         return ret;
   }

   return DEVICE_OK;
}

int CPureFocus::OnPiezoPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      MMThreadGuard guard(lock_);
      int ret = SendSerialCommand(port_.c_str(), "PIEZO", "\r");
      if (ret != DEVICE_OK)
         return ret;

      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;

      int raw = std::stoi(resp);
      if (raw >= 0 && raw <= 4095)
         pProp->Set((float)((float)raw / 4095.0 * piezoRange_));
      else
         return ERR_COMMUNICATION;
   }
   else if (eAct == MM::AfterSet)
   {
      double val;
      pProp->Get(val);
      if (val >= 0.0 && val <= piezoRange_)
      {
         int step = (int) (val / piezoRange_ * 4095);
         std::ostringstream os;
         os << "PIEZO," << step;
         int ret = SendSerialCommand(port_.c_str(), os.str().c_str(), "\r");
         if (ret != DEVICE_OK)
            return ret;

         string resp;
         ret = GetResponse(resp);
         if (ret != DEVICE_OK)
            return ret;

         if (resp != "0")
            return ERR_COMMUNICATION;

      }

   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Communication methods
///////////////////////////////////////////////////////////////////////////////

int CPureFocus::GetResponse(string& resp)
{
   resp.clear();
   int ret = GetSerialAnswer(port_.c_str(), "\r", resp);
   if (ret != DEVICE_OK)
      return ret;
   // needed?
   return PurgeComPort(port_.c_str());
}

// Not needed anymore since we handle CR directly in GetResponse
string CPureFocus::RemoveLineEndings(string input)
{
   string output = input;
   output.erase(remove(output.begin(), output.end(), '\r'), output.end());
   output.erase(remove(output.begin(), output.end(), '\n'), output.end());
   return output;
}

// Add these new functions at the end of file, before the last }

int CPureFocus::OnPinholeColumns(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      int ret = UpdatePinholeProperties();
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set((long)pinholeColumns_);
   }
   else if (eAct == MM::AfterSet)
   {
      long columns;
      pProp->Get(columns);

      MMThreadGuard guard(lock_);
      ostringstream cmd;
      cmd << "PINHOLE," << columns << "," << pinholeWidth_;
      int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r");
      if (ret != DEVICE_OK)
         return ret;

      // Get response to confirm command was executed
      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;

      if (resp != "0") // Check if the response matches expected success code
         return ERR_UNEXPECTED_RESPONSE;

      pinholeColumns_ = (int)columns;
   }

   return DEVICE_OK;
}

int CPureFocus::OnPinholeWidth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      int ret = UpdatePinholeProperties();
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set((long)pinholeWidth_);
   }
   else if (eAct == MM::AfterSet)
   {
      long width;
      pProp->Get(width);

      MMThreadGuard guard(lock_);
      ostringstream cmd;
      cmd << "PINHOLE," << pinholeColumns_ << "," << width;
      int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r");
      if (ret != DEVICE_OK)
         return ret;

      // Get response to confirm command was executed
      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;

      if (resp != "0") // Check if the response matches expected success code
         return ERR_UNEXPECTED_RESPONSE;

      pinholeWidth_ = (int)width;
   }

   return DEVICE_OK;
}

int CPureFocus::OnLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      int ret = UpdateLaserPower();
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set((long)laserPower_);
   }
   else if (eAct == MM::AfterSet)
   {
      long power;
      pProp->Get(power);

      // Validate the range
      if (power < 0 || power > 4095)
         return DEVICE_INVALID_PROPERTY_VALUE;

      MMThreadGuard guard(lock_);
      ostringstream cmd;
      cmd << "LASER," << power;
      int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r");
      if (ret != DEVICE_OK)
         return ret;

      // Get response to confirm command was executed
      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;

      if (resp != "0") // Check if the response matches expected success code
         return ERR_UNEXPECTED_RESPONSE;

      laserPower_ = (int)power;
   }

   return DEVICE_OK;
}

int CPureFocus::OnObjective(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      int ret = UpdateLaserPower();
      if (ret != DEVICE_OK)
         return ret;

      pProp->Set((long)objective_);
   }
   else if (eAct == MM::AfterSet)
   {
      long objective;
      pProp->Get(objective);

      // Validate the range
      if (objective < 1 || objective > 6)
         return DEVICE_INVALID_PROPERTY_VALUE;

      MMThreadGuard guard(lock_);
      ostringstream cmd;
      cmd << "OBJ," << objective;
      int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r");
      if (ret != DEVICE_OK)
         return ret;

      // Get response to confirm command was executed
      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;

      if (resp != "0") // Check if the response matches expected success code
         return ERR_UNEXPECTED_RESPONSE;

      objective_ = (int)objective;
   }

   return DEVICE_OK;
}

int CPureFocus::OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(version_.c_str());
   }
   return DEVICE_OK;
}

int CPureFocus::OnBuildDate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(date_.c_str());
   }
   return DEVICE_OK;
}

int CPureFocus::OnFocusControl(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // Query current pinhole settings
      int ret = SendSerialCommand(port_.c_str(), "CONFIG", "\r");
      if (ret != DEVICE_OK)
         return ret;

      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;

      // Parse the response which should be in format "m,s" where m == 'S', 'P', or 'H', and s == 'S' or 'L'
      size_t commaPos = resp.find(",");
      if (commaPos == string::npos)
         return ERR_UNEXPECTED_RESPONSE;

      std::string result;
      char m = resp.substr(0, commaPos).c_str()[0];
      switch (m) {
         case 'S':
            result = g_Stepper;
            break;
         case 'P':
            result = g_Piezo;
            break;
         case 'H':
            result = g_Measure;
            break;
         default:
            GetCoreCallback()->LogMessage(this, "Received invalid response to command CONFIG", false);
            return ERR_COMMUNICATION;
         break;
      }
      pProp->Set(result.c_str());
   }
   return DEVICE_OK;
}

int CPureFocus::UpdatePinholeProperties()
{
   MMThreadGuard guard(lock_);

   // Query current pinhole settings
   int ret = SendSerialCommand(port_.c_str(), "PINHOLE", "\r");
   if (ret != DEVICE_OK)
      return ret;

   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // Parse the response which should be in format "c,w" where c and w are integers
   size_t commaPos = resp.find(",");
   if (commaPos == string::npos)
      return ERR_UNEXPECTED_RESPONSE;

   try {
      pinholeColumns_ = std::stoi(resp.substr(0, commaPos));
      pinholeWidth_ = std::stoi(resp.substr(commaPos + 1));
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }

   return DEVICE_OK;
}

int CPureFocus::UpdateLaserPower()
{
   MMThreadGuard guard(lock_);

   // Query current laser power
   int ret = SendSerialCommand(port_.c_str(), "LASER", "\r");
   if (ret != DEVICE_OK)
      return ret;

   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // Parse the response which should be in format "LASER,n" where n is an integer
   size_t commaPos = resp.find(",");
   if (commaPos == string::npos || resp.substr(0, commaPos) != "LASER")
      return ERR_UNEXPECTED_RESPONSE;

   try {
      laserPower_ = std::stoi(resp.substr(commaPos + 1));
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }

   return DEVICE_OK;
}

int CPureFocus::UpdateObjective()
{
   MMThreadGuard guard(lock_);

   // Query current laser power
   int ret = SendSerialCommand(port_.c_str(), "OBJ", "\r");
   if (ret != DEVICE_OK)
      return ret;

   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   try {
      objective_ = std::stoi(resp);
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }

   return DEVICE_OK;
}

size_t CPureFocus::findNthChar(const std::string& str, char targetChar, int n) {
   size_t pos = 0;
   for (int i = 0; i < n; i++) {
      pos = str.find(targetChar, pos + (i > 0));
      if (pos == std::string::npos) 
         return pos;
   }
   return pos;
}
