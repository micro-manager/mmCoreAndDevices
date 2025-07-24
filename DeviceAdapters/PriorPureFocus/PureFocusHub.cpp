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

#include "PureFocus.h"
#include <cstdio>
#include <string>
#include <math.h>
#include "DeviceUtils.h"
#include <sstream>

using namespace std;


///////////////////////////////////////////////////////////////////////////////
// PureFocusHub implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

PureFocusHub::PureFocusHub() :
   initialized_(false),
   name_(g_PureFocusDeviceName),
   port_("Undefined"),
   locked_(false),
   busy_(false),
   answerTimeoutMs_(1000),
   piezoRange_(0),
   statusMode_(0),
   dateString_(""),
   date_(""),
   version_(""),
   offset_(10000),
   inRange_(false),
   inFocus_(false),
   rawPiezo_(0),
   offsetDevice_(0),
   autofocusDevice_(0)
{
   InitializeDefaultErrorMessages();

   // create pre-initialization properties
   // ------------------------------------

   // Port
   CPropertyAction* pAct = new CPropertyAction(this, &PureFocusHub::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

   CreateProperty("Center Piezo", "Yes", MM::String, false, 0, true);
   AddAllowedValue("Center Piezo", "Yes");
   AddAllowedValue("Center Piezo", "No");
}

PureFocusHub::~PureFocusHub()
{
   if (initialized_)
      Shutdown();
}

int PureFocusHub::DetectInstalledDevices()
{
   ClearInstalledDevices();
   // make sure this method is called before we look for available devices
   InitializeModuleData();

   char hubName[MM::MaxStrLength];
   GetName(hubName); // this device name
   for (unsigned i=0; i<GetNumberOfDevices(); i++)
   { 
      char deviceName[MM::MaxStrLength];
      bool success = GetDeviceName(i, deviceName, MM::MaxStrLength);
      if (success && (strcmp(hubName, deviceName) != 0))
      {
         MM::Device* pDev = CreateDevice(deviceName);
         AddInstalledDevice(pDev);
      }
   }

   return DEVICE_OK;
}

void PureFocusHub::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}

int PureFocusHub::Initialize()
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

   MMThreadGuard guard(lock_);
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

   // version 
   CPropertyAction* pAct = new CPropertyAction(this, &PureFocusHub::OnVersion);
   ret = CreateProperty("Version", version_.c_str(), MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // build date
   pAct = new CPropertyAction(this, &PureFocusHub::OnBuildDate);
   ret = CreateProperty("BuildDate", date_.c_str(), MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Piezo or Stepper (or Measure???)
   pAct = new CPropertyAction(this, &PureFocusHub::OnFocusControl);
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

      pAct = new CPropertyAction(this, &PureFocusHub::OnPiezoPosition);
      ret = CreateProperty("PiezoPosition", "0.0", MM::Float, false, pAct);
      if (ret != DEVICE_OK)
         return ret;

      SetPropertyLimits("PiezoPosition", 0.0, piezoRange_);

      char value[MM::MaxStrLength];
      GetProperty("Center Piezo", value);
      if (strcmp(value, "Yes") == 0)
      {
         SetProperty("PiezoPosition", std::to_string(piezoRange_ / 2).c_str());
      }

   }

   stopThread_ = false;
   readerThread_ = std::thread(&PureFocusHub::Updater, this);

   initialized_ = true;
   return DEVICE_OK;
}


int PureFocusHub::Shutdown()
{
   if (initialized_)
   {
      stopThread_ = true;
      readerThread_.join();
      MMThreadGuard guard(deviceLock_);
      if (autofocusDevice_ != 0)
      {
         autofocusDevice_->RemoveHub();
         autofocusDevice_ = 0;
      }
      if (offsetDevice_ != 0)
      {
         offsetDevice_->RemoveHub();
         offsetDevice_ = 0;
      }
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool PureFocusHub::Busy()
{
   // For more complex implementations, you might want to query the device status
   // to determine if it's busy, rather than relying on a flag
   // MMThreadGuard guard(lock_);
   return busy_;
}

int PureFocusHub::SetServo(bool state)
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

int PureFocusHub::GetServo(bool& state)
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

bool PureFocusHub::IsInFocus()
{
   bool inFocus;
   {
      MMThreadGuard guard(lock_);
      int ret = SendSerialCommand(port_.c_str(), "FOCUS", "\r");
      if (ret != DEVICE_OK)
         return false;

      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return false;

      // Parse response (should be "0" or "1")
      inFocus = resp == "1";
   }
   if (inFocus_ != inFocus)
   {
      inFocus_ = inFocus;
      MMThreadGuard guard(deviceLock_);
      if (autofocusDevice_ != 0)
         autofocusDevice_->CallbackInFocus(inFocus);
   }

   return inFocus_;
}

bool PureFocusHub::IsSampleDetected()
{
   string resp;
   {
      MMThreadGuard guard(lock_);
      int ret = SendSerialCommand(port_.c_str(), "SAMPLE", "\r");
      if (ret != DEVICE_OK)
         return false;

      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return false;
   }

   // Parse response (should be "0" or "1")
   bool isSampleDetected = resp == "1";
   if (inRange_ != isSampleDetected)
   {
      inRange_ = isSampleDetected;
      MMThreadGuard guard(deviceLock_);
      if (autofocusDevice_ != 0)
         autofocusDevice_->CallbackSampleDetected(isSampleDetected);
   }

   return isSampleDetected;
}


int PureFocusHub::GetOffset(long& offset)
{
   string resp;
   {
      MMThreadGuard guard(lock_);
      int ret = SendSerialCommand(port_.c_str(), "LENSP", "\r");
      if (ret != DEVICE_OK)
         return ret;

      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;
   }

   // Try to parse the response as a number
   try {
      offset = std::stoi(resp);
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }
   if (offset_ != offset) 
   {
      offset_ = offset;
      MMThreadGuard guard(deviceLock_);
      if (offsetDevice_ != 0)
         offsetDevice_->CallbackPositionSteps(offset);
   }

   return DEVICE_OK;
}

int PureFocusHub::SetOffset(long offset)
{
   MMThreadGuard guard(lock_);

   ostringstream cmd;
   cmd << "LENSG," << offset; // Set offset command with parameter
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


int PureFocusHub::GetFocusScore(double& score)
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


bool PureFocusHub::IsOffsetLensBusy()
{
   MMThreadGuard guard(lock_);
   int ret = SendSerialCommand(port_.c_str(), "LENS$", "\r");
   if (ret != DEVICE_OK)
      return ret;

   string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   // Try to parse the response as a number
   int result = 0;
   try {
      result = std::stoi(resp);
   }
   catch (std::exception&) {
      return false;
   }
   return result == 1; // 1 == moving, 0 == idle

}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int PureFocusHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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



int PureFocusHub::OnPiezoPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double position;
      int ret = GetPiezoPosition(position);
      if (ret != DEVICE_OK)
         return ret;
      pProp->Set(position);
   }
   else if (eAct == MM::AfterSet)
   {
      double val;
      pProp->Get(val);
      if (val >= 0.0 && val <= piezoRange_)
      {
         return SetPiezoPosition(val);
      }

   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Communication methods
///////////////////////////////////////////////////////////////////////////////

int PureFocusHub::GetResponse(string& resp)
{
   resp.clear();
   int ret = GetSerialAnswer(port_.c_str(), "\r", resp);
   if (ret != DEVICE_OK)
      return ret;
   // needed?
   return PurgeComPort(port_.c_str());
}

// Not needed anymore since we handle CR directly in GetResponse
string PureFocusHub::RemoveLineEndings(string input)
{
   string output = input;
   output.erase(remove(output.begin(), output.end(), '\r'), output.end());
   output.erase(remove(output.begin(), output.end(), '\n'), output.end());
   return output;
}



int PureFocusHub::OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(version_.c_str());
   }
   return DEVICE_OK;
}

int PureFocusHub::OnBuildDate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(date_.c_str());
   }
   return DEVICE_OK;
}

int PureFocusHub::OnFocusControl(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      MMThreadGuard guard(lock_);
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

int PureFocusHub::GetPinholeProperties(int& pinholeColumns, int& pinholeWidth)
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
      pinholeColumns = std::stoi(resp.substr(0, commaPos));
      pinholeWidth = std::stoi(resp.substr(commaPos + 1));
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }

   return DEVICE_OK;
}

int PureFocusHub::SetPinholeProperties(int columns, int width)
{
   MMThreadGuard guard(lock_);

   ostringstream cmd;
   cmd << "PINHOLE," << columns << "," << width;
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

   return DEVICE_OK;
}

int PureFocusHub::GetLaserPower(int& laserPower)
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
      laserPower = std::stoi(resp.substr(commaPos + 1));
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }

   return DEVICE_OK;
}


int PureFocusHub::SetLaserPower(int power)
{
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
   return DEVICE_OK;
}

int PureFocusHub::GetObjective(int& objective)
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
      objective = std::stoi(resp);
   }
   catch (std::exception&) {
      return ERR_UNEXPECTED_RESPONSE;
   }

   return DEVICE_OK;
}

int PureFocusHub::SetObjective(int objective)
{
   MMThreadGuard guard(lock_);

   ostringstream cmd;
   cmd << "OBJ," << objective;
   int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r");
   if (ret != DEVICE_OK)
      return ret;

   // Get response to confirm command was executed
   std::string resp;
   ret = GetResponse(resp);
   if (ret != DEVICE_OK)
      return ret;

   if (resp != "0") // Check if the response matches expected success code
      return ERR_UNEXPECTED_RESPONSE;

   return DEVICE_OK;
}

int PureFocusHub::GetPiezoPosition(double& um)
{
   int raw;
   {
      MMThreadGuard guard(lock_);
      int ret = SendSerialCommand(port_.c_str(), "PIEZO", "\r");
      if (ret != DEVICE_OK)
         return ret;

      string resp;
      ret = GetResponse(resp);
      if (ret != DEVICE_OK)
         return ret;

      raw = std::stoi(resp);
      if (raw >= 0 && raw <= 4095)
         um = (float)((float)raw / 4095.0 * piezoRange_);
      else
         return ERR_COMMUNICATION;
   }
   if (raw != rawPiezo_)
   {
      rawPiezo_ = raw;
      GetCoreCallback()->OnPropertyChanged(this, "PiezoPosition", std::to_string(um).c_str());
   }
   return DEVICE_OK;
}

int PureFocusHub::SetPiezoPosition(double um)
{
   int step = (int)(um / piezoRange_ * 4095);
   std::ostringstream os;
   os << "PIEZO," << step;
   string resp;
   int ret;
   {
      MMThreadGuard guard(lock_);
      ret = SendSerialCommand(port_.c_str(), os.str().c_str(), "\r");
      if (ret != DEVICE_OK)
         return ret;

      ret = GetResponse(resp);
   }
   if (ret != DEVICE_OK)
      return ret;

   if (resp != "0")
      return ERR_COMMUNICATION;

   return DEVICE_OK;
}

int PureFocusHub::GetList(std::string& list)
{
   MMThreadGuard guard(lock_);
   int ret = SendSerialCommand(port_.c_str(), "LIST", "\r");
   if (ret != DEVICE_OK)
      return ret;
   std::string answer = "";
   std::ostringstream result;
   while (answer != "END")
   {
      ret = GetSerialAnswer(port_.c_str(), "\r", answer);
      if (ret != DEVICE_OK)
         return ret;
      result << answer << "\r\n";
   }
   list = result.str();
   return DEVICE_OK;
}

size_t PureFocusHub::findNthChar(const std::string& str, char targetChar, int n) {
   size_t pos = 0;
   for (int i = 0; i < n; i++) {
      pos = str.find(targetChar, pos + (i > 0));
      if (pos == std::string::npos) 
         return pos;
   }
   return pos;
}


////////////////////////////////////////////////////////////////////////////////////
// Reader Thread
///////////////////////////////////////////////////////////////////////////////////
void PureFocusHub::Updater() 
{
   long offset;
   double pos;
   while (!stopThread_)
   {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      GetOffset(offset);
      if (IsSampleDetected())
         IsInFocus();
      GetPiezoPosition(pos);
   }

}
