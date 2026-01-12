///////////////////////////////////////////////////////////////////////////////
// FILE:          RappLasers.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Rapp Laser Controller adapter
// COPYRIGHT:     University of California, San Francisco, 2026
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
// AUTHOR:        Nico Stuurman, n.stuurman@ucsf.edu, 01/12/2026

#include "RappLasers.h"
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include <sstream>
#include <cstring>

const char* g_DeviceName = "RappLaser";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceName, MM::GenericDevice, "Rapp Laser Controller");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_DeviceName) == 0)
   {
      return new RappLaser();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// RappLaser implementation
///////////////////////////////////////////////////////////////////////////////

// Static members
MMThreadLock RappLaser::lock_;

/**
 * Constructor
 */
RappLaser::RappLaser() :
   port_("Undefined"),
   initialized_(false),
   answerTimeoutMs_(1000.0),
   shutterOpen_(false),
   lightOn_(false),
   intensityPercent_(0.0),
   pollingThread_(0)
{
   InitializeDefaultErrorMessages();

   // Custom error messages
   SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Cannot change port after initialization");
   SetErrorText(ERR_NO_PORT_SET, "Port property not set");
   SetErrorText(ERR_DEVICE_NOT_FOUND, "Rapp laser not found on specified port");
   SetErrorText(ERR_COMMUNICATION, "Serial communication timeout");
   SetErrorText(ERR_INVALID_RESPONSE, "Invalid response from laser");
   SetErrorText(ERR_COMMAND_FAILED, "Laser command failed");

   // Pre-initialization property: Port
   CPropertyAction* pAct = new CPropertyAction(this, &RappLaser::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

/**
 * Destructor
 */
RappLaser::~RappLaser()
{
   Shutdown();
}

/**
 * Get device name
 */
void RappLaser::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceName);
}

/**
 * Initialize the device
 */
int RappLaser::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   // Verify port is set
   if (port_ == "Undefined")
      return ERR_NO_PORT_SET;

   // Clear serial port
   int ret = PurgeComPort(port_.c_str());
   if (ret != DEVICE_OK)
      return ret;

   // Send initialization sequence
   ret = SendInitializationSequence();
   if (ret != DEVICE_OK)
   {
      LogMessage("Failed to send initialization sequence", false);
      return ret;
   }

   // Query device info
   ret = QuerySerialNumber(serialNumber_);
   if (ret != DEVICE_OK)
   {
      LogMessage("Failed to query serial number", false);
      return ERR_DEVICE_NOT_FOUND;
   }

   ret = QueryLaserName(laserName_);
   if (ret != DEVICE_OK)
   {
      LogMessage("Failed to query laser name", false);
      return ERR_DEVICE_NOT_FOUND;
   }

   // Log device info
   std::ostringstream msg;
   msg << "Connected to Rapp Laser: " << laserName_ << " (S/N: " << serialNumber_ << ")";
   LogMessage(msg.str().c_str(), false);

   // Create name and description
   ret = CreateProperty(MM::g_Keyword_Name, g_DeviceName, MM::String, true);
   if (ret != DEVICE_OK)
      return ret;

   ret = CreateProperty(MM::g_Keyword_Description, "Rapp Laser Controller", MM::String, true);
   if (ret != DEVICE_OK)
      return ret;

   // Create read-only properties
   CPropertyAction* pAct = new CPropertyAction(this, &RappLaser::OnSerialNumber);
   ret = CreateProperty("SerialNumber", serialNumber_.c_str(), MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   pAct = new CPropertyAction(this, &RappLaser::OnLaserName);
   ret = CreateProperty("LaserName", laserName_.c_str(), MM::String, true, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Create read-write properties
   pAct = new CPropertyAction(this, &RappLaser::OnShutter);
   ret = CreateProperty("Shutter", "Closed", MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue("Shutter", "Open");
   AddAllowedValue("Shutter", "Closed");

   pAct = new CPropertyAction(this, &RappLaser::OnLight);
   ret = CreateProperty("Light", "Off", MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue("Light", "On");
   AddAllowedValue("Light", "Off");

   pAct = new CPropertyAction(this, &RappLaser::OnIntensity);
   ret = CreateProperty("Intensity (%)", "0", MM::Float, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits("Intensity (%)", 0.0, 100.0);

   // Initialize cached state
   shutterOpen_ = false;
   lightOn_ = false;
   intensityPercent_ = 0.0;

   // Start polling thread
   pollingThread_ = new PollingThread(*this);
   pollingThread_->Start();

   initialized_ = true;
   return DEVICE_OK;
}

/**
 * Shutdown the device
 */
int RappLaser::Shutdown()
{
   if (initialized_)
   {
      // Stop polling thread first
      if (pollingThread_)
      {
         pollingThread_->Stop();
         delete pollingThread_;
         pollingThread_ = 0;
      }

      // Turn off light and close shutter
      SetLightState(false);
      SetShutterState(false);

      initialized_ = false;
   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Property action handlers
///////////////////////////////////////////////////////////////////////////////

/**
 * Port property handler
 */
int RappLaser::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         // Cannot change port after initialization
         pProp->Set(port_.c_str());
         return ERR_PORT_CHANGE_FORBIDDEN;
      }

      pProp->Get(port_);
   }

   return DEVICE_OK;
}

/**
 * Shutter property handler
 */
int RappLaser::OnShutter(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      MMThreadGuard guard(lock_);
      pProp->Set(shutterOpen_ ? "Open" : "Closed");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);

      bool open = (value == "Open");
      int ret = SetShutterState(open);
      if (ret != DEVICE_OK)
         return ret;
   }

   return DEVICE_OK;
}

/**
 * Light property handler
 */
int RappLaser::OnLight(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      MMThreadGuard guard(lock_);
      pProp->Set(lightOn_ ? "On" : "Off");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);

      bool on = (value == "On");
      int ret = SetLightState(on);
      if (ret != DEVICE_OK)
         return ret;
   }

   return DEVICE_OK;
}

/**
 * Intensity property handler
 */
int RappLaser::OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      MMThreadGuard guard(lock_);
      pProp->Set(intensityPercent_);
   }
   else if (eAct == MM::AfterSet)
   {
      double percent;
      pProp->Get(percent);

      int ret = SetIntensity(percent);
      if (ret != DEVICE_OK)
         return ret;
   }

   return DEVICE_OK;
}

/**
 * Serial number property handler
 */
int RappLaser::OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      MMThreadGuard guard(lock_);
      pProp->Set(serialNumber_.c_str());
   }

   return DEVICE_OK;
}

/**
 * Laser name property handler
 */
int RappLaser::OnLaserName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      MMThreadGuard guard(lock_);
      pProp->Set(laserName_.c_str());
   }

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Protocol implementation
///////////////////////////////////////////////////////////////////////////////

/**
 * Send initialization sequence
 */
int RappLaser::SendInitializationSequence()
{
   // First set of init commands (not including serial number and laser name queries)
   const unsigned char initCmds[] = {0x17, 0x11, 0x14, 0x16, 0x1A, 0x15};
   const int numCmds = sizeof(initCmds) / sizeof(initCmds[0]);

   // Clear port first
   PurgeComPort(port_.c_str());
   CDeviceUtils::SleepMs(50);

   for (int i = 0; i < numCmds; i++)
   {
      int ret = SendCommand(initCmds[i]);
      if (ret != DEVICE_OK)
         return ret;

      // Small delay between commands
      CDeviceUtils::SleepMs(50);

      // Read and discard response (these are short responses)
      unsigned char response[256];
      unsigned long read = 0;
      ReadResponse(response, sizeof(response), read);

      // Log response if debug logging enabled
      if (read > 0)
      {
         std::ostringstream msg;
         msg << "Init cmd 0x" << std::hex << (int)initCmds[i] << " response (" << std::dec << read << " bytes): ";
         for (unsigned long j = 0; j < read && j < 20; j++)
         {
            msg << "0x" << std::hex << (int)response[j] << " ";
         }
         LogMessage(msg.str().c_str(), true);
      }
   }

   // Now query serial number (0x19) - this returns a fixed 8-byte ASCII string
   std::string tempSerialNum;
   int ret = QuerySerialNumber(tempSerialNum);
   if (ret != DEVICE_OK)
   {
      LogMessage("Failed to query serial number during init", false);
      return ret;
   }
   LogMessage(("Init: Serial Number = " + tempSerialNum).c_str(), true);

   // Send 0x18 command
   ret = SendCommand(0x18);
   if (ret != DEVICE_OK)
      return ret;
   CDeviceUtils::SleepMs(50);
   unsigned char response[256];
   unsigned long read = 0;
   ReadResponse(response, sizeof(response), read);
   if (read > 0)
   {
      std::ostringstream msg;
      msg << "Init cmd 0x18 response (" << std::dec << read << " bytes): ";
      for (unsigned long j = 0; j < read && j < 20; j++)
      {
         msg << "0x" << std::hex << (int)response[j] << " ";
      }
      LogMessage(msg.str().c_str(), true);
   }

   // Query laser name (0x1C) - this returns a null-terminated string
   std::string tempLaserName;
   ret = QueryLaserName(tempLaserName);
   if (ret != DEVICE_OK)
   {
      LogMessage("Failed to query laser name during init", false);
      return ret;
   }
   LogMessage(("Init: Laser Name = " + tempLaserName).c_str(), true);

   return DEVICE_OK;
}

/**
 * Send single-byte command
 */
int RappLaser::SendCommand(unsigned char cmd)
{
   return WriteToComPort(port_.c_str(), &cmd, 1);
}

/**
 * Send command with single parameter
 */
int RappLaser::SendCommand(unsigned char cmd, unsigned char param)
{
   unsigned char buffer[2] = {cmd, param};
   return WriteToComPort(port_.c_str(), buffer, 2);
}

/**
 * Send command with multiple parameters
 */
int RappLaser::SendCommand(unsigned char cmd, const unsigned char* params, int paramLen)
{
   unsigned char buffer[256];
   buffer[0] = cmd;
   if (params != NULL && paramLen > 0)
   {
      memcpy(&buffer[1], params, paramLen);
   }

   int totalLen = 1 + paramLen;
   return WriteToComPort(port_.c_str(), buffer, totalLen);
}

/**
 * Read response from device
 */
int RappLaser::ReadResponse(unsigned char* response, int maxLen, unsigned long& bytesRead)
{
   double startTime = GetCurrentMMTime().getMsec();
   bytesRead = 0;

   while (bytesRead < (unsigned long)maxLen)
   {
      unsigned long read = 0;
      int ret = ReadFromComPort(port_.c_str(), &response[bytesRead], maxLen - bytesRead, read);

      if (ret != DEVICE_OK)
         return ret;

      bytesRead += read;

      // Check timeout
      double elapsed = GetCurrentMMTime().getMsec() - startTime;
      if (elapsed > answerTimeoutMs_)
         break;

      // Small delay if no data yet
      if (read == 0)
         CDeviceUtils::SleepMs(1);
      else
         break; // Got some data, return it
   }

   return DEVICE_OK;
}

/**
 * Read ASCII string response with null terminator (for 0x19 and 0x1C commands)
 */
int RappLaser::ReadStringResponse(std::string& response)
{
   unsigned char buffer[256];
   memset(buffer, 0, sizeof(buffer));
   double startTime = GetCurrentMMTime().getMsec();
   unsigned long bytesRead = 0;
   bool foundTerminator = false;

   // Read until we find 0x00 terminator or timeout
   while (bytesRead < sizeof(buffer) - 1 && !foundTerminator)
   {
      unsigned long read = 0;
      int ret = ReadFromComPort(port_.c_str(), &buffer[bytesRead], 1, read);

      if (ret != DEVICE_OK)
         return ret;

      if (read > 0)
      {
         // Check if we found the null terminator
         if (buffer[bytesRead] == 0x00)
         {
            foundTerminator = true;
            break;
         }
         bytesRead += read;
      }

      // Check timeout
      double elapsed = GetCurrentMMTime().getMsec() - startTime;
      if (elapsed > answerTimeoutMs_)
         break;

      // Small delay if no data
      if (read == 0)
         CDeviceUtils::SleepMs(1);
   }

   // Convert to string, filtering printable characters
   response.clear();
   for (unsigned long i = 0; i < bytesRead; i++)
   {
      if (buffer[i] >= 0x20 && buffer[i] < 0x7F)
      {
         response += (char)buffer[i];
      }
   }

   return DEVICE_OK;
}

/**
 * Read fixed-length response, ensuring we get all bytes
 */
int RappLaser::ReadFixedResponse(unsigned char* response, int expectedLen, unsigned long& bytesRead)
{
   double startTime = GetCurrentMMTime().getMsec();
   bytesRead = 0;

   // Keep reading until we get all expected bytes or timeout
   while (bytesRead < (unsigned long)expectedLen)
   {
      unsigned long read = 0;
      int ret = ReadFromComPort(port_.c_str(), &response[bytesRead], expectedLen - bytesRead, read);

      if (ret != DEVICE_OK)
         return ret;

      bytesRead += read;

      // Check timeout
      double elapsed = GetCurrentMMTime().getMsec() - startTime;
      if (elapsed > answerTimeoutMs_)
      {
         std::ostringstream msg;
         msg << "ReadFixedResponse timeout: expected " << expectedLen << " bytes, got " << bytesRead;
         LogMessage(msg.str().c_str(), false);
         break;
      }

      // Small delay if no data yet
      if (read == 0)
         CDeviceUtils::SleepMs(1);
   }

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// High-level device commands
///////////////////////////////////////////////////////////////////////////////

/**
 * Set shutter state
 */
int RappLaser::SetShutterState(bool open)
{
   MMThreadGuard guard(lock_);

   unsigned char param = open ? 0x01 : 0x00;

   // Send command
   int ret = SendCommand(CMD_SHUTTER, param);
   if (ret != DEVICE_OK)
      return ERR_COMMUNICATION;

   // Read and validate fixed 2-byte response
   unsigned char response[2];
   unsigned long read = 0;
   ret = ReadFixedResponse(response, 2, read);

   if (ret != DEVICE_OK || read != 2)
   {
      std::ostringstream msg;
      msg << "Shutter command: invalid response length (expected 2, got " << read << ")";
      LogMessage(msg.str().c_str(), false);
      return ERR_COMMUNICATION;
   }

   if (response[0] != CMD_SHUTTER || response[1] != RESPONSE_OK)
   {
      std::ostringstream msg;
      msg << "Shutter command failed: response = 0x" << std::hex << (int)response[0] << " 0x" << (int)response[1];
      LogMessage(msg.str().c_str(), false);
      return ERR_INVALID_RESPONSE;
   }

   // Update cached state
   shutterOpen_ = open;

   return DEVICE_OK;
}

/**
 * Get shutter state
 */
int RappLaser::GetShutterState(bool& open)
{
   MMThreadGuard guard(lock_);
   open = shutterOpen_;
   return DEVICE_OK;
}

/**
 * Set light state
 */
int RappLaser::SetLightState(bool on)
{
   MMThreadGuard guard(lock_);

   unsigned char param = on ? 0x01 : 0x00;

   // Send command
   int ret = SendCommand(CMD_LIGHT, param);
   if (ret != DEVICE_OK)
      return ERR_COMMUNICATION;

   // Read and validate fixed 2-byte response
   unsigned char response[2];
   unsigned long read = 0;
   ret = ReadFixedResponse(response, 2, read);

   if (ret != DEVICE_OK || read != 2)
   {
      std::ostringstream msg;
      msg << "Light command: invalid response length (expected 2, got " << read << ")";
      LogMessage(msg.str().c_str(), false);
      return ERR_COMMUNICATION;
   }

   if (response[0] != CMD_LIGHT || response[1] != RESPONSE_OK)
   {
      std::ostringstream msg;
      msg << "Light command failed: response = 0x" << std::hex << (int)response[0] << " 0x" << (int)response[1];
      LogMessage(msg.str().c_str(), false);
      return ERR_INVALID_RESPONSE;
   }

   // Update cached state
   lightOn_ = on;

   return DEVICE_OK;
}

/**
 * Get light state
 */
int RappLaser::GetLightState(bool& on)
{
   MMThreadGuard guard(lock_);
   on = lightOn_;
   return DEVICE_OK;
}

/**
 * Set intensity (percentage 0-100)
 */
int RappLaser::SetIntensity(double intensityPercent)
{
   MMThreadGuard guard(lock_);

   // Convert percentage to raw value
   long rawIntensity = PercentageToRaw(intensityPercent);

   // Convert to 3-byte representation
   unsigned char params[4];
   params[0] = 0x00; // First byte always 0x00
   IntensityToBytes(rawIntensity, &params[1]);

   // Send command
   int ret = SendCommand(CMD_INTENSITY, params, 4);
   if (ret != DEVICE_OK)
      return ERR_COMMUNICATION;

   // Read and validate fixed 2-byte response
   unsigned char response[2];
   unsigned long read = 0;
   ret = ReadFixedResponse(response, 2, read);

   if (ret != DEVICE_OK || read != 2)
   {
      std::ostringstream msg;
      msg << "Intensity command: invalid response length (expected 2, got " << read << ")";
      LogMessage(msg.str().c_str(), false);
      return ERR_COMMUNICATION;
   }

   if (response[0] != CMD_INTENSITY || response[1] != RESPONSE_OK)
   {
      std::ostringstream msg;
      msg << "Intensity command failed: response = 0x" << std::hex << (int)response[0] << " 0x" << (int)response[1];
      LogMessage(msg.str().c_str(), false);
      return ERR_INVALID_RESPONSE;
   }

   // Update cached state (will be updated by polling thread shortly anyway)
   intensityPercent_ = intensityPercent;

   return DEVICE_OK;
}

/**
 * Get intensity
 */
int RappLaser::GetIntensity(double& intensityPercent)
{
   MMThreadGuard guard(lock_);
   intensityPercent = intensityPercent_;
   return DEVICE_OK;
}

/**
 * Query serial number (returns fixed 8-byte ASCII string)
 */
int RappLaser::QuerySerialNumber(std::string& serialNum)
{
   MMThreadGuard guard(lock_);

   // Send command
   int ret = SendCommand(CMD_SERIAL_NUMBER);
   if (ret != DEVICE_OK)
      return ret;

   // Read fixed 8-byte response (e.g., "230-2644")
   unsigned char buffer[8];
   unsigned long bytesRead = 0;
   ret = ReadFixedResponse(buffer, 8, bytesRead);
   if (ret != DEVICE_OK || bytesRead != 8)
   {
      std::ostringstream msg;
      msg << "QuerySerialNumber: expected 8 bytes, got " << bytesRead;
      LogMessage(msg.str().c_str(), false);
      return ERR_COMMUNICATION;
   }

   // Convert to string
   serialNum.clear();
   for (int i = 0; i < 8; i++)
   {
      if (buffer[i] >= 0x20 && buffer[i] < 0x7F)
         serialNum += (char)buffer[i];
   }

   return DEVICE_OK;
}

/**
 * Query laser name
 */
int RappLaser::QueryLaserName(std::string& laserName)
{
   MMThreadGuard guard(lock_);

   // Send command
   int ret = SendCommand(CMD_LASER_NAME);
   if (ret != DEVICE_OK)
      return ret;

   // Read string response
   ret = ReadStringResponse(laserName);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}

/**
 * Query status (called by polling thread)
 */
int RappLaser::QueryStatus()
{
   MMThreadGuard guard(lock_);

   // Send 0x40 command
   int ret = SendCommand(CMD_STATUS);
   if (ret != DEVICE_OK)
      return ret;

   // Read fixed 9-byte response
   unsigned char response[9];
   unsigned long read = 0;
   ret = ReadFixedResponse(response, 9, read);

   if (ret != DEVICE_OK || read < 8)
      return ret; // Not fatal, will retry next poll

   // Validate response
   if (response[0] != CMD_STATUS)
      return ERR_INVALID_RESPONSE;

   // Parse state from response
   // Byte 2: Light state (0x00=off, 0x01=on)
   // Byte 3: Shutter state (0x00=closed, 0x01=open)
   // Bytes 5-7: Intensity (big-endian 3-byte value)
   lightOn_ = (response[2] == 0x01);
   shutterOpen_ = (response[3] == 0x01);

   // Parse intensity from bytes 5-7 (need at least 8 bytes for this)
   if (read >= 8)
   {
      long rawIntensity = BytesToIntensity(&response[5]);
      intensityPercent_ = RawToPercentage(rawIntensity);
   }

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Utility methods
///////////////////////////////////////////////////////////////////////////////

/**
 * Convert percentage to raw protocol value
 */
long RappLaser::PercentageToRaw(double percentage)
{
   if (percentage < 0)
      percentage = 0;
   if (percentage > 100)
      percentage = 100;

   return 1000 + (long)(percentage * 990.0);
}

/**
 * Convert raw protocol value to percentage
 */
double RappLaser::RawToPercentage(long rawValue)
{
   if (rawValue < 1000)
      rawValue = 1000;
   if (rawValue > 100000)
      rawValue = 100000;

   return (rawValue - 1000) / 990.0;
}

/**
 * Convert raw value to 3-byte big-endian representation
 */
void RappLaser::IntensityToBytes(long intensity, unsigned char* bytes)
{
   bytes[0] = (intensity >> 16) & 0xFF;
   bytes[1] = (intensity >> 8) & 0xFF;
   bytes[2] = intensity & 0xFF;
}

/**
 * Convert 3-byte big-endian to raw value
 */
long RappLaser::BytesToIntensity(const unsigned char* bytes)
{
   return ((long)bytes[0] << 16) | ((long)bytes[1] << 8) | bytes[2];
}

///////////////////////////////////////////////////////////////////////////////
// PollingThread implementation
///////////////////////////////////////////////////////////////////////////////

/**
 * Constructor
 */
PollingThread::PollingThread(RappLaser& device) :
   device_(device),
   stop_(false)
{
}

/**
 * Destructor
 */
PollingThread::~PollingThread()
{
   Stop();
   wait();
}

/**
 * Thread service method (main loop)
 */
int PollingThread::svc()
{
   while (!stop_)
   {
      // Query status (send 0x40 command)
      device_.QueryStatus();

      // Sleep 100ms between polls
      CDeviceUtils::SleepMs(100);
   }

   return 0;
}

/**
 * Start the thread
 */
void PollingThread::Start()
{
   stop_ = false;
   activate();
}
