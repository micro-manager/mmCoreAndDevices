///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentIX85XYStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 XY Stage device implementation (IX5-SSA hardware)
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

#include "EvidentIX85XYStage.h"
#include "EvidentIX85XYStageProtocol.h"
#include "EvidentIX85XYStageModel.h"
#include "ModuleInterface.h"
#include <sstream>
#include <cmath>
#include <chrono>

using namespace IX85XYStage;

const char* g_DeviceName = "IX85_XYStage";

// Property names
const char* g_PropertyPort = "Port";
const char* g_PropertySpeed = "Speed";
const char* g_PropertyJogEnable = "JOG Enable";
const char* g_PropertyJogSensitivity = "JOG Sensitivity";
const char* g_PropertyJogDirectionX = "JOG Direction X";
const char* g_PropertyJogDirectionY = "JOG Direction Y";

// Property values
const char* g_Yes = "Yes";
const char* g_No = "No";
const char* g_Normal = "Normal";
const char* g_Reverse = "Reverse";
///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceName, MM::XYStageDevice, "Evident IX85 XY Stage");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == nullptr)
      return nullptr;

   if (strcmp(deviceName, g_DeviceName) == 0)
      return new EvidentIX85XYStage();

   return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage Constructor
///////////////////////////////////////////////////////////////////////////////

EvidentIX85XYStage::EvidentIX85XYStage() :
   initialized_(false),
   port_("Undefined"),
   name_(g_DeviceName),
   stepSizeXUm_(0.078125),  // 78.125nm per step
   stepSizeYUm_(0.078125),
   stopMonitoring_(false),
   responseReady_(false)
{
   InitializeDefaultErrorMessages();

   // Add custom error messages
   SetErrorText(ERR_PORT_NOT_SET, "Port not set. Use the Port property to select a serial port.");
   SetErrorText(ERR_COMMAND_TIMEOUT, "Command timeout - no response from device");
   SetErrorText(ERR_NEGATIVE_ACK, "Device returned negative acknowledgment");
   SetErrorText(ERR_INVALID_RESPONSE, "Invalid response from device");

   // Pre-initialization property: Port
   CPropertyAction* pAct = new CPropertyAction(this, &EvidentIX85XYStage::OnPort);
   CreateProperty(g_PropertyPort, "Undefined", MM::String, false, pAct, true);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage Destructor
///////////////////////////////////////////////////////////////////////////////

EvidentIX85XYStage::~EvidentIX85XYStage()
{
   Shutdown();
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::GetName
///////////////////////////////////////////////////////////////////////////////

void EvidentIX85XYStage::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::Initialize
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   // Check if port is set
   if (port_ == "Undefined")
      return ERR_PORT_NOT_SET;

   // Purge any old data from the port
   PurgeComPort(port_.c_str());

   // CRITICAL: Start monitoring thread BEFORE sending any commands
   StartMonitoring();

   // Login to remote mode
   int ret = LoginToRemoteMode();
   if (ret != DEVICE_OK)
   {
      StopMonitoring();
      return ret;
   }

   // Query version
   ret = QueryVersion();
   if (ret != DEVICE_OK)
   {
      StopMonitoring();
      return ret;
   }

   // Verify this is an IX5-SSA device
   ret = VerifyDevice();
   if (ret != DEVICE_OK)
   {
      StopMonitoring();
      return ret;
   }

   // Initialize the stage (home it)
   ret = InitializeStage();
   if (ret != DEVICE_OK)
   {
      StopMonitoring();
      return ret;
   }

   // Enable XY Stage JOG operation
   ret = EnableJog(true);
   if (ret != DEVICE_OK)
   {
      StopMonitoring();
      return ret;
   }

   // Enable position notifications
   ret = EnableNotifications(true);
   if (ret != DEVICE_OK)
   {
      StopMonitoring();
      return ret;
   }

   // Set speed
   long initial, high, accel;
   model_.GetSpeed(initial, high, accel);
   std::string cmd = BuildCommand(CMD_XY_SPEED, (int)initial, (int)high, (int)accel);
   std::string response;
   ret = ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
   {
      StopMonitoring();
      return ret;
   }

   // Create post-initialization properties
   CPropertyAction* pAct = new CPropertyAction(this, &EvidentIX85XYStage::OnSpeed);
   ret = CreateProperty(g_PropertySpeed, "256000", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_PropertySpeed, 0, 512000);

   pAct = new CPropertyAction(this, &EvidentIX85XYStage::OnJogEnable);
   ret = CreateProperty(g_PropertyJogEnable, g_No, MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue(g_PropertyJogEnable, g_Yes);
   AddAllowedValue(g_PropertyJogEnable, g_No);

   pAct = new CPropertyAction(this, &EvidentIX85XYStage::OnJogSensitivity);
   ret = CreateProperty(g_PropertyJogSensitivity, "8", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_PropertyJogSensitivity, 1, 16);

   pAct = new CPropertyAction(this, &EvidentIX85XYStage::OnJogDirectionX);
   ret = CreateProperty(g_PropertyJogDirectionX, g_Normal, MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue(g_PropertyJogDirectionX, g_Normal);
   AddAllowedValue(g_PropertyJogDirectionX, g_Reverse);

   pAct = new CPropertyAction(this, &EvidentIX85XYStage::OnJogDirectionY);
   ret = CreateProperty(g_PropertyJogDirectionY, g_Normal, MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue(g_PropertyJogDirectionY, g_Normal);
   AddAllowedValue(g_PropertyJogDirectionY, g_Reverse);

   initialized_ = true;
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::Shutdown
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   // Disable position notifications first
   EnableNotifications(false);

   // Disable jog if enabled
   if (model_.IsJogEnabled())
   {
      EnableJog(false);
   }

   // Return to local mode
   std::string cmd = BuildCommand(CMD_LOGIN, 0);
   std::string response;
   ExecuteCommand(cmd, response);

   // Stop monitoring thread LAST (after all commands sent)
   StopMonitoring();

   initialized_ = false;
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::Busy
///////////////////////////////////////////////////////////////////////////////

bool EvidentIX85XYStage::Busy()
{
   // Check model state (updated by position notifications)
   return model_.IsBusy();
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::SetPositionSteps
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::SetPositionSteps(long x, long y)
{
   // Set target in model first
   model_.SetTarget(x, y);

   // Check if we're already at the target
   if (model_.IsAtTarget(XY_STAGE_POSITION_TOLERANCE))
   {
      // Already at target, no need to move
      model_.SetBusy(false);
      return DEVICE_OK;
   }

   // Build XYG command for absolute positioning
   std::string cmd = BuildCommand(CMD_XY_GOTO, x, y);

   model_.SetBusy(true);

   std::string response;
   int ret = ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
   {
      model_.SetBusy(false);
      return ret;
   }

   // Check for positive acknowledgment
   if (!IsPositiveAck(response, CMD_XY_GOTO))
   {
      model_.SetBusy(false);
      if (IsNegativeAck(response, CMD_XY_GOTO))
         return ERR_NEGATIVE_ACK;
      return ERR_INVALID_RESPONSE;
   }

   // Check again if already at target (for very short moves that complete instantly)
   if (model_.IsAtTarget(XY_STAGE_POSITION_TOLERANCE))
   {
      model_.SetBusy(false);
   }

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::GetPositionSteps
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::GetPositionSteps(long& x, long& y)
{
   // Get position from model (updated by position notifications)
   model_.GetPosition(x, y);
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::SetRelativePositionSteps
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::SetRelativePositionSteps(long x, long y)
{
   // Get current position from model
   long currentX, currentY;
   model_.GetPosition(currentX, currentY);

   // Calculate absolute target position
   long targetX = currentX + x;
   long targetY = currentY + y;

   // Set target in model
   model_.SetTarget(targetX, targetY);

   // Check if we're already at the target (e.g., relative move of 0,0)
   if (model_.IsAtTarget(XY_STAGE_POSITION_TOLERANCE))
   {
      // Already at target, no need to move
      model_.SetBusy(false);
      return DEVICE_OK;
   }

   // Use XYG (absolute positioning) for relative moves
   // Note: XYM on IX5-SSA appears to do absolute positioning, not relative
   std::string cmd = BuildCommand(CMD_XY_GOTO, targetX, targetY);

   model_.SetBusy(true);

   std::string response;
   int ret = ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
   {
      model_.SetBusy(false);
      return ret;
   }

   // Check for positive acknowledgment
   if (!IsPositiveAck(response, CMD_XY_GOTO))
   {
      model_.SetBusy(false);
      if (IsNegativeAck(response, CMD_XY_GOTO))
         return ERR_NEGATIVE_ACK;
      return ERR_INVALID_RESPONSE;
   }

   // Check again if already at target (for very short moves that complete instantly)
   if (model_.IsAtTarget(XY_STAGE_POSITION_TOLERANCE))
   {
      model_.SetBusy(false);
   }

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::Home
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::Home()
{
   return InitializeStage();
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::Stop
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::Stop()
{
   std::string response;
   int ret = ExecuteCommand(CMD_XY_STOP, response);
   if (ret == DEVICE_OK)
   {
      model_.SetBusy(false);
   }
   return ret;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::SetOrigin
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::SetOrigin()
{
   // The SSA doesn't have a "set origin" command, so we just note the current
   // position as 0,0 in software. This would require tracking an offset.
   // For now, return not supported.
   return DEVICE_UNSUPPORTED_COMMAND;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::GetLimitsUm
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax)
{
   std::pair<double, double> xy = ConvertPositionStepsToUm(XY_STAGE_MIN_POS_X, XY_STAGE_MIN_POS_Y);
   xMin = xy.first;
   yMin = xy.second;
   xy = ConvertPositionStepsToUm(XY_STAGE_MAX_POS_X, XY_STAGE_MAX_POS_Y);
   xMax = xy.first;
   yMax = xy.second;
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentIX85XYStage::GetStepLimits
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax)
{
   xMin = XY_STAGE_MIN_POS_X;
   xMax = XY_STAGE_MAX_POS_X;
   yMin = XY_STAGE_MIN_POS_Y;
   yMax = XY_STAGE_MAX_POS_Y;
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Property Handlers
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         pProp->Set(port_.c_str());
         return ERR_PORT_CHANGE_FORBIDDEN;
      }
      pProp->Get(port_);
   }
   return DEVICE_OK;
}

int EvidentIX85XYStage::OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      long initial, high, accel;
      model_.GetSpeed(initial, high, accel);
      pProp->Set(high);
   }
   else if (eAct == MM::AfterSet)
   {
      long speed;
      pProp->Get(speed);

      long initial, high, accel;
      model_.GetSpeed(initial, high, accel);
      high = speed;
      model_.SetSpeed(initial, high, accel);

      // Send XYSPD command
      std::string cmd = BuildCommand(CMD_XY_SPEED, (int)initial, (int)high, (int)accel);
      std::string response;
      int ret = ExecuteCommand(cmd, response);
      if (ret != DEVICE_OK)
         return ret;

      if (!IsPositiveAck(response, CMD_XY_SPEED))
      {
         if (IsNegativeAck(response, CMD_XY_SPEED))
            return ERR_NEGATIVE_ACK;
         return ERR_INVALID_RESPONSE;
      }
   }
   return DEVICE_OK;
}

int EvidentIX85XYStage::OnJogEnable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool enabled = model_.IsJogEnabled();
      pProp->Set(enabled ? g_Yes : g_No);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      bool enabled = (value == g_Yes);

      // Enable/disable jog
      int ret = EnableJog(enabled);
      if (ret != DEVICE_OK)
         return ret;

      // Enable/disable encoders
      std::string cmd;
      std::string response;
      if (enabled)
      {
         // Enable encoder 1 (X)
         cmd = BuildCommand(CMD_ENCODER_1, 1);
         ret = ExecuteCommand(cmd, response);
         if (ret != DEVICE_OK)
            return ret;

         // Enable encoder 2 (Y)
         cmd = BuildCommand(CMD_ENCODER_2, 1);
         ret = ExecuteCommand(cmd, response);
         if (ret != DEVICE_OK)
            return ret;
      }
      else
      {
         // Disable encoders
         cmd = BuildCommand(CMD_ENCODER_1, 0);
         ExecuteCommand(cmd, response);

         cmd = BuildCommand(CMD_ENCODER_2, 0);
         ExecuteCommand(cmd, response);
      }
   }
   return DEVICE_OK;
}

int EvidentIX85XYStage::OnJogSensitivity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      int sensitivity = model_.GetJogSensitivity();
      pProp->Set((long)sensitivity);
   }
   else if (eAct == MM::AfterSet)
   {
      long sensitivity;
      pProp->Get(sensitivity);
      model_.SetJogSensitivity((int)sensitivity);

      std::string cmd = BuildCommand(CMD_XY_JOG_SENSITIVITY, (int)sensitivity);
      std::string response;
      int ret = ExecuteCommand(cmd, response);
      if (ret != DEVICE_OK)
         return ret;

      if (!IsPositiveAck(response, CMD_XY_JOG_SENSITIVITY))
      {
         if (IsNegativeAck(response, CMD_XY_JOG_SENSITIVITY))
            return ERR_NEGATIVE_ACK;
         return ERR_INVALID_RESPONSE;
      }
   }
   return DEVICE_OK;
}

int EvidentIX85XYStage::OnJogDirectionX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool reverse = model_.GetJogDirectionX();
      pProp->Set(reverse ? g_Reverse : g_Normal);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      bool reverse = (value == g_Reverse);
      model_.SetJogDirectionX(reverse);

      int direction = reverse ? 0 : 1;
      std::string cmd = BuildCommand(CMD_XY_JOG_DIR_X, direction);
      std::string response;
      int ret = ExecuteCommand(cmd, response);
      if (ret != DEVICE_OK)
         return ret;

      if (!IsPositiveAck(response, CMD_XY_JOG_DIR_X))
      {
         if (IsNegativeAck(response, CMD_XY_JOG_DIR_X))
            return ERR_NEGATIVE_ACK;
         return ERR_INVALID_RESPONSE;
      }
   }
   return DEVICE_OK;
}

int EvidentIX85XYStage::OnJogDirectionY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool reverse = model_.GetJogDirectionY();
      pProp->Set(reverse ? g_Reverse : g_Normal);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      bool reverse = (value == g_Reverse);
      model_.SetJogDirectionY(reverse);

      int direction = reverse ? 0 : 1;
      std::string cmd = BuildCommand(CMD_XY_JOG_DIR_Y, direction);
      std::string response;
      int ret = ExecuteCommand(cmd, response);
      if (ret != DEVICE_OK)
         return ret;

      if (!IsPositiveAck(response, CMD_XY_JOG_DIR_Y))
      {
         if (IsNegativeAck(response, CMD_XY_JOG_DIR_Y))
            return ERR_NEGATIVE_ACK;
         return ERR_INVALID_RESPONSE;
      }
   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Threading Methods
///////////////////////////////////////////////////////////////////////////////

void EvidentIX85XYStage::StartMonitoring()
{
   stopMonitoring_ = false;
   monitorThread_ = std::thread(&EvidentIX85XYStage::MonitorThreadFunc, this);
}

void EvidentIX85XYStage::StopMonitoring()
{
   stopMonitoring_ = true;
   if (monitorThread_.joinable())
      monitorThread_.join();
}

void EvidentIX85XYStage::MonitorThreadFunc()
{
   std::string message;

   while (!stopMonitoring_)
   {
      // Read one byte at a time from serial port
      unsigned char byte;
      unsigned long bytesRead = 0;
      int ret = ReadFromComPort(port_.c_str(), &byte, 1, bytesRead);

      if (ret != DEVICE_OK || bytesRead == 0)
      {
         // No data available, sleep briefly to avoid busy waiting
         CDeviceUtils::SleepMs(10);
         continue;
      }

      // Build message until we get a newline
      if (byte == '\n')
      {
         // Remove any trailing \r
         if (!message.empty() && message[message.length() - 1] == '\r')
            message.erase(message.length() - 1);

         if (!message.empty())
         {
            // Check if this is a notification or a command response
            if (IsNotificationTag(message))
            {
               // Process notification directly
               ProcessNotification(message);
            }
            else
            {
               // Signal waiting command thread
               {
                  std::lock_guard<std::mutex> lock(responseMutex_);
                  pendingResponse_ = message;
                  responseReady_ = true;
               }
               responseCV_.notify_one();
            }

            message.clear();
         }
      }
      else if (byte != '\r')
      {
         message += static_cast<char>(byte);
      }
   }
}

void EvidentIX85XYStage::ProcessNotification(const std::string& notification)
{
   std::vector<std::string> params = ParseParameters(notification);

   // Handle position notification (NXYP)
   if (notification.find("NXYP ") == 0)
   {
      if (params.size() >= 2)
      {
         // Parse position values
         // Note: Negative positions are valid (stage range is -743680 to 743680)
         // ParseLongParameter returns -1 for invalid/unparseable values
         std::string xStr = params[0];
         std::string yStr = params[1];

         // Check for invalid position markers (X, x, +, !, etc.)
         if (xStr.empty() || yStr.empty() ||
             xStr == "X" || xStr == "x" || xStr == "+" || xStr == "!" ||
             yStr == "X" || yStr == "x" || yStr == "+" || yStr == "!")
         {
            return;  // Invalid position, skip
         }

         long x = ParseLongParameter(xStr);
         long y = ParseLongParameter(yStr);

         model_.SetPosition(x, y);

         // Check if we've reached the target
         if (model_.IsAtTarget(XY_STAGE_POSITION_TOLERANCE))
         {
            model_.SetBusy(false);
         }

         // Notify core of position change
         std::pair<double, double> xy = ConvertPositionStepsToUm(x, y);
         OnXYStagePositionChanged(xy.first, xy.second);
      }
      return;
   }

   // Handle encoder 1 notification (E1)
   if (notification.find("E1 ") == 0)
   {
      if (params.size() >= 1)
      {
         int delta = ParseIntParameter(params[0]);
         if (delta != 0)
         {
            // Calculate new encoder position with wrapping
            long currentPos = model_.GetEncoderPosition(1);
            long newPos = currentPos + delta;

            // Wrap encoder position (assuming 0-99 range)
            while (newPos < 0)
               newPos += 100;
            while (newPos >= 100)
               newPos -= 100;

            model_.SetEncoderPosition(1, newPos);

            // Move stage in X direction based on delta and jog settings
            int sensitivity = model_.GetJogSensitivity();
            bool reverse = model_.GetJogDirectionX();
            long moveAmount = delta * sensitivity;
            if (reverse)
               moveAmount = -moveAmount;

            // Get current position and calculate new position
            long x, y;
            model_.GetPosition(x, y);
            long newX = x + moveAmount;

            // Clamp to limits
            long minX, maxX;
            model_.GetLimitsX(minX, maxX);
            if (newX < minX) newX = minX;
            if (newX > maxX) newX = maxX;

            // Send move command (don't wait for response to avoid blocking monitor thread)
            std::string cmd = BuildCommand(CMD_XY_GOTO, newX, y);
            SendCommand(cmd);
            model_.SetTarget(newX, y);
            model_.SetBusy(true);
         }
      }
      return;
   }

   // Handle encoder 2 notification (E2)
   if (notification.find("E2 ") == 0)
   {
      if (params.size() >= 1)
      {
         int delta = ParseIntParameter(params[0]);
         if (delta != 0)
         {
            // Calculate new encoder position with wrapping
            long currentPos = model_.GetEncoderPosition(2);
            long newPos = currentPos + delta;

            // Wrap encoder position (assuming 0-99 range)
            while (newPos < 0)
               newPos += 100;
            while (newPos >= 100)
               newPos -= 100;

            model_.SetEncoderPosition(2, newPos);

            // Move stage in Y direction based on delta and jog settings
            int sensitivity = model_.GetJogSensitivity();
            bool reverse = model_.GetJogDirectionY();
            long moveAmount = delta * sensitivity;
            if (reverse)
               moveAmount = -moveAmount;

            // Get current position and calculate new position
            long x, y;
            model_.GetPosition(x, y);
            long newY = y + moveAmount;

            // Clamp to limits
            long minY, maxY;
            model_.GetLimitsY(minY, maxY);
            if (newY < minY) newY = minY;
            if (newY > maxY) newY = maxY;

            // Send move command (don't wait for response to avoid blocking monitor thread)
            std::string cmd = BuildCommand(CMD_XY_GOTO, x, newY);
            SendCommand(cmd);
            model_.SetTarget(x, newY);
            model_.SetBusy(true);
         }
      }
      return;
   }
}

///////////////////////////////////////////////////////////////////////////////
// Command/Response Pattern
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::ExecuteCommand(const std::string& cmd, std::string& response, long timeoutMs)
{
   // Lock to ensure only one command at a time
   std::lock_guard<std::mutex> cmdLock(commandMutex_);

   // Clear any pending response
   {
      std::lock_guard<std::mutex> respLock(responseMutex_);
      responseReady_ = false;
      pendingResponse_.clear();
   }

   // Send command
   int ret = SendCommand(cmd);
   if (ret != DEVICE_OK)
      return ret;

   // Wait for response
   ret = GetResponse(response, timeoutMs);
   return ret;
}

///////////////////////////////////////////////////////////////////////////////
// Private Helper Methods
///////////////////////////////////////////////////////////////////////////////

int EvidentIX85XYStage::SendCommand(const std::string& cmd)
{
   int ret = SendSerialCommand(port_.c_str(), cmd.c_str(), TERMINATOR);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}

int EvidentIX85XYStage::GetResponse(std::string& response, long timeoutMs)
{
   std::unique_lock<std::mutex> lock(responseMutex_);

   // Wait for response with timeout
   bool received = responseCV_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
      [this] { return responseReady_; });

   if (!received || !responseReady_)
      return ERR_COMMAND_TIMEOUT;

   response = pendingResponse_;
   responseReady_ = false;
   pendingResponse_.clear();

   return DEVICE_OK;
}

MM::DeviceDetectionStatus EvidentIX85XYStage::DetectDevice(void)
{
   if (initialized_)
      return MM::CanCommunicate;

   // All conditions must be satisfied...
   MM::DeviceDetectionStatus result = MM::Misconfigured;
   char answerTO[MM::MaxStrLength];

   try
   {
      std::string portLowerCase = port_;
      for (std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its)
      {
         *its = (char)tolower(*its);
      }
      if (0 < portLowerCase.length() && 0 != portLowerCase.compare("undefined") && 0 != portLowerCase.compare("unknown"))
      {
         result = MM::CanNotCommunicate;
         // Record current port settings
         GetCoreCallback()->GetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);

         // Device specific default communication parameters
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "115200");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_StopBits, "2");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_Parity, "Even");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "Verbose", "0");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", "5000.0");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "DelayBetweenCharsMs", "0");
         MM::Device* pS = GetCoreCallback()->GetDevice(this, port_.c_str());
         pS->Initialize();

         // Try to query the unit name
         PurgeComPort(port_.c_str());
         std::string response;
         std::string cmd = "U?";
         int ret = SendSerialCommand(port_.c_str(), cmd.c_str(), TERMINATOR);
         if (ret == DEVICE_OK)
         {
            ret = GetSerialAnswer(port_.c_str(), TERMINATOR, response);
            if (ret == DEVICE_OK && response.find("IX5-SSA") != std::string::npos)
            {
               result = MM::CanCommunicate;
            }
         }

         pS->Shutdown();
         // Always restore the AnswerTimeout to the default
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);
      }
   }
   catch (...)
   {
      LogMessage("Exception in DetectDevice!", false);
   }

   return result;
}

int EvidentIX85XYStage::LoginToRemoteMode()
{
   std::string cmd = BuildCommand(CMD_LOGIN, 1);
   std::string response;
   int ret = ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
      return ret;

   if (!IsPositiveAck(response, CMD_LOGIN))
   {
      if (IsNegativeAck(response, CMD_LOGIN))
         return ERR_NEGATIVE_ACK;
      return ERR_INVALID_RESPONSE;
   }

   return DEVICE_OK;
}

int EvidentIX85XYStage::QueryVersion()
{
   std::string cmd = BuildCommand(CMD_VERSION, 1);
   std::string response;
   int ret = ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
      return ret;

   // Parse version from response
   std::vector<std::string> params = ParseParameters(response);
   if (params.size() > 0)
   {
      model_.SetVersion(params[0]);
   }

   return DEVICE_OK;
}

int EvidentIX85XYStage::VerifyDevice()
{
   std::string cmd = BuildCommand(CMD_UNIT);
   cmd += "?";
   std::string response;
   int ret = ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
      return ret;

   // Response should contain "IX5-SSA"
   if (response.find("IX5-SSA") == std::string::npos)
      return ERR_DEVICE_NOT_AVAILABLE;

   return DEVICE_OK;
}

int EvidentIX85XYStage::InitializeStage()
{
   // Use XYINIT to initialize the stage
   // This can take a long time (several seconds), so use longer timeout
   std::string response;
   const long initTimeout = 30000;  // 30 seconds
   int ret = ExecuteCommand(CMD_XY_INIT, response, initTimeout);
   if (ret != DEVICE_OK)
      return ret;

   if (!IsPositiveAck(response, CMD_XY_INIT))
   {
      if (IsNegativeAck(response, CMD_XY_INIT))
         return ERR_NEGATIVE_ACK;
      return ERR_INVALID_RESPONSE;
   }

   return DEVICE_OK;
}

int EvidentIX85XYStage::EnableJog(bool enable)
{
   std::string cmd = BuildCommand(CMD_XY_JOG, enable ? 1 : 0);
   std::string response;
   int ret = ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
      return ret;

   if (!IsPositiveAck(response, CMD_XY_JOG))
   {
      if (IsNegativeAck(response, CMD_XY_JOG))
         return ERR_NEGATIVE_ACK;
      return ERR_INVALID_RESPONSE;
   }

   model_.SetJogEnabled(enable);
   return DEVICE_OK;
}

int EvidentIX85XYStage::EnableNotifications(bool enable)
{
   std::string cmd = BuildCommand(CMD_XY_POSITION_NOTIFY, enable ? 1 : 0);
   std::string response;
   int ret = ExecuteCommand(cmd, response);
   if (ret != DEVICE_OK)
      return ret;

   if (!IsPositiveAck(response, CMD_XY_POSITION_NOTIFY))
   {
      if (IsNegativeAck(response, CMD_XY_POSITION_NOTIFY))
         return ERR_NEGATIVE_ACK;
      return ERR_INVALID_RESPONSE;
   }

   return DEVICE_OK;
}
