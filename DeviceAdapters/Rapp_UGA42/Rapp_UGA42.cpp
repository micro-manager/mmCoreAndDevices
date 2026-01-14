///////////////////////////////////////////////////////////////////////////////
// FILE:          Rapp_UGA42.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Rapp UGA-42 Scanner adapter
//
// COPYRIGHT:     University of California, San Francisco
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER(S) OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Nico Stuurman, 2025
//
// Based on the Rapp UGA-40 adapter by Arthur Edelstein
//
// Runtime requirements: 
// obstools.dll
// "ROE UGA42 SDK.dll"


#include "Rapp_UGA42.h"
#include "ModuleInterface.h"
#include <sstream>
#include <algorithm>

const char* g_RappUGA42ScannerName = "RappUGA42Scanner";
const char* g_PropertyVirtualComPort = "VirtualComPort";

// Default laser parameters (conservative settings for continuous laser)
const UINT32 DEFAULT_DIGITAL_RISE_TIME = 1;      // 1 microsecond
const UINT32 DEFAULT_DIGITAL_FALL_TIME = 1;      // 1 microsecond
const UINT32 DEFAULT_ANALOG_CHANGE_TIME = 1;     // 1 microsecond
const UINT32 DEFAULT_MIN_INTENSITY = 0;          // 0%
const UINT32 DEFAULT_MAX_INTENSITY = 10000;      // 100%
const UINT32 DEFAULT_LASER_FREQUENCY = 0;        // Continuous laser
const UINT32 DEFAULT_TICK_TIME = 50;             // 50 microseconds
const int DEFAULT_SPOT_SIZE = 50;                // Device coordinates

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_RappUGA42ScannerName, MM::GalvoDevice, "Rapp UGA-42 Scanner");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_RappUGA42ScannerName) == 0)
   {
      return new RappUGA42Scanner();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// RappUGA42Scanner implementation
///////////////////////////////////////////////////////////////////////////////

RappUGA42Scanner::RappUGA42Scanner() :
   initialized_(false),
   port_(""),
   device_(nullptr),
   debugMode_(false),  // Default to fast mode
   laserID_(0),
   laserAdded_(false),
   laserPort_(RMIPort1),
   digitalRiseTime_(DEFAULT_DIGITAL_RISE_TIME),
   digitalFallTime_(DEFAULT_DIGITAL_FALL_TIME),
   analogChangeTime_(DEFAULT_ANALOG_CHANGE_TIME),
   minIntensity_(DEFAULT_MIN_INTENSITY),
   maxIntensity_(DEFAULT_MAX_INTENSITY),
   laserFrequency_(DEFAULT_LASER_FREQUENCY),
   currentIntensity_(5000),  // 50% default
   spotSize_(DEFAULT_SPOT_SIZE),
   tickTime_(DEFAULT_TICK_TIME),
   pulseTime_us_(1000.0),    // 1ms default
   scanMode_(ScanMode::Accurate),
   ttlTriggerPort_(InputPorts::None),
   ttlTriggerBehavior_(TriggerBehaviour::Rising),
   currentX_(0.0),
   currentY_(0.0),
   polygonRepetitions_(1),
   lastSequenceID_(0),
   sequenceRunning_(false),
   loadedPolygonSequenceID_(0),
   polygonsLoaded_(false),
   workerThread_(nullptr)
{
   // Pre-initialization properties
   // CPropertyAction* pAct = new CPropertyAction(this, &RappUGA42Scanner::OnPort);
   // CreateProperty(g_PropertyVirtualComPort, "", MM::String, false, pAct, true);

   // Debug Mode property (pre-initialization)
   CPropertyAction* pAct = new CPropertyAction(this, &RappUGA42Scanner::OnDebugMode);
   CreateProperty("DebugMode", "False", MM::String, false, pAct, true);
   AddAllowedValue("DebugMode", "True");
   AddAllowedValue("DebugMode", "False");

   InitializeDefaultErrorMessages();
   SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Port cannot be changed after initialization");
   SetErrorText(ERR_DEVICE_NOT_FOUND, "No UGA-42 device found");
   SetErrorText(ERR_CONNECTION_FAILED, "Failed to connect to UGA-42 device");
   SetErrorText(ERR_LASER_SETUP_FAILED, "Failed to configure laser");
   SetErrorText(ERR_SEQUENCE_UPLOAD_FAILED, "Failed to upload sequence to device");
   SetErrorText(ERR_SEQUENCE_START_FAILED, "Failed to start sequence");
   SetErrorText(ERR_INVALID_DEVICE_STATE, "Device is in invalid state for this operation");
   SetErrorText(ERR_MEMORY_OVERLOAD, "Device memory full");
   SetErrorText(ERR_SEQUENCE_INVALID, "Sequence is invalid");
}

RappUGA42Scanner::~RappUGA42Scanner()
{
   Shutdown();
}

void RappUGA42Scanner::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_RappUGA42ScannerName);
}

int RappUGA42Scanner::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   LogMessage("UGA42: Initialize() starting, debugMode=" + std::string(debugMode_ ? "true" : "false"), true);

   // Create device instance
   device_ = new UGA42(debugMode_);

   // Search for devices
   UINT32 deviceCount = device_->Search();
   if (deviceCount == 0)
   {
      delete device_;
      device_ = nullptr;
      return ERR_DEVICE_NOT_FOUND;
   }

   // If port not specified, get list of available ports
   if (port_.empty())
   {
      std::vector<std::string> availablePorts;
      for (UINT32 i = 0; i < deviceCount; i++)
      {
         DeviceInfo devInfo;
         if (device_->GetDeviceInfo(i, &devInfo) == RETCODE::OK)
         {
            availablePorts.push_back(std::string(devInfo.COMPort));
         }
      }
      if (availablePorts.empty())
      {
         delete device_;
         device_ = nullptr;
         return ERR_DEVICE_NOT_FOUND;
      }
      port_ = availablePorts[0];
   }

   // Connect to device
   LogMessage("UGA42: Connecting to port " + port_, true);
   UINT32 ret = device_->Connect(port_.c_str());
   if (ret != RETCODE::OK)
   {
      LogMessage("UGA42: Connect failed with code " + std::to_string(ret), true);
      delete device_;
      device_ = nullptr;
      return MapRetCode(ret);
   }
   LogMessage("UGA42: Connected successfully", true);

   // Verify connection
   State state;
   LogMessage("UGA42: Calling GetState to verify connection", true);
   ret = device_->GetState(&state);
   LogMessage("UGA42: GetState returned " + std::to_string(ret) + ", state=" + std::to_string(static_cast<int>(state)), true);
   if (ret != RETCODE::OK || state == State::Disconnected)
   {
      delete device_;
      device_ = nullptr;
      return ERR_CONNECTION_FAILED;
   }

   // Set tick time
   LogMessage("UGA42: Setting tick time to " + std::to_string(tickTime_), true);
   ret = device_->SetTickTime(tickTime_);
   if (ret != RETCODE::OK)
   {
      LogMessage("UGA42: SetTickTime failed with code " + std::to_string(ret), true);
      device_->Disconnect();
      delete device_;
      device_ = nullptr;
      return MapRetCode(ret);
   }
   LogMessage("UGA42: Tick time set successfully", true);

   // Add and configure laser
   LogMessage("UGA42: Adding laser", true);
   int result = AddLaser();
   if (result != DEVICE_OK)
   {
      device_->Disconnect();
      delete device_;
      device_ = nullptr;
      return result;
   }

   // Create post-initialization properties
   CPropertyAction* pAct;

   // Scan Mode
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnScanMode);
   CreateProperty("ScanMode", "Accurate", MM::String, false, pAct);
   AddAllowedValue("ScanMode", "Accurate");
   AddAllowedValue("ScanMode", "Fast");

   // Spot Size
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnSpotSize);
   CreateIntegerProperty("SpotSize", spotSize_, false, pAct);
   SetPropertyLimits("SpotSize", 1, 1000);

   // Laser Intensity
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnLaserIntensity);
   CreateIntegerProperty("LaserIntensity", currentIntensity_, false, pAct);
   SetPropertyLimits("LaserIntensity", minIntensity_, maxIntensity_);

   // TTL Trigger Mode
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnTTLTriggerMode);
   CreateProperty("TTLTriggerMode", "None", MM::String, false, pAct);
   AddAllowedValue("TTLTriggerMode", "None");
   AddAllowedValue("TTLTriggerMode", "Port1");
   AddAllowedValue("TTLTriggerMode", "Port2");
   AddAllowedValue("TTLTriggerMode", "Port1_Once");
   AddAllowedValue("TTLTriggerMode", "Port2_Once");

   // TTL Trigger Behavior
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnTTLTriggerBehavior);
   CreateProperty("TTLTriggerBehavior", "Rising", MM::String, false, pAct);
   AddAllowedValue("TTLTriggerBehavior", "Rising");
   AddAllowedValue("TTLTriggerBehavior", "Falling");

   // Laser Type
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnLaserType);
   CreateProperty("LaserType", "Continuous", MM::String, false, pAct);
   AddAllowedValue("LaserType", "Continuous");
   AddAllowedValue("LaserType", "Pulsed");

   // Laser Frequency (for pulsed lasers)
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnLaserFrequency);
   CreateIntegerProperty("LaserFrequency", laserFrequency_, false, pAct);
   SetPropertyLimits("LaserFrequency", 0, 100000);

   // Tick Time
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnTickTime);
   CreateIntegerProperty("TickTime", tickTime_, false, pAct);
   SetPropertyLimits("TickTime", 40, 1000);

   // Advanced laser parameters
   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnLaserPort);
   CreateProperty("LaserPort", "RMIPort1", MM::String, false, pAct);
   AddAllowedValue("LaserPort", "RMIPort1");
   AddAllowedValue("LaserPort", "RMIPort2");
   AddAllowedValue("LaserPort", "RMIPort3");
   AddAllowedValue("LaserPort", "RMIPort4");

   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnDigitalRiseTime);
   CreateIntegerProperty("DigitalRiseTime", digitalRiseTime_, false, pAct);
   SetPropertyLimits("DigitalRiseTime", 0, 10000);

   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnDigitalFallTime);
   CreateIntegerProperty("DigitalFallTime", digitalFallTime_, false, pAct);
   SetPropertyLimits("DigitalFallTime", 0, 10000);

   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnAnalogChangeTime);
   CreateIntegerProperty("AnalogChangeTime", analogChangeTime_, false, pAct);
   SetPropertyLimits("AnalogChangeTime", 0, 10000);

   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnMinIntensity);
   CreateIntegerProperty("MinIntensity", minIntensity_, false, pAct);
   SetPropertyLimits("MinIntensity", 0, 10000);

   pAct = new CPropertyAction(this, &RappUGA42Scanner::OnMaxIntensity);
   CreateIntegerProperty("MaxIntensity", maxIntensity_, false, pAct);
   SetPropertyLimits("MaxIntensity", 0, 10000);

   // Start worker thread AFTER all initialization is complete
   // This prevents race conditions between main thread SDK calls and worker keepalive
   LogMessage("UGA42: Starting worker thread", true);
   workerThread_ = new DeviceWorkerThread(*this);
   workerThread_->Start();

   initialized_ = true;
   LogMessage("UGA42: Initialize() completed successfully", true);
   return DEVICE_OK;
}

int RappUGA42Scanner::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   // Stop worker thread first
   if (workerThread_)
   {
      workerThread_->Stop();
      delete workerThread_;
      workerThread_ = nullptr;
   }

   if (device_ != nullptr)
   {
      // Stop any running sequences
      device_->Stop();

      // Delete all sequences from device memory (includes loaded polygons)
      device_->DeleteAllSequences();

      // Remove laser
      if (laserAdded_)
      {
         device_->RemoveLaser(laserID_);
         laserAdded_ = false;
      }

      // Disconnect
      device_->Disconnect();

      delete device_;
      device_ = nullptr;
   }

   // Reset polygon state
   loadedPolygonSequenceID_ = 0;
   polygonsLoaded_ = false;

   initialized_ = false;
   return DEVICE_OK;
}

int RappUGA42Scanner::UpdateLaser()
{
   if (laserAdded_)
   {
      device_->RemoveLaser(laserID_);
      laserAdded_ = false;
   }
   return AddLaser();
}

bool RappUGA42Scanner::Busy()
{
   if (!initialized_ || device_ == nullptr || workerThread_ == nullptr)
      return false;

   return workerThread_->IsBusy();
}

///////////////////////////////////////////////////////////////////////////////
// Galvo API Implementation
///////////////////////////////////////////////////////////////////////////////

int RappUGA42Scanner::PointAndFire(double x, double y, double pulseTime_us)
{
   if (!initialized_ || device_ == nullptr || workerThread_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   // Create command and enqueue it
   PointAndFireCommand* cmd = new PointAndFireCommand(x, y, pulseTime_us);
   workerThread_->EnqueueCommand(cmd);

   return DEVICE_OK;  // Return immediately
}

int RappUGA42Scanner::SetSpotInterval(double pulseTime_us)
{
   pulseTime_us_ = pulseTime_us;
   return DEVICE_OK;
}

int RappUGA42Scanner::SetPosition(double x, double y)
{
   if (!initialized_ || device_ == nullptr || workerThread_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   // Create command and enqueue it
   SetPositionCommand* cmd = new SetPositionCommand(x, y);
   workerThread_->EnqueueCommand(cmd);

   return DEVICE_OK;  // Return immediately
}

int RappUGA42Scanner::GetPosition(double& x, double& y)
{
   if (!initialized_ || device_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   // Return cached values instead of querying device
   // These are updated by SetPosition and PointAndFire commands
   x = currentX_;
   y = currentY_;

   return DEVICE_OK;
}

int RappUGA42Scanner::SetIlluminationState(bool on)
{
   if (!initialized_ || device_ == nullptr || workerThread_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   // Create command and enqueue it
   SetIlluminationCommand* cmd = new SetIlluminationCommand(on);
   workerThread_->EnqueueCommand(cmd);

   return DEVICE_OK;  // Return immediately
}

double RappUGA42Scanner::GetXRange()
{
   return MAXDeviceCoordinatesXY;
}

double RappUGA42Scanner::GetYRange()
{
   return MAXDeviceCoordinatesXY;
}

///////////////////////////////////////////////////////////////////////////////
// Polygon API Implementation
///////////////////////////////////////////////////////////////////////////////

int RappUGA42Scanner::AddPolygonVertex(int polygonIndex, double x, double y)
{
   if (polygonIndex < 0)
      return DEVICE_INVALID_PROPERTY_VALUE;

   // Ensure polygon vector is large enough
   if (static_cast<size_t>(polygonIndex) >= polygons_.size())
   {
      polygons_.resize(polygonIndex + 1);
   }

   RPOINTF point;
   point.x = static_cast<float>(x);
   point.y = static_cast<float>(y);
   polygons_[polygonIndex].push_back(point);

   return DEVICE_OK;
}

int RappUGA42Scanner::DeletePolygons()
{
   polygons_.clear();

   // Mark polygons as no longer loaded
   // Note: Don't delete the sequence here - let LoadPolygons() handle cleanup
   polygonsLoaded_ = false;

   return DEVICE_OK;
}

int RappUGA42Scanner::LoadPolygons()
{
   if (!initialized_ || device_ == nullptr || workerThread_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   if (polygons_.empty())
   {
      polygonsLoaded_ = false;
      return DEVICE_OK;
   }

   // Create command and enqueue it
   LoadPolygonsCommand* cmd = new LoadPolygonsCommand(polygons_);
   workerThread_->EnqueueCommand(cmd);

   return DEVICE_OK;  // Return immediately
}

int RappUGA42Scanner::SetPolygonRepetitions(int repetitions)
{
   if (repetitions < 0)
      return DEVICE_INVALID_PROPERTY_VALUE;

   polygonRepetitions_ = repetitions;
   return DEVICE_OK;
}

int RappUGA42Scanner::RunPolygons()
{
   if (!initialized_ || device_ == nullptr || workerThread_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   if (polygons_.empty())
      return DEVICE_OK;  // Nothing to do

   // Create command and enqueue it
   RunPolygonsCommand* cmd = new RunPolygonsCommand(polygons_, polygonRepetitions_);
   workerThread_->EnqueueCommand(cmd);

   return DEVICE_OK;  // Return immediately
}

int RappUGA42Scanner::RunSequence()
{
   // For now, this is a simplified implementation
   // Full sequence programming would require parsing string commands
   // and converting them to SequenceObject arrays
   return DEVICE_UNSUPPORTED_COMMAND;
}

int RappUGA42Scanner::StopSequence()
{
   if (!initialized_ || device_ == nullptr || workerThread_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   // Only stops RunPolygons sequences, not PointAndFire
   workerThread_->StopPolygonSequence();

   return DEVICE_OK;
}

int RappUGA42Scanner::GetChannel(char* channelName)
{
   CDeviceUtils::CopyLimitedString(channelName, "1");
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Property Handlers
///////////////////////////////////////////////////////////////////////////////

int RappUGA42Scanner::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         return ERR_PORT_CHANGE_FORBIDDEN;
      }
      pProp->Get(port_);
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnDebugMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(debugMode_ ? "True" : "False");
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         return ERR_PORT_CHANGE_FORBIDDEN;
      }

      std::string value;
      pProp->Get(value);
      debugMode_ = (value == "True");
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnScanMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(scanMode_ == ScanMode::Accurate ? "Accurate" : "Fast");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string mode;
      pProp->Get(mode);
      scanMode_ = (mode == "Accurate") ? ScanMode::Accurate : ScanMode::Fast;

      if (device_ != nullptr)
      {
         UINT32 ret = device_->SetScanMode(scanMode_);
         return MapRetCode(ret);
      }
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnSpotSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(spotSize_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      spotSize_ = static_cast<int>(value);

      if (device_ != nullptr && laserAdded_)
      {
         UINT32 ret = device_->SetLaserStepSize(laserID_, spotSize_);
         return MapRetCode(ret);
      }
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnLaserIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(currentIntensity_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      currentIntensity_ = static_cast<UINT32>(value);

      if (device_ != nullptr && laserAdded_ && IsDeviceIdle())
      {
         UINT32 ret = device_->SetLaserIntensity(laserID_, currentIntensity_);
         return MapRetCode(ret);
      }
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnTTLTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string mode;
      switch (ttlTriggerPort_)
      {
         case InputPorts::None:       mode = "None"; break;
         case InputPorts::Port1:      mode = "Port1"; break;
         case InputPorts::Port2:      mode = "Port2"; break;
         case InputPorts::Port1_Once: mode = "Port1_Once"; break;
         case InputPorts::Port2_Once: mode = "Port2_Once"; break;
         default:                     mode = "None"; break;
      }
      pProp->Set(mode.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string mode;
      pProp->Get(mode);

      if (mode == "None")            ttlTriggerPort_ = InputPorts::None;
      else if (mode == "Port1")      ttlTriggerPort_ = InputPorts::Port1;
      else if (mode == "Port2")      ttlTriggerPort_ = InputPorts::Port2;
      else if (mode == "Port1_Once") ttlTriggerPort_ = InputPorts::Port1_Once;
      else if (mode == "Port2_Once") ttlTriggerPort_ = InputPorts::Port2_Once;
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnTTLTriggerBehavior(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(ttlTriggerBehavior_ == TriggerBehaviour::Rising ? "Rising" : "Falling");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string behavior;
      pProp->Get(behavior);
      ttlTriggerBehavior_ = (behavior == "Rising") ?
         TriggerBehaviour::Rising : TriggerBehaviour::Falling;
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnLaserType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(laserFrequency_ == 0 ? "Continuous" : "Pulsed");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string type;
      pProp->Get(type);
      if (type == "Continuous")
         laserFrequency_ = 0;
      // If changed to pulsed, user needs to set frequency via LaserFrequency property

      return UpdateLaser();
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnLaserFrequency(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(laserFrequency_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      laserFrequency_ = static_cast<UINT32>(value);

      return UpdateLaser();
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnTickTime(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(tickTime_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      tickTime_ = static_cast<UINT32>(value);

      if (device_ != nullptr)
      {
         UINT32 ret = device_->SetTickTime(tickTime_);
         return MapRetCode(ret);
      }
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnLaserPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string port;
      switch (laserPort_)
      {
         case OutputPorts::RMIPort1: port = "RMIPort1"; break;
         case OutputPorts::RMIPort2: port = "RMIPort2"; break;
         case OutputPorts::RMIPort3: port = "RMIPort3"; break;
         case OutputPorts::RMIPort4: port = "RMIPort4"; break;
         default: port = "RMIPort1"; break;
      }
      pProp->Set(port.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string port;
      pProp->Get(port);

      if (port == "RMIPort1")      laserPort_ = OutputPorts::RMIPort1;
      else if (port == "RMIPort2") laserPort_ = OutputPorts::RMIPort2;
      else if (port == "RMIPort3") laserPort_ = OutputPorts::RMIPort3;
      else if (port == "RMIPort4") laserPort_ = OutputPorts::RMIPort4;

      return UpdateLaser();
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnDigitalRiseTime(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(digitalRiseTime_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      digitalRiseTime_ = static_cast<UINT32>(value);

      return UpdateLaser();
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnDigitalFallTime(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(digitalFallTime_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      digitalFallTime_ = static_cast<UINT32>(value);

      return UpdateLaser();
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnAnalogChangeTime(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(analogChangeTime_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      analogChangeTime_ = static_cast<UINT32>(value);

      return UpdateLaser();
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnMinIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(minIntensity_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      minIntensity_ = static_cast<UINT32>(value);
   }
   return DEVICE_OK;
}

int RappUGA42Scanner::OnMaxIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(maxIntensity_));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      maxIntensity_ = static_cast<UINT32>(value);
   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////////////////

int RappUGA42Scanner::MapRetCode(UINT32 retCode)
{
   switch (static_cast<RETCODE>(retCode))
   {
      case RETCODE::OK:
         return DEVICE_OK;
      case RETCODE::ConnectionError:
         return ERR_CONNECTION_FAILED;
      case RETCODE::InvalidParam:
         return DEVICE_INVALID_PROPERTY_VALUE;
      case RETCODE::MemOverload:
         return ERR_MEMORY_OVERLOAD;
      case RETCODE::SequenceInvalid:
         return ERR_SEQUENCE_INVALID;
      case RETCODE::ERR:
      default:
         return DEVICE_ERR;
   }
}

int RappUGA42Scanner::AddLaser()
{
   if (laserAdded_)
      return DEVICE_OK;

   LogMessage("UGA42 AddLaser: port=" + std::to_string(static_cast<int>(laserPort_)) +
              ", riseTime=" + std::to_string(digitalRiseTime_) +
              ", fallTime=" + std::to_string(digitalFallTime_) +
              ", freq=" + std::to_string(laserFrequency_), true);

   UINT32 ret = device_->AddLaser(
      digitalRiseTime_,
      digitalFallTime_,
      analogChangeTime_,
      laserPort_,
      minIntensity_,
      maxIntensity_,
      laserFrequency_,
      &laserID_);

   LogMessage("UGA42 AddLaser: returned " + std::to_string(ret) + ", laserID=" + std::to_string(laserID_), true);

   if (ret != RETCODE::OK)
      return ERR_LASER_SETUP_FAILED;

   // Set spot size
   ret = device_->SetLaserStepSize(laserID_, spotSize_);
   LogMessage("UGA42 AddLaser: SetLaserStepSize returned " + std::to_string(ret), true);
   if (ret != RETCODE::OK)
      return ERR_LASER_SETUP_FAILED;

   // Reset calibration to identity (1:1 mapping)
   ret = device_->ResetCalibration(laserID_);
   LogMessage("UGA42 AddLaser: ResetCalibration returned " + std::to_string(ret), true);
   if (ret != RETCODE::OK)
      return ERR_LASER_SETUP_FAILED;

   laserAdded_ = true;
   LogMessage("UGA42 AddLaser: Laser added successfully", true);
   return DEVICE_OK;
}

State RappUGA42Scanner::GetDeviceState()
{
   if (device_ == nullptr)
      return State::Disconnected;

   State state;
   UINT32 ret = device_->GetState(&state);
   if (ret != RETCODE::OK)
      return State::Disconnected;

   return state;
}

bool RappUGA42Scanner::IsDeviceIdle()
{
   State state = GetDeviceState();
   return (state == State::IDLE);
}

bool RappUGA42Scanner::Reconnect()
{
   LogMessage("UGA42: Reconnect() starting", true);

   if (device_ == nullptr)
   {
      LogMessage("UGA42: Reconnect failed - device is null", true);
      return false;
   }

   // Remove existing laser before disconnect (SDK remembers laser even after disconnect)
   if (laserAdded_)
   {
      LogMessage("UGA42: Reconnect - Removing existing laser " + std::to_string(laserID_), true);
      device_->RemoveLaser(laserID_);
      laserAdded_ = false;
   }

   // Disconnect first
   device_->Disconnect();

   // Reconnect
   UINT32 ret = device_->Connect(port_.c_str());
   if (ret != RETCODE::OK)
   {
      LogMessage("UGA42: Reconnect failed at Connect: " + std::to_string(ret), true);
      return false;
   }
   LogMessage("UGA42: Reconnect - Connect successful", true);

   // Verify connection
   State state;
   ret = device_->GetState(&state);
   if (ret != RETCODE::OK || state == State::Disconnected)
   {
      LogMessage("UGA42: Reconnect failed at GetState: ret=" + std::to_string(ret) +
                 ", state=" + std::to_string(static_cast<int>(state)), true);
      return false;
   }
   LogMessage("UGA42: Reconnect - GetState successful, state=" + std::to_string(static_cast<int>(state)), true);

   // Reconfigure tick time
   ret = device_->SetTickTime(tickTime_);
   if (ret != RETCODE::OK)
   {
      LogMessage("UGA42: Reconnect failed at SetTickTime: " + std::to_string(ret), true);
      return false;
   }
   LogMessage("UGA42: Reconnect - SetTickTime successful", true);

   // Re-add laser
   laserAdded_ = false;  // Reset flag so AddLaser will run
   int result = AddLaser();
   if (result != DEVICE_OK)
   {
      LogMessage("UGA42: Reconnect failed at AddLaser: " + std::to_string(result), true);
      return false;
   }

   LogMessage("UGA42: Reconnect successful", true);
   return true;
}

///////////////////////////////////////////////////////////////////////////////
// Command Implementations
///////////////////////////////////////////////////////////////////////////////

int PointAndFireCommand::Execute(RappUGA42Scanner& device)
{
   device.LogMessage("UGA42 PointAndFire: x=" + std::to_string(x_) +
                     ", y=" + std::to_string(y_) +
                     ", pulseTime=" + std::to_string(pulseTime_us_) + "us" +
                     ", intensity=" + std::to_string(device.currentIntensity_) +
                     ", laserID=" + std::to_string(device.laserID_), true);

   // Create sequence point
   SequencePoint* point = new SequencePoint(x_, y_);
   point->StartTick = 0;
   point->LaserID[0] = device.laserID_;
   point->Intensity[0] = device.currentIntensity_;
   point->Repeats = static_cast<UINT32>(pulseTime_us_ / device.tickTime_);
   if (point->Repeats < 1)
      point->Repeats = 1;

   device.LogMessage("UGA42 PointAndFire: Repeats=" + std::to_string(point->Repeats), true);

   // Upload sequence
   SequenceObject* sequence[1] = { point };
   UINT16 sequenceID;
   UINT32 ret = device.device_->UploadSequence(sequence, 1, &sequenceID);

   delete point;

   device.LogMessage("UGA42 PointAndFire: UploadSequence returned " + std::to_string(ret) +
                     ", sequenceID=" + std::to_string(sequenceID), true);

   if (ret != RETCODE::OK)
      return device.MapRetCode(ret);

   // Start sequence
   device.LogMessage("UGA42 PointAndFire: Starting sequence, triggerPort=" +
                     std::to_string(static_cast<int>(device.ttlTriggerPort_)), true);
   ret = device.device_->StartSequence(sequenceID, device.ttlTriggerPort_,
                                       device.ttlTriggerBehavior_, 1);
   device.LogMessage("UGA42 PointAndFire: StartSequence returned " + std::to_string(ret), true);
   if (ret != RETCODE::OK)
   {
      device.device_->DeleteSequence(sequenceID);
      return device.MapRetCode(ret);
   }

   // Poll until complete (blocks the worker thread)
   State state;
   int pollCount = 0;
   do {
      CDeviceUtils::SleepMs(10);
      device.device_->GetState(&state);
      pollCount++;
   } while (state == State::SequenceRunning);

   device.LogMessage("UGA42 PointAndFire: Sequence completed after " + std::to_string(pollCount) +
                     " polls, final state=" + std::to_string(static_cast<int>(state)), true);

   // Clean up
   device.device_->DeleteSequence(sequenceID);

   // Update current position
   device.currentX_ = x_;
   device.currentY_ = y_;

   return DEVICE_OK;
}

int SetPositionCommand::Execute(RappUGA42Scanner& device)
{
   device.LogMessage("UGA42 SetPosition: x=" + std::to_string(x_) + ", y=" + std::to_string(y_), true);

   // Convert to UINT32 for device
   UINT32 deviceX = static_cast<UINT32>(x_);
   UINT32 deviceY = static_cast<UINT32>(y_);

   UINT32 ret = device.device_->MoveAbsolute(deviceX, deviceY);
   device.LogMessage("UGA42 SetPosition: MoveAbsolute returned " + std::to_string(ret), true);
   if (ret != RETCODE::OK)
      return device.MapRetCode(ret);

   device.currentX_ = x_;
   device.currentY_ = y_;

   return DEVICE_OK;
}

int SetIlluminationCommand::Execute(RappUGA42Scanner& device)
{
   device.LogMessage("UGA42 SetIllumination: on=" + std::string(on_ ? "true" : "false") +
                     ", laserID=" + std::to_string(device.laserID_), true);
   UINT32 ret = device.device_->SetLaserONOFF(device.laserID_, on_ ? TRUE : FALSE);
   device.LogMessage("UGA42 SetIllumination: SetLaserONOFF returned " + std::to_string(ret), true);
   return device.MapRetCode(ret);
}

int LoadPolygonsCommand::Execute(RappUGA42Scanner& device)
{
   device.LogMessage("UGA42 LoadPolygons: " + std::to_string(polygons_.size()) + " polygons", true);

   // Delete any previously loaded polygon sequence
   if (device.loadedPolygonSequenceID_ != 0)
   {
      device.LogMessage("UGA42 LoadPolygons: Deleting previous sequence " +
                        std::to_string(device.loadedPolygonSequenceID_), true);
      device.device_->DeleteSequence(device.loadedPolygonSequenceID_);
      device.loadedPolygonSequenceID_ = 0;
   }

   if (polygons_.empty())
   {
      device.LogMessage("UGA42 LoadPolygons: No polygons to load", true);
      device.polygonsLoaded_ = false;
      return DEVICE_OK;
   }

   // Create sequence objects for each polygon
   std::vector<SequenceObject*> sequenceObjects;
   UINT32 currentTick = 0;

   for (size_t i = 0; i < polygons_.size(); i++)
   {
      if (polygons_[i].empty())
         continue;

      SequencePolygon* poly = new SequencePolygon(
         polygons_[i].data(),
         static_cast<UINT32>(polygons_[i].size()),
         TRUE);  // Filled

      poly->StartTick = currentTick;
      poly->LaserID[0] = device.laserID_;
      poly->Intensity[0] = device.currentIntensity_;
      poly->Repeats = 1;

      sequenceObjects.push_back(poly);

      // Calculate timing for next polygon
      UINT32 objTickCount = 0;
      UINT32 illTickCount = 0;
      device.device_->GetTiming(*poly, &objTickCount, &illTickCount);
      currentTick += objTickCount;
   }

   if (sequenceObjects.empty())
   {
      device.polygonsLoaded_ = false;
      return DEVICE_OK;
   }

   // Upload sequence to device and CACHE the ID
   UINT16 sequenceID;
   UINT32 ret = device.device_->UploadSequence(
      sequenceObjects.data(),
      static_cast<UINT32>(sequenceObjects.size()),
      &sequenceID);

   // Clean up sequence objects
   for (auto obj : sequenceObjects)
   {
      delete obj;
   }

   if (ret != RETCODE::OK)
   {
      device.polygonsLoaded_ = false;
      return device.MapRetCode(ret);
   }

   // Store the sequence ID for later use by RunPolygons
   device.loadedPolygonSequenceID_ = sequenceID;
   device.polygonsLoaded_ = true;
   device.LogMessage("UGA42 LoadPolygons: Uploaded sequence ID " + std::to_string(sequenceID), true);

   return DEVICE_OK;
}

int RunPolygonsCommand::Execute(RappUGA42Scanner& device)
{
   device.LogMessage("UGA42 RunPolygons: repetitions=" + std::to_string(repetitions_) +
                     ", sequenceID=" + std::to_string(device.loadedPolygonSequenceID_), true);

   // Check if polygons were pre-loaded
   if (!device.polygonsLoaded_ || device.loadedPolygonSequenceID_ == 0)
   {
      device.LogMessage("UGA42 RunPolygons: No polygons loaded", true);
      // Polygons not loaded, return error
      return ERR_INVALID_DEVICE_STATE;
   }

   // Start the pre-loaded sequence (FAST - no upload needed)
   UINT32 ret = device.device_->StartSequence(
      device.loadedPolygonSequenceID_,
      device.ttlTriggerPort_,
      device.ttlTriggerBehavior_,
      static_cast<UINT32>(repetitions_));

   device.LogMessage("UGA42 RunPolygons: StartSequence returned " + std::to_string(ret), true);

   if (ret != RETCODE::OK)
      return device.MapRetCode(ret);

   device.lastSequenceID_ = device.loadedPolygonSequenceID_;
   device.sequenceRunning_ = true;

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// DeviceWorkerThread Implementation
///////////////////////////////////////////////////////////////////////////////

DeviceWorkerThread::DeviceWorkerThread(RappUGA42Scanner& device) :
   device_(device),
   stop_(false),
   activeCommand_(nullptr),
   stopPolygonRequested_(false),
   lastKeepaliveTime_(std::chrono::steady_clock::now())
{
}

DeviceWorkerThread::~DeviceWorkerThread()
{
   Stop();
   wait();
}

void DeviceWorkerThread::Start()
{
   stop_ = false;
   activate();
}

void DeviceWorkerThread::Stop()
{
   {
      std::lock_guard<std::mutex> lock(queueMutex_);
      stop_ = true;
   }
   queueCV_.notify_one();
}

void DeviceWorkerThread::EnqueueCommand(Command* cmd)
{
   {
      std::lock_guard<std::mutex> lock(queueMutex_);
      commandQueue_.push(cmd);
   }
   queueCV_.notify_one();
}

bool DeviceWorkerThread::IsBusy()
{
   std::lock_guard<std::mutex> lock(queueMutex_);
   return !commandQueue_.empty() || activeCommand_ != nullptr;
}

void DeviceWorkerThread::StopPolygonSequence()
{
   stopPolygonRequested_ = true;

   // Clear any queued RunPolygons commands
   std::lock_guard<std::mutex> lock(queueMutex_);
   std::queue<Command*> filteredQueue;
   while (!commandQueue_.empty())
   {
      Command* cmd = commandQueue_.front();
      commandQueue_.pop();
      if (cmd->GetType() != Command::RUN_POLYGONS)
      {
         filteredQueue.push(cmd);
      }
      else
      {
         delete cmd;  // Discard RunPolygons commands
      }
   }
   commandQueue_ = filteredQueue;
}

int DeviceWorkerThread::svc()
{
   device_.LogMessage("UGA42 Worker: Thread started", true);

   // Immediate first GetState call on thread start to prevent initial timeout
   if (!device_.debugMode_ && device_.device_ != nullptr)
   {
      State state;
      UINT32 ret = device_.device_->GetState(&state);
      device_.LogMessage("UGA42 Worker: Initial GetState returned " +
                         std::to_string(ret) + ", state=" + std::to_string(static_cast<int>(state)), true);
      lastKeepaliveTime_ = std::chrono::steady_clock::now();
   }

   while (!stop_)
   {
      Command* cmd = nullptr;

      // Wait for command or stop signal (with timeout for keepalive check)
      {
         std::unique_lock<std::mutex> lock(queueMutex_);

         // Wait with 500ms timeout to periodically check keepalive
         queueCV_.wait_for(lock, std::chrono::milliseconds(500), [this] {
            return !commandQueue_.empty() || stop_;
         });

         if (stop_)
         {
            device_.LogMessage("UGA42 Worker: Stop requested, exiting", true);
            break;
         }

         if (!commandQueue_.empty())
         {
            cmd = commandQueue_.front();
            commandQueue_.pop();
            activeCommand_ = cmd;
            device_.LogMessage("UGA42 Worker: Dequeued command type " + std::to_string(cmd->GetType()), true);
         }
      }

      // Send keepalive if needed (not in debug mode and 1+ seconds elapsed)
      // In fast mode, GetState must be called every 4 seconds to maintain connection.
      // We use 1 second for safety margin. This runs regardless of command queue state
      // to ensure keepalive happens even during long idle periods.
      auto now = std::chrono::steady_clock::now();
      if (!device_.debugMode_)
      {
         auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - lastKeepaliveTime_).count();

         if (elapsed >= 1)
         {
            // Send keepalive GetState call
            State state;
            UINT32 ret = device_.device_->GetState(&state);
            device_.LogMessage("UGA42 Worker: Keepalive GetState returned " + std::to_string(ret) +
                              ", state=" + std::to_string(static_cast<int>(state)) +
                              ", elapsed=" + std::to_string(elapsed) + "s", true);

            // If connection error, attempt reconnect
            if (ret != RETCODE::OK || state == State::Disconnected)
            {
               device_.LogMessage("UGA42 Worker: Connection lost, attempting reconnect", true);
               if (device_.Reconnect())
               {
                  device_.LogMessage("UGA42 Worker: Reconnect succeeded", true);
               }
               else
               {
                  device_.LogMessage("UGA42 Worker: Reconnect failed", true);
               }
            }

            lastKeepaliveTime_ = now;
         }
      }

      // Execute command outside of lock
      if (cmd)
      {
         // Check if we need to stop a RunPolygons sequence
         if (stopPolygonRequested_ && cmd->GetType() == Command::RUN_POLYGONS)
         {
            device_.LogMessage("UGA42 Worker: Stopping polygon sequence as requested", true);
            device_.device_->Stop();
            device_.sequenceRunning_ = false;
            stopPolygonRequested_ = false;
         }

         device_.LogMessage("UGA42 Worker: Executing command type " + std::to_string(cmd->GetType()), true);
         int result = cmd->Execute(device_);
         device_.LogMessage("UGA42 Worker: Command completed with result " + std::to_string(result), true);
         delete cmd;
         activeCommand_ = nullptr;

         // Update keepalive time after command (device communication happened)
         lastKeepaliveTime_ = std::chrono::steady_clock::now();
      }
   }

   device_.LogMessage("UGA42 Worker: Thread exiting", true);
   return 0;
}
