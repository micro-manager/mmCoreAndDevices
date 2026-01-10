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
   debugMode_(true),  // Start with debug mode for development
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
   sequenceRunning_(false)
{
   // Pre-initialization properties
   // CPropertyAction* pAct = new CPropertyAction(this, &RappUGA42Scanner::OnPort);
   // CreateProperty(g_PropertyVirtualComPort, "", MM::String, false, pAct, true);

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
   UINT32 ret = device_->Connect(port_.c_str());
   if (ret != RETCODE::OK)
   {
      delete device_;
      device_ = nullptr;
      return MapRetCode(ret);
   }

   // Verify connection
   State state;
   ret = device_->GetState(&state);
   if (ret != RETCODE::OK || state == State::Disconnected)
   {
      delete device_;
      device_ = nullptr;
      return ERR_CONNECTION_FAILED;
   }

   // Set tick time
   ret = device_->SetTickTime(tickTime_);
   if (ret != RETCODE::OK)
   {
      device_->Disconnect();
      delete device_;
      device_ = nullptr;
      return MapRetCode(ret);
   }

   // Add and configure laser
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

   initialized_ = true;
   return DEVICE_OK;
}

int RappUGA42Scanner::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   if (device_ != nullptr)
   {
      // Stop any running sequences
      device_->Stop();

      // Delete all sequences from device memory
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
   if (!initialized_ || device_ == nullptr)
      return false;

   State state = GetDeviceState();
   return (state == State::SequenceRunning ||
           state == State::Calibrating ||
           state == State::Initializing);
}

///////////////////////////////////////////////////////////////////////////////
// Galvo API Implementation
///////////////////////////////////////////////////////////////////////////////

int RappUGA42Scanner::PointAndFire(double x, double y, double pulseTime_us)
{
   if (!initialized_ || device_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   if (!IsDeviceIdle())
      return ERR_INVALID_DEVICE_STATE;

   // Create a sequence point with the given coordinates (already in device coordinates)
   SequencePoint* point = new SequencePoint(x, y);
   point->StartTick = 0;
   point->LaserID[0] = laserID_;
   point->Intensity[0] = currentIntensity_;
   point->Repeats = static_cast<UINT32>(pulseTime_us / tickTime_);
   if (point->Repeats < 1)
      point->Repeats = 1;

   // Upload sequence
   SequenceObject* sequence[1] = { point };
   UINT16 sequenceID;
   UINT32 ret = device_->UploadSequence(sequence, 1, &sequenceID);

   delete point;

   if (ret != RETCODE::OK)
      return MapRetCode(ret);

   // Start sequence
   ret = device_->StartSequence(sequenceID, ttlTriggerPort_, ttlTriggerBehavior_, 1);
   if (ret != RETCODE::OK)
   {
      device_->DeleteSequence(sequenceID);
      return MapRetCode(ret);
   }

   // Wait for sequence to complete
   State state;
   do {
      CDeviceUtils::SleepMs(10);
      device_->GetState(&state);
   } while (state == State::SequenceRunning);

   // Clean up
   device_->DeleteSequence(sequenceID);

   // Update current position
   currentX_ = x;
   currentY_ = y;

   return DEVICE_OK;
}

int RappUGA42Scanner::SetSpotInterval(double pulseTime_us)
{
   pulseTime_us_ = pulseTime_us;
   return DEVICE_OK;
}

int RappUGA42Scanner::SetPosition(double x, double y)
{
   if (!initialized_ || device_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   if (!IsDeviceIdle())
      return ERR_INVALID_DEVICE_STATE;

   // Convert to UINT32 for device (coordinates already in device range 0-65535)
   UINT32 deviceX = static_cast<UINT32>(x);
   UINT32 deviceY = static_cast<UINT32>(y);

   UINT32 ret = device_->MoveAbsolute(deviceX, deviceY);
   if (ret != RETCODE::OK)
      return MapRetCode(ret);

   currentX_ = x;
   currentY_ = y;

   return DEVICE_OK;
}

int RappUGA42Scanner::GetPosition(double& x, double& y)
{
   if (!initialized_ || device_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   POINT pos;
   UINT32 ret = device_->GetActualPosition(&pos);
   if (ret != RETCODE::OK)
      return MapRetCode(ret);

   x = static_cast<double>(pos.x);
   y = static_cast<double>(pos.y);

   currentX_ = x;
   currentY_ = y;

   return DEVICE_OK;
}

int RappUGA42Scanner::SetIlluminationState(bool on)
{
   if (!initialized_ || device_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   if (!IsDeviceIdle())
      return ERR_INVALID_DEVICE_STATE;

   UINT32 ret = device_->SetLaserONOFF(laserID_, on ? TRUE : FALSE);
   return MapRetCode(ret);
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
   return DEVICE_OK;
}

int RappUGA42Scanner::LoadPolygons()
{
   // Polygons are loaded on-demand when RunPolygons() is called
   return DEVICE_OK;
}

int RappUGA42Scanner::SetPolygonRepetitions(int repetitions)
{
   if (repetitions < 1)
      return DEVICE_INVALID_PROPERTY_VALUE;

   polygonRepetitions_ = repetitions;
   return DEVICE_OK;
}

int RappUGA42Scanner::RunPolygons()
{
   if (!initialized_ || device_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   if (!IsDeviceIdle())
      return ERR_INVALID_DEVICE_STATE;

   if (polygons_.empty())
      return DEVICE_OK;  // Nothing to do

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
      poly->LaserID[0] = laserID_;
      poly->Intensity[0] = currentIntensity_;
      poly->Repeats = 1;

      sequenceObjects.push_back(poly);

      // Calculate timing for next polygon
      UINT32 objTickCount = 0;
      UINT32 illTickCount = 0;
      device_->GetTiming(*poly, &objTickCount, &illTickCount);
      currentTick += objTickCount;
   }

   if (sequenceObjects.empty())
      return DEVICE_OK;

   // Upload sequence
   UINT16 sequenceID;
   UINT32 ret = device_->UploadSequence(
      sequenceObjects.data(),
      static_cast<UINT32>(sequenceObjects.size()),
      &sequenceID);

   // Clean up sequence objects
   for (auto obj : sequenceObjects)
   {
      delete obj;
   }

   if (ret != RETCODE::OK)
      return MapRetCode(ret);

   // Start sequence
   ret = device_->StartSequence(
      sequenceID,
      ttlTriggerPort_,
      ttlTriggerBehavior_,
      static_cast<UINT32>(polygonRepetitions_));

   if (ret != RETCODE::OK)
   {
      device_->DeleteSequence(sequenceID);
      return MapRetCode(ret);
   }

   lastSequenceID_ = sequenceID;
   sequenceRunning_ = true;

   return DEVICE_OK;
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
   if (!initialized_ || device_ == nullptr)
      return DEVICE_NOT_CONNECTED;

   UINT32 ret = device_->Stop();
   sequenceRunning_ = false;

   return MapRetCode(ret);
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

   UINT32 ret = device_->AddLaser(
      digitalRiseTime_,
      digitalFallTime_,
      analogChangeTime_,
      laserPort_,
      minIntensity_,
      maxIntensity_,
      laserFrequency_,
      &laserID_);

   if (ret != RETCODE::OK)
      return ERR_LASER_SETUP_FAILED;

   // Set spot size
   ret = device_->SetLaserStepSize(laserID_, spotSize_);
   if (ret != RETCODE::OK)
      return ERR_LASER_SETUP_FAILED;

   // Reset calibration to identity (1:1 mapping)
   ret = device_->ResetCalibration(laserID_);
   if (ret != RETCODE::OK)
      return ERR_LASER_SETUP_FAILED;

   laserAdded_ = true;
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
