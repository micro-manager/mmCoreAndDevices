///////////////////////////////////////////////////////////////////////////////
// FILE:          EnderscopeStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//
// DESCRIPTION:   Enderscope Stage adapter (Marlin/Enderscope-compatible)
//                Adapted in spirit from the Marzhauser-LStep adapter shape.
///////////////////////////////////////////////////////////////////////////////

#ifdef WIN32
// Prevent windows.h (pulled in transitively) from defining min/max macros,
// which break std::min/std::max.
#define NOMINMAX
#pragma warning(disable : 4355)
#endif

#include "EnderscopeStage.h"

#include "MMDevice.h"
#include "ModuleInterface.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

using namespace std;

const char* g_EnderscopeXYStageDeviceName = "EnderscopeXYStage";
const char* g_EnderscopeZStageDeviceName = "EnderscopeZStage";

namespace
{
const long kDefaultBaudRate = 115200;
const long kDefaultReadTimeoutMs = 1000;
const double kDefaultStepSizeUm = 1.0;

const char* kGCodeAbsolute = "G90";
const char* kGCodeRelative = "G91";
const char* kGCodeHomeAll = "G28";
const char* kGCodeHomeXY = "G28 X Y";
const char* kGCodeHomeZ = "G28 Z";
const char* kGCodeFinish = "M400";
const char* kGCodePosition = "M114";
const char* kGCodeStop = "M410";

inline long RoundToLong(double value)
{
   return static_cast<long>(value >= 0.0 ? value + 0.5 : value - 0.5);
}
} // namespace

MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_EnderscopeXYStageDeviceName, MM::XYStageDevice, "Enderscope XY Stage (Marlin G-code)");
   RegisterDevice(g_EnderscopeZStageDeviceName, MM::StageDevice, "Enderscope Z Stage (Marlin G-code)");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
   {
      return 0;
   }

   if (strcmp(deviceName, g_EnderscopeXYStageDeviceName) == 0)
   {
      return new EnderscopeXYStage();
   }

   if (strcmp(deviceName, g_EnderscopeZStageDeviceName) == 0)
   {
      return new EnderscopeZStage();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

EnderscopeBase::EnderscopeBase(MM::Device* device)
   : initialized_(false),
     port_("Undefined"),
     baudRate_(kDefaultBaudRate),
     readTimeoutMs_(kDefaultReadTimeoutMs),
     device_(device),
     core_(0)
{
}

EnderscopeBase::~EnderscopeBase() {}

int EnderscopeBase::CheckDeviceStatus()
{
   if (core_ == 0)
   {
      return DEVICE_NOT_CONNECTED;
   }

   int ret = ClearPort();
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   double x = 0.0;
   double y = 0.0;
   double z = 0.0;
   ret = QueryPositionMm(x, y, z);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   initialized_ = true;
   return DEVICE_OK;
}

int EnderscopeBase::ClearPort()
{
   if (core_ == 0)
   {
      return DEVICE_NOT_CONNECTED;
   }

   const int bufSize = 255;
   unsigned char clear[bufSize];
   unsigned long read = bufSize;

   int ret = DEVICE_OK;
   while (static_cast<int>(read) == bufSize)
   {
      ret = core_->ReadFromSerial(device_, port_.c_str(), clear, bufSize, read);
      if (ret != DEVICE_OK)
      {
         return ret;
      }
   }

   return DEVICE_OK;
}

int EnderscopeBase::SendCommand(const std::string& command) const
{
   if (core_ == 0)
   {
      return DEVICE_NOT_CONNECTED;
   }

   const char* txTerm = "\n";
   return core_->SetSerialCommand(device_, port_.c_str(), command.c_str(), txTerm);
}

int EnderscopeBase::ReadLine(std::string& line) const
{
   if (core_ == 0)
   {
      return DEVICE_NOT_CONNECTED;
   }

   const size_t bufSize = 2048;
   char buffer[bufSize];
   memset(buffer, 0, bufSize);

   const char* rxTerm = "\n";
   int ret = core_->GetSerialAnswer(device_, port_.c_str(), bufSize, buffer, rxTerm);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   line = Trim(buffer);
   return DEVICE_OK;
}

int EnderscopeBase::CommandExpectOk(const std::string& command) const
{
   int ret = SendCommand(command);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   const long maxReads = std::max(1L, readTimeoutMs_ / 10L);
   for (long i = 0; i < maxReads; ++i)
   {
      std::string line;
      ret = ReadLine(line);
      if (ret != DEVICE_OK)
      {
         return ret;
      }

      if (line.empty())
      {
         continue;
      }

      if (line.rfind("ok", 0) == 0)
      {
         return DEVICE_OK;
      }
   }

   return DEVICE_SERIAL_TIMEOUT;
}

int EnderscopeBase::QueryPositionMm(double& x, double& y, double& z) const
{
   int ret = SendCommand(kGCodePosition);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   bool sawDataLine = false;
   bool sawOk = false;
   std::string dataLine;

   const long maxReads = std::max(2L, readTimeoutMs_ / 10L + 2L);
   for (long i = 0; i < maxReads; ++i)
   {
      std::string line;
      ret = ReadLine(line);
      if (ret != DEVICE_OK)
      {
         return ret;
      }

      if (line.empty())
      {
         continue;
      }

      if (line.rfind("ok", 0) == 0)
      {
         sawOk = true;
         if (sawDataLine)
         {
            break;
         }
         continue;
      }

      if (!sawDataLine)
      {
         dataLine = line;
         sawDataLine = true;
      }
   }

   if (!sawDataLine || !sawOk)
   {
      return DEVICE_SERIAL_INVALID_RESPONSE;
   }

   if (!ParseAxisValue(dataLine, 'X', x) || !ParseAxisValue(dataLine, 'Y', y) || !ParseAxisValue(dataLine, 'Z', z))
   {
      return DEVICE_SERIAL_INVALID_RESPONSE;
   }

   return DEVICE_OK;
}

std::string EnderscopeBase::Trim(const std::string& input)
{
   const std::string whitespace = " \t\r\n";
   const size_t first = input.find_first_not_of(whitespace);
   if (first == std::string::npos)
   {
      return std::string();
   }

   const size_t last = input.find_last_not_of(whitespace);
   return input.substr(first, last - first + 1);
}

bool EnderscopeBase::ParseAxisValue(const std::string& line, char axis, double& value)
{
   const std::string key = std::string(1, axis) + ":";
   const size_t pos = line.find(key);
   if (pos == std::string::npos)
   {
      return false;
   }

   const size_t numberStart = pos + key.size();
   size_t numberEnd = numberStart;
   while (numberEnd < line.size())
   {
      const char c = line[numberEnd];
      const bool numeric = (c == '+') || (c == '-') || (c == '.') || (c == 'e') || (c == 'E') || std::isdigit(static_cast<unsigned char>(c));
      if (!numeric)
      {
         break;
      }
      ++numberEnd;
   }

   if (numberEnd == numberStart)
   {
      return false;
   }

   const std::string num = line.substr(numberStart, numberEnd - numberStart);
   char* endPtr = 0;
   value = std::strtod(num.c_str(), &endPtr);
   return endPtr != num.c_str();
}

EnderscopeXYStage::EnderscopeXYStage()
   : EnderscopeBase(this),
     stepSizeXUm_(kDefaultStepSizeUm),
     stepSizeYUm_(kDefaultStepSizeUm),
     originXUm_(0.0),
     originYUm_(0.0),
     lastXUm_(0.0),
     lastYUm_(0.0)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Port property cannot be changed after initialization.");

   CreateProperty(MM::g_Keyword_Name, g_EnderscopeXYStageDeviceName, MM::String, true);
   CreateProperty(MM::g_Keyword_Description, "Enderscope XY stage adapter", MM::String, true);

   CPropertyAction* pAct = new CPropertyAction(this, &EnderscopeXYStage::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

   pAct = new CPropertyAction(this, &EnderscopeXYStage::OnBaudRate);
   CreateProperty("BaudRate", CDeviceUtils::ConvertToString(baudRate_), MM::Integer, false, pAct, true);

   AddAllowedValue("BaudRate", "9600");
   AddAllowedValue("BaudRate", "57600");
   AddAllowedValue("BaudRate", "115200");
   AddAllowedValue("BaudRate", "250000");

   pAct = new CPropertyAction(this, &EnderscopeXYStage::OnReadTimeout);
   CreateProperty("ReadTimeoutMs", CDeviceUtils::ConvertToString(readTimeoutMs_), MM::Integer, false, pAct, true);
   SetPropertyLimits("ReadTimeoutMs", 100, 10000);
}

EnderscopeXYStage::~EnderscopeXYStage()
{
   Shutdown();
}

void EnderscopeXYStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_EnderscopeXYStageDeviceName);
}

int EnderscopeXYStage::Initialize()
{
   core_ = GetCoreCallback();

   int ret = CheckDeviceStatus();
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   CPropertyAction* pAct = new CPropertyAction(this, &EnderscopeXYStage::OnStepSizeX);
   ret = CreateProperty("StepSizeX [um]", CDeviceUtils::ConvertToString(stepSizeXUm_), MM::Float, false, pAct);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   pAct = new CPropertyAction(this, &EnderscopeXYStage::OnStepSizeY);
   ret = CreateProperty("StepSizeY [um]", CDeviceUtils::ConvertToString(stepSizeYUm_), MM::Float, false, pAct);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   initialized_ = true;
   return DEVICE_OK;
}

int EnderscopeXYStage::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

bool EnderscopeXYStage::Busy()
{
   return false;
}

int EnderscopeXYStage::SetAbsoluteMm(double xMm, double yMm)
{
   int ret = CommandExpectOk(kGCodeAbsolute);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   ostringstream cmd;
   cmd << "G0 X " << xMm << " Y " << yMm;
   ret = CommandExpectOk(cmd.str());
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   return CommandExpectOk(kGCodeFinish);
}

int EnderscopeXYStage::SetRelativeMm(double dxMm, double dyMm)
{
   int ret = CommandExpectOk(kGCodeRelative);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   ostringstream cmd;
   cmd << "G0 X " << dxMm << " Y " << dyMm;
   ret = CommandExpectOk(cmd.str());
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   return CommandExpectOk(kGCodeFinish);
}

int EnderscopeXYStage::SetPositionUm(double x, double y)
{
   const double xMm = (x + originXUm_) / 1000.0;
   const double yMm = (y + originYUm_) / 1000.0;

   const int ret = SetAbsoluteMm(xMm, yMm);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   lastXUm_ = x;
   lastYUm_ = y;
   return DEVICE_OK;
}

int EnderscopeXYStage::GetPositionUm(double& x, double& y)
{
   double xMm = 0.0;
   double yMm = 0.0;
   double zMm = 0.0;

   int ret = QueryPositionMm(xMm, yMm, zMm);
   if (ret != DEVICE_OK)
   {
      x = lastXUm_;
      y = lastYUm_;
      return ret;
   }

   x = xMm * 1000.0 - originXUm_;
   y = yMm * 1000.0 - originYUm_;

   lastXUm_ = x;
   lastYUm_ = y;
   return DEVICE_OK;
}

int EnderscopeXYStage::SetRelativePositionUm(double dx, double dy)
{
   const int ret = SetRelativeMm(dx / 1000.0, dy / 1000.0);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   lastXUm_ += dx;
   lastYUm_ += dy;
   return DEVICE_OK;
}

int EnderscopeXYStage::SetPositionSteps(long x, long y)
{
   const double xUm = static_cast<double>(x) * stepSizeXUm_;
   const double yUm = static_cast<double>(y) * stepSizeYUm_;
   return SetPositionUm(xUm, yUm);
}

int EnderscopeXYStage::GetPositionSteps(long& x, long& y)
{
   double xUm = 0.0;
   double yUm = 0.0;
   int ret = GetPositionUm(xUm, yUm);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   x = RoundToLong(xUm / stepSizeXUm_);
   y = RoundToLong(yUm / stepSizeYUm_);
   return DEVICE_OK;
}

int EnderscopeXYStage::SetRelativePositionSteps(long x, long y)
{
   const double dxUm = static_cast<double>(x) * stepSizeXUm_;
   const double dyUm = static_cast<double>(y) * stepSizeYUm_;
   return SetRelativePositionUm(dxUm, dyUm);
}

int EnderscopeXYStage::Home()
{
   int ret = CommandExpectOk(kGCodeHomeXY);
   if (ret != DEVICE_OK)
   {
      ret = CommandExpectOk(kGCodeHomeAll);
      if (ret != DEVICE_OK)
      {
         return ret;
      }
   }

   ret = CommandExpectOk(kGCodeFinish);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   lastXUm_ = 0.0;
   lastYUm_ = 0.0;
   originXUm_ = 0.0;
   originYUm_ = 0.0;
   return DEVICE_OK;
}

int EnderscopeXYStage::Stop()
{
   return CommandExpectOk(kGCodeStop);
}

int EnderscopeXYStage::SetOrigin()
{
   double x = 0.0;
   double y = 0.0;
   int ret = GetPositionUm(x, y);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   originXUm_ += x;
   originYUm_ += y;
   lastXUm_ = 0.0;
   lastYUm_ = 0.0;
   return DEVICE_OK;
}

int EnderscopeXYStage::SetAdapterOriginUm(double x, double y)
{
   double hwXmm = 0.0;
   double hwYmm = 0.0;
   double hwZmm = 0.0;
   int ret = QueryPositionMm(hwXmm, hwYmm, hwZmm);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   originXUm_ = hwXmm * 1000.0 - x;
   originYUm_ = hwYmm * 1000.0 - y;
   lastXUm_ = x;
   lastYUm_ = y;
   return DEVICE_OK;
}

int EnderscopeXYStage::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax)
{
   xMin = -std::numeric_limits<double>::max();
   xMax = std::numeric_limits<double>::max();
   yMin = -std::numeric_limits<double>::max();
   yMax = std::numeric_limits<double>::max();
   return DEVICE_OK;
}

int EnderscopeXYStage::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax)
{
   xMin = std::numeric_limits<long>::min();
   xMax = std::numeric_limits<long>::max();
   yMin = std::numeric_limits<long>::min();
   yMax = std::numeric_limits<long>::max();
   return DEVICE_OK;
}

int EnderscopeXYStage::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int EnderscopeXYStage::OnBaudRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(baudRate_);
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         pProp->Set(baudRate_);
         return DEVICE_CAN_NOT_SET_PROPERTY;
      }
      pProp->Get(baudRate_);
   }

   return DEVICE_OK;
}

int EnderscopeXYStage::OnReadTimeout(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(readTimeoutMs_);
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         pProp->Set(readTimeoutMs_);
         return DEVICE_CAN_NOT_SET_PROPERTY;
      }
      pProp->Get(readTimeoutMs_);
   }

   return DEVICE_OK;
}

int EnderscopeXYStage::OnStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stepSizeXUm_);
   }
   else if (eAct == MM::AfterSet)
   {
      double value = 0.0;
      pProp->Get(value);
      if (value <= 0.0)
      {
         pProp->Set(stepSizeXUm_);
         return DEVICE_INVALID_INPUT_PARAM;
      }
      stepSizeXUm_ = value;
   }

   return DEVICE_OK;
}

int EnderscopeXYStage::OnStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stepSizeYUm_);
   }
   else if (eAct == MM::AfterSet)
   {
      double value = 0.0;
      pProp->Get(value);
      if (value <= 0.0)
      {
         pProp->Set(stepSizeYUm_);
         return DEVICE_INVALID_INPUT_PARAM;
      }
      stepSizeYUm_ = value;
   }

   return DEVICE_OK;
}

EnderscopeZStage::EnderscopeZStage()
   : EnderscopeBase(this),
     stepSizeUm_(kDefaultStepSizeUm),
     originZUm_(0.0),
     lastZUm_(0.0)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Port property cannot be changed after initialization.");

   CreateProperty(MM::g_Keyword_Name, g_EnderscopeZStageDeviceName, MM::String, true);
   CreateProperty(MM::g_Keyword_Description, "Enderscope Z stage adapter", MM::String, true);

   CPropertyAction* pAct = new CPropertyAction(this, &EnderscopeZStage::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

   pAct = new CPropertyAction(this, &EnderscopeZStage::OnBaudRate);
   CreateProperty("BaudRate", CDeviceUtils::ConvertToString(baudRate_), MM::Integer, false, pAct, true);

   AddAllowedValue("BaudRate", "9600");
   AddAllowedValue("BaudRate", "57600");
   AddAllowedValue("BaudRate", "115200");
   AddAllowedValue("BaudRate", "250000");

   pAct = new CPropertyAction(this, &EnderscopeZStage::OnReadTimeout);
   CreateProperty("ReadTimeoutMs", CDeviceUtils::ConvertToString(readTimeoutMs_), MM::Integer, false, pAct, true);
   SetPropertyLimits("ReadTimeoutMs", 100, 10000);
}

EnderscopeZStage::~EnderscopeZStage()
{
   Shutdown();
}

void EnderscopeZStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_EnderscopeZStageDeviceName);
}

int EnderscopeZStage::Initialize()
{
   core_ = GetCoreCallback();

   int ret = CheckDeviceStatus();
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   CPropertyAction* pAct = new CPropertyAction(this, &EnderscopeZStage::OnStepSize);
   ret = CreateProperty("StepSize [um]", CDeviceUtils::ConvertToString(stepSizeUm_), MM::Float, false, pAct);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   initialized_ = true;
   return DEVICE_OK;
}

int EnderscopeZStage::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

bool EnderscopeZStage::Busy()
{
   return false;
}

int EnderscopeZStage::SetAbsoluteMm(double zMm)
{
   int ret = CommandExpectOk(kGCodeAbsolute);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   ostringstream cmd;
   cmd << "G0 Z " << zMm;
   ret = CommandExpectOk(cmd.str());
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   return CommandExpectOk(kGCodeFinish);
}

int EnderscopeZStage::SetRelativeMm(double dzMm)
{
   int ret = CommandExpectOk(kGCodeRelative);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   ostringstream cmd;
   cmd << "G0 Z " << dzMm;
   ret = CommandExpectOk(cmd.str());
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   return CommandExpectOk(kGCodeFinish);
}

int EnderscopeZStage::SetPositionUm(double pos)
{
   const double zMm = (pos + originZUm_) / 1000.0;
   const int ret = SetAbsoluteMm(zMm);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   lastZUm_ = pos;
   return DEVICE_OK;
}

int EnderscopeZStage::SetRelativePositionUm(double d)
{
   const int ret = SetRelativeMm(d / 1000.0);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   lastZUm_ += d;
   return DEVICE_OK;
}

int EnderscopeZStage::GetPositionUm(double& pos)
{
   double xMm = 0.0;
   double yMm = 0.0;
   double zMm = 0.0;

   int ret = QueryPositionMm(xMm, yMm, zMm);
   if (ret != DEVICE_OK)
   {
      pos = lastZUm_;
      return ret;
   }

   pos = zMm * 1000.0 - originZUm_;
   lastZUm_ = pos;
   return DEVICE_OK;
}

int EnderscopeZStage::SetPositionSteps(long steps)
{
   const double posUm = static_cast<double>(steps) * stepSizeUm_;
   return SetPositionUm(posUm);
}

int EnderscopeZStage::GetPositionSteps(long& steps)
{
   double posUm = 0.0;
   int ret = GetPositionUm(posUm);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   steps = RoundToLong(posUm / stepSizeUm_);
   return DEVICE_OK;
}

int EnderscopeZStage::Stop()
{
   return CommandExpectOk(kGCodeStop);
}

int EnderscopeZStage::Home()
{
   int ret = CommandExpectOk(kGCodeHomeZ);
   if (ret != DEVICE_OK)
   {
      ret = CommandExpectOk(kGCodeHomeAll);
      if (ret != DEVICE_OK)
      {
         return ret;
      }
   }

   ret = CommandExpectOk(kGCodeFinish);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   lastZUm_ = 0.0;
   originZUm_ = 0.0;
   return DEVICE_OK;
}

int EnderscopeZStage::SetOrigin()
{
   double z = 0.0;
   int ret = GetPositionUm(z);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   originZUm_ += z;
   lastZUm_ = 0.0;
   return DEVICE_OK;
}

int EnderscopeZStage::SetAdapterOriginUm(double d)
{
   double xMm = 0.0;
   double yMm = 0.0;
   double zMm = 0.0;
   int ret = QueryPositionMm(xMm, yMm, zMm);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   originZUm_ = zMm * 1000.0 - d;
   lastZUm_ = d;
   return DEVICE_OK;
}

int EnderscopeZStage::GetLimits(double& min, double& max)
{
   min = -std::numeric_limits<double>::max();
   max = std::numeric_limits<double>::max();
   return DEVICE_OK;
}

int EnderscopeZStage::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int EnderscopeZStage::OnBaudRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(baudRate_);
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         pProp->Set(baudRate_);
         return DEVICE_CAN_NOT_SET_PROPERTY;
      }
      pProp->Get(baudRate_);
   }

   return DEVICE_OK;
}

int EnderscopeZStage::OnReadTimeout(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(readTimeoutMs_);
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         pProp->Set(readTimeoutMs_);
         return DEVICE_CAN_NOT_SET_PROPERTY;
      }
      pProp->Get(readTimeoutMs_);
   }

   return DEVICE_OK;
}

int EnderscopeZStage::OnStepSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stepSizeUm_);
   }
   else if (eAct == MM::AfterSet)
   {
      double value = 0.0;
      pProp->Get(value);
      if (value <= 0.0)
      {
         pProp->Set(stepSizeUm_);
         return DEVICE_INVALID_INPUT_PARAM;
      }
      stepSizeUm_ = value;
   }

   return DEVICE_OK;
}
