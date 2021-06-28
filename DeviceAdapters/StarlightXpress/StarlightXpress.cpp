/*
* FILE:   StarlightXpress.cpp
* AUTHOR: Elliot Steele, April 2021
* 
* Copyright 2021 Elliot Steele
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
* copies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all 
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
* PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "StarlightXpress.h"

#include <sstream>

const char *StarlightXpressFilterWheel::device_name = "Filter Wheel";
const char *StarlightXpressFilterWheel::device_desc = "Starlight Xpress Filter Wheel";
const char *StarlightXpressFilterWheel::filterCalibrationModeName = "Filter Calibration Mode";
const char *StarlightXpressFilterWheel::filterNumberName = "Number of Filters";
const char *StarlightXpressFilterWheel::autoValue = "Auto";
const char *StarlightXpressFilterWheel::manualValue = "Manual";
const char* StarlightXpressFilterWheel::pollDelayName = "Poll Delay (ms)";

const StarlightXpressFilterWheel::Command StarlightXpressFilterWheel::Command::GetNFilters(0, 1);
const StarlightXpressFilterWheel::Command StarlightXpressFilterWheel::Command::GetCurrentFilter(0, 0);

#define SXPR_ERROR 108903
#define SXPR_PORT_CHANGE_FORBIDDEN 108904
#define SXPR_N_FILTERS_CHANGE_FORBIDDEN 108905

#ifdef min
#undef min
#endif // min

#ifdef max
#undef max
#endif // max



MODULE_API void InitializeModuleData()
{
   RegisterDevice(StarlightXpressFilterWheel::device_name, MM::StateDevice, StarlightXpressFilterWheel::device_desc);
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, StarlightXpressFilterWheel::device_name) == 0)
   {
      StarlightXpressFilterWheel* device = new StarlightXpressFilterWheel();
      return device;
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

StarlightXpressFilterWheel::StarlightXpressFilterWheel()
   : m_port("Undefined"),
   m_initialised(false),
   m_busy(false),
   m_runCalibration(true),
   m_n_filters(0),
   m_response_timeout_ms(1000),
   m_poll_delay_ms(2000),
   m_current_filter_dirty(true)
{
   InitializeDefaultErrorMessages();
   SetErrorText(SXPR_PORT_CHANGE_FORBIDDEN, "Port change is forbidden after initialization");

   CreateProperty(MM::g_Keyword_Name, device_name, MM::String, true);
   CreateProperty(MM::g_Keyword_Description, device_desc, MM::String, true);


   CPropertyAction* pAct = new CPropertyAction (this, &StarlightXpressFilterWheel::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

   pAct = new CPropertyAction(this, &StarlightXpressFilterWheel::OnRunCalibration);
   CreateProperty(filterCalibrationModeName, autoValue, MM::String, false, pAct, true);
   AddAllowedValue(filterCalibrationModeName, autoValue);
   AddAllowedValue(filterCalibrationModeName, manualValue);

   pAct = new CPropertyAction(this, &StarlightXpressFilterWheel::OnNFilters);
   CreateIntegerProperty(filterNumberName, 0, false, pAct, true);
}

int StarlightXpressFilterWheel::Initialize()
{
   PurgeComPort(m_port.c_str());

   if (m_n_filters == 0) {
      try {
         m_n_filters = get_n_filters();
      }
      catch (const std::runtime_error& e) {
         SetErrorText(SXPR_ERROR, e.what());
         return SXPR_ERROR;
      }
   }

   LogMessage(std::string("N Filters : ") + CDeviceUtils::ConvertToString(m_n_filters));

   int ret = CreateIntegerProperty(MM::g_Keyword_Closed_Position, 0, false);
   if (ret != DEVICE_OK)
      return ret;

   for (int i = 0; i < m_n_filters; i++) {
      std::stringstream ss;
      ss << "Filter-" << i;
      SetPositionLabel(i, ss.str().c_str());
      AddAllowedValue(MM::g_Keyword_Closed_Position, CDeviceUtils::ConvertToString(i));
   }

   CPropertyAction* pAct = new CPropertyAction(this, &StarlightXpressFilterWheel::OnState);
   ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   pAct = new CPropertyAction(this, &CStateBase::OnLabel);
   ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   pAct = new CPropertyAction(this, &StarlightXpressFilterWheel::OnPollDelay);
   ret = CreateIntegerProperty(StarlightXpressFilterWheel::pollDelayName, m_poll_delay_ms, false, pAct);
   if (ret != DEVICE_OK)
       return ret;

   m_initialised = true;
   return DEVICE_OK;
}

int StarlightXpressFilterWheel::Shutdown() 
{ 
   return DEVICE_OK;
}

void StarlightXpressFilterWheel::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, StarlightXpressFilterWheel::device_name);
}

bool StarlightXpressFilterWheel::Busy()
{
   return m_busy;
}

unsigned long StarlightXpressFilterWheel::GetNumberOfPositions() const
{
   return m_n_filters;
}

int StarlightXpressFilterWheel::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      try {
         unsigned char filter = get_current_filter();
         pProp->Set(static_cast<long>(filter));
      }
      catch (const std::runtime_error& e) {
         SetErrorText(SXPR_ERROR, e.what());
         return SXPR_ERROR;
      }
   }
   else if (eAct == MM::AfterSet) {
      long filter = 0;
      pProp->Get(filter);

      try {
         set_current_filter(static_cast<unsigned char>(filter));
      }
      catch (const std::runtime_error& e) {
         SetErrorText(SXPR_ERROR, e.what());
         return SXPR_ERROR;
      }
   }

   return DEVICE_OK;
}

int StarlightXpressFilterWheel::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(m_port.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (m_initialised)
      {
         // revert
         pProp->Set(m_port.c_str());
         return SXPR_PORT_CHANGE_FORBIDDEN;
      }
      pProp->Get(m_port);
   }

   return DEVICE_OK;

}

int StarlightXpressFilterWheel::OnRunCalibration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(m_runCalibration ? autoValue : manualValue);
   }
   else {
      std::string val;
      pProp->Get(val);

      if (val == autoValue) {
         m_runCalibration = true;
      }
      else if (val == manualValue) {
         m_runCalibration = false;
      }
      else {
         SetErrorText(SXPR_ERROR, ("Invalid value " + val + " for Run Calibration Property").c_str());
         return SXPR_ERROR;
      }
   }
   return DEVICE_OK;
}

int StarlightXpressFilterWheel::OnNFilters(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(static_cast<long>(m_n_filters));
   }
   else {
      if (m_initialised) {
         SetErrorText(SXPR_N_FILTERS_CHANGE_FORBIDDEN, "Cannot change number of filters after initialisation!");
         return SXPR_N_FILTERS_CHANGE_FORBIDDEN;
      }
      else if (m_runCalibration && m_n_filters != 0) {
         SetErrorText(SXPR_N_FILTERS_CHANGE_FORBIDDEN, "Cannot change number of filters when in automatic mode!");
         return SXPR_N_FILTERS_CHANGE_FORBIDDEN;
      }
      else {
         long v;
         pProp->Get(v);
         m_n_filters = static_cast<int>(v);
      }
   }

   return DEVICE_OK;
}

int StarlightXpressFilterWheel::OnPollDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet) {
        pProp->Set(static_cast<long>(m_poll_delay_ms));
    }
    else if (eAct == MM::AfterSet) {
        long v;
        pProp->Get(v);
        m_poll_delay_ms = static_cast<int>(v);
    }

    return DEVICE_OK;
}

StarlightXpressFilterWheel::Response StarlightXpressFilterWheel::send(Command cmd)
{
   PurgeComPort(m_port.c_str());
   std::stringstream ss;
   ss << "Sending Command : {" << (int)cmd.fst << ", " << (int)cmd.snd << "}";
   LogMessage(ss.str(), true);

   unsigned char data[2] = { cmd.fst, cmd.snd };

   int ret = WriteToComPort(m_port.c_str(), data, 2) != DEVICE_OK;
   if (ret) {
      LogMessage("Failed to send command to Starlight Xpress Filter Wheel!", true);
      throw std::runtime_error(std::string("Failed to send command to Starlight Xpress Filter Wheel! Error : ") + CDeviceUtils::ConvertToString(ret));
   }

   unsigned long bytes_read = 0;
   MM::TimeoutMs timeout(GetCurrentMMTime(), MM::MMTime(m_response_timeout_ms * 1000.0));
   while (bytes_read < 2) {
      if (timeout.expired(GetCurrentMMTime())) {
         throw std::runtime_error("Timed out while awaiting response from Starlight Xpress Filter Wheel!");
      }

      int ret = ReadFromComPort(m_port.c_str(), data + bytes_read, 2 - bytes_read, bytes_read);
      if (ret != DEVICE_OK) {
         throw std::runtime_error("Failed to read response from Starlight Xpress Filter Wheel!");
      }
   }

   ss.str(std::string());
   ss << "Recieved Response : {" << (int)data[0] << ", " << (int)data[1] << "}";
   LogMessage(ss.str(), true);

   return Response(data[0], data[1]);
}

int StarlightXpressFilterWheel::get_n_filters()
{
   LogMessage("GET N FILTERS START", true);
   m_busy = true;
   Response r = m_runCalibration ? send(Command::GetNFilters) : send(Command::GetCurrentFilter);
   while (r.fst == 0) {
      CDeviceUtils::SleepMs(1000);
      r = send(Command::GetCurrentFilter);
   }
   m_busy = false;
   LogMessage("GET N FILTERS END", true);
   return r.snd;
}

int StarlightXpressFilterWheel::get_current_filter()
{
   LogMessage("GET CURRENT FILTER START", true);

   if (!m_current_filter_dirty) {
	   LogMessage("GET CURRENT FILTER END", true);
       return m_current_filter;
   }

   m_busy = true;
   Response r = send(Command::GetCurrentFilter);
   while (r.fst == 0) {
      CDeviceUtils::SleepMs(250);
      r = send(Command::GetCurrentFilter);
   }

   m_current_filter = r.fst - 1;
   m_current_filter_dirty = false;
   m_busy = false;

   LogMessage("GET CURRENT FILTER END", true);

   return r.fst - 1;
}

void StarlightXpressFilterWheel::set_current_filter(unsigned char n)
{
   LogMessage("SET CURRENT FILTER START", true);
   m_current_filter_dirty = true;
   m_busy = true;
   send(Command::SetCurrentFilter(n));

   const int filter_distance = std::labs(m_current_filter - n);
   const int delay = m_poll_delay_ms * std::min(filter_distance, m_n_filters - filter_distance);

   CDeviceUtils::SleepMs(delay);
   get_current_filter();
   LogMessage("SET CURRENT FILTER END", true);
   m_busy = false;
}
