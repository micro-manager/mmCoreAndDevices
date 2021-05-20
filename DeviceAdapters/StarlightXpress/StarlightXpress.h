/*
* FILE:   StarlightXpress.h
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

#ifndef STARLIGHT_XPRESS_H
#define STARLIGHT_XPRESS_H

#include <DeviceBase.h>
#include <string>


class StarlightXpressFilterWheel : public CStateDeviceBase<StarlightXpressFilterWheel> {
   public: 
      static const char* device_name;
      static const char* device_desc;

      StarlightXpressFilterWheel();

      int Initialize();
      int Shutdown();

      void GetName(char* pszName) const;
      bool Busy();
      unsigned long GetNumberOfPositions() const;

      int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnRunCalibration(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnNFilters(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnPollDelay(MM::PropertyBase* pProp, MM::ActionType eAct);

   private:
      static const char *filterCalibrationModeName;
      static const char *filterNumberName;
      static const char *autoValue;
      static const char *manualValue;
      static const char *pollDelayName;

      struct Command { 
         Command(unsigned char fst, unsigned char snd) : fst(fst), snd(snd) {}
         unsigned char fst; unsigned char snd; 

         const static Command GetNFilters;
         const static Command GetCurrentFilter;
         static Command SetCurrentFilter(unsigned char n) { return Command(n + 129, 0); }
      };
      struct Response {
         Response(unsigned char fst, unsigned char snd) : fst(fst), snd(snd) {}
         unsigned char fst; unsigned char snd; 
      };

      Response send(Command cmd);

      int get_n_filters();
      int get_current_filter();
      void set_current_filter(unsigned char filter);

      std::string m_port;
      bool m_busy;
      bool m_initialised;
      bool m_runCalibration;
      int m_n_filters;
      unsigned long m_response_timeout_ms; 
      int m_current_filter;
      bool m_current_filter_dirty;
      int m_poll_delay_ms;
};

#endif
