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

private:
   static const char *filterCalibrationModeName;
   static const char *filterNumberName;
   static const char *autoValue;
   static const char *manualValue;

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
};

#endif