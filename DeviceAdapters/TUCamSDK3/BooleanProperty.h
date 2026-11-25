#ifndef _BOOLEANPROPERTY_H_
#define _BOOLEANPROPERTY_H_

#include "MMTUCam.h"

class CMMTUCam;

static const char * const g_StatusON = "On";
static const char * const g_StatusOFF = "Off";

class TuBooleanProperty
{
public:
   TuBooleanProperty(const std::string &MM_name,
                     TUC_HDEVICE hDevice,
                     const std::string & featureName,
                     CMMTUCam* callback, bool readOnly);
   ~TuBooleanProperty();

protected:
   int OnBoolean(MM::PropertyBase* pProp, MM::ActionType eAct);
   typedef MM::Action<TuBooleanProperty> CPropertyAction;
   void initialise(bool readOnly);
   virtual MM::ActionFunctor* CreatePropertyAction();

   TuBooleanProperty(){};

   std::string featureName_;
   CMMTUCam* callback_;
   std::string MM_name_;
   TUC_HDEVICE m_hDevice;
};

#endif // include only once
