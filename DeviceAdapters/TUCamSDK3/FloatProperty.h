#ifndef _FLOATPROPERTY_H_
#define _FLOATPROPERTY_H_

#include "MMTUCam.h"

class CMMTUCam;

class TuFloatProperty
{
public:
   TuFloatProperty(const std::string & MM_name,
                   TUC_HDEVICE hDevice,
                   const std::string & featureName,
                   double minVal, double maxVal,
                   CMMTUCam* callback,
                   bool readOnly, bool needsCallBack = false);
   ~TuFloatProperty();

protected:
   int OnFloat(MM::PropertyBase* pProp, MM::ActionType eAct);
   typedef MM::Action<TuFloatProperty> CPropertyAction;

private:
   bool callBackRegistered_;
   std::string featureName_;
   CMMTUCam* callback_;
   std::string MM_name_;
   TUC_HDEVICE m_hDevice;
};

#endif // _FLOATPROPERTY_H_
