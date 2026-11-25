#ifndef _INTEGERPROPERTY_H_
#define _INTEGERPROPERTY_H_

#include "MMTUCam.h"

class CMMTUCam;

class TuIntegerProperty
{
public:
   TuIntegerProperty(const std::string & MM_name,
                     TUC_HDEVICE hDevice,
                     const std::string & featureName,
                     int64_t minVal, int64_t maxVal,
                     CMMTUCam* callback,
                     bool readOnly, bool needsCallBack = false);
   ~TuIntegerProperty();

protected:
   int OnInteger(MM::PropertyBase* pProp, MM::ActionType eAct);
   typedef MM::Action<TuIntegerProperty> CPropertyAction;

private:
   bool callBackRegistered_;
   std::string featureName_;
   CMMTUCam* callback_;
   std::string MM_name_;
   TUC_HDEVICE m_hDevice;
};

#endif // _INTEGERPROPERTY_H_
