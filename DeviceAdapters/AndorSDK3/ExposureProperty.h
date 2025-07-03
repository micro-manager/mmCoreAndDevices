#ifndef _EXPOSUREPROPERTY_H_
#define _EXPOSUREPROPERTY_H_

#include "atcore++.h"
#include "MMDeviceConstants.h"
#include "Property.h"
#include "IProperty.h"

class ICallBackManager;

class TExposureProperty : public andor::IObserver, public IProperty
{
public:
   TExposureProperty(const std::string & MM_name,
                  andor::IFloat* float_feature,
                  ICallBackManager* callback,
                  bool readOnly, bool needsCallBack);
   ~TExposureProperty();

protected:
   void Update(andor::ISubject* Subject);
   int OnFloat(MM::PropertyBase* pProp, MM::ActionType eAct);
   typedef MM::Action<TExposureProperty> CPropertyAction;

private:
   void setFeatureWithinLimits(double new_value);
   bool valueIsWithinLimits(double new_value);

private:
   andor::IFloat* float_feature_;
   ICallBackManager* callback_;
   std::string MM_name_;
   bool callbackRegistered_;
   static const int DEC_PLACES_ERROR = 4;
};

#endif