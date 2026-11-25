#ifndef _CSTRINGPROPERTY_H_
#define _CSTRINGPROPERTY_H_

#include "MMTUCam.h"

class CMMTUCam;

class TuCstringProperty
{
public:
   TuCstringProperty(const std::string & MM_name,
                     TUC_HDEVICE hDevice,
                     const std::string & featureName,
                     TUC_FEATURE_TYPE featureType,
                     TUC_FEATURE_REPRESENTATION representation,
                     const std::string & initialValue,
                     CMMTUCam* callback, bool readOnly);
   ~TuCstringProperty();

protected:
   int OnStringFun(MM::PropertyBase * pProp, MM::ActionType eAct);
   typedef MM::Action<TuCstringProperty> CPropertyAction;
   void initialise(bool readOnly);

   TuCstringProperty(){};

   std::string featureName_;
   TUC_FEATURE_TYPE featureType_;
   TUC_FEATURE_REPRESENTATION representation_;
   std::string initialValue_;
   CMMTUCam* callback_;
   std::string MM_name_;
   TUC_HDEVICE m_hDevice;
};

#endif // include only once
