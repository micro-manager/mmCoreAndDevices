#ifndef _ENUMPROPERTY_H_
#define _ENUMPROPERTY_H_

#include "MMTUCam.h"

class CMMTUCam;

class TuEnumProperty
{
public:
   TuEnumProperty(const std::string & MM_name,
                  TUC_HDEVICE hDevice,
                  const std::string & featureName,
                  CMMTUCam* callback,
                  bool readOnly, bool needsCallBack = false);
   ~TuEnumProperty();

protected:
   int OnEnum(MM::PropertyBase* pProp, MM::ActionType eAct);
   typedef MM::Action<TuEnumProperty> CPropertyAction;
   void initialise(bool readOnly);

private:
	void UpdatePixelFormatAllowedValues();
    bool CollectEnumValues(std::vector<std::string>& values);

   bool callBackRegistered_;
   std::string featureName_;
   CMMTUCam* callback_;
   std::string MM_name_;
   TUC_HDEVICE m_hDevice;
};

#endif // _ENUMPROPERTY_H_
