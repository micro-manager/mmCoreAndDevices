#ifndef _COMMANDPROPERTY_H_
#define _COMMANDPROPERTY_H_

#include "MMTUCam.h"

class CMMTUCam;

class TuCommandProperty
{
public:
   TuCommandProperty(const std::string & MM_name,
                     TUC_HDEVICE hDevice,
                     const std::string & featureName,
                     CMMTUCam* callback, bool readOnly);
   ~TuCommandProperty();

protected:
   int OnCommand(MM::PropertyBase* pProp, MM::ActionType eAct);
   typedef MM::Action<TuCommandProperty> CPropertyAction;
   void initialise(bool readOnly);
   virtual MM::ActionFunctor* CreatePropertyAction();

   TuCommandProperty(){};

   std::string featureName_;
   CMMTUCam* callback_;
   std::string MM_name_;
   TUC_HDEVICE m_hDevice;
};

#endif // include only once
