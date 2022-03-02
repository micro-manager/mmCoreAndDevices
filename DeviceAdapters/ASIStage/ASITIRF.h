/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASITIRF_H_
#define _ASITIRF_H_

#include "ASIBase.h"

class TIRF : public CGenericBase<TIRF>, public ASIBase
{
public:
    TIRF();
    ~TIRF();

    // Device API
    int Initialize(); //Generic
    int Shutdown();   //Generic

    void GetName(char* pszName) const;
    bool Busy();
    bool SupportsDeviceDetection(void);
    MM::DeviceDetectionStatus DetectDevice(void);

    // Generic API
    //------------
    MM::DeviceType GetType();


    // action interface
    // ----------------

    int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAngle(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnScaleFactor(MM::PropertyBase* pProp, MM::ActionType eAct);


    //private:

    int SetAngle(double angle);
    double GetAngle();

    std::string axis_;
    int answerTimeoutMs_;
    double scaleFactor_;
    double unitFactor_;
};

#endif // _ASITIRF_H_
