/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASIMAGNIFIER_H_
#define _ASIMAGNIFIER_H_

#include "ASIBase.h"

class Magnifier : public CMagnifierBase<Magnifier>, public ASIBase
{
public:
    Magnifier();
    ~Magnifier();

    // Device API
    // ----------
    int Initialize();
    int Shutdown();

    void GetName(char* pszName) const;
    bool Busy();

    bool SupportsDeviceDetection(void);
    MM::DeviceDetectionStatus DetectDevice(void);

    // Magnifier API
    //------------
    MM::DeviceType GetType();
    double GetMagnification();

    // action interface
    // ----------------
    int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMagnification(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    int SetMagnification(double mag);

    std::string axis_;
    int answerTimeoutMs_;
};

#endif // _ASIMAGNIFIER_H_
