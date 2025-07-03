/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASITIRF.h"

TIRF::TIRF() :
    ASIBase(this, ""),
    axis_("F"), // normally the TIRF axis is the F axis
    answerTimeoutMs_(1000),
    scaleFactor_(1),
    unitFactor_(10000)
{
    InitializeDefaultErrorMessages();

    // create pre-initialization properties
    // ------------------------------------

    // Name
    CreateProperty(MM::g_Keyword_Name, g_TIRFDeviceName, MM::String, true);

    // Description
    CreateProperty(MM::g_Keyword_Description, g_TIRFDeviceDescription, MM::String, true);

    // Port
    CPropertyAction* pAct = new CPropertyAction(this, &TIRF::OnPort);
    CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

    // Axis
    pAct = new CPropertyAction(this, &TIRF::OnAxis);
    CreateProperty("Axis", "F", MM::String, false, pAct, true);
    AddAllowedValue("Axis", "A");
    AddAllowedValue("Axis", "B");
    AddAllowedValue("Axis", "C");
    AddAllowedValue("Axis", "F");

    pAct = new CPropertyAction(this, &TIRF::OnScaleFactor);
    CreateProperty("ScaleFactor(mm)", "3.0", MM::Float, false, pAct, true);
}

TIRF::~TIRF()
{
    Shutdown();
}

void TIRF::GetName(char* Name) const
{
    CDeviceUtils::CopyLimitedString(Name, g_TIRFDeviceName);
}

bool TIRF::SupportsDeviceDetection()
{
    return true;
}

MM::DeviceDetectionStatus TIRF::DetectDevice()
{
    return ASIDetectDevice(*this, *GetCoreCallback(), port_, answerTimeoutMs_);
}

int TIRF::Initialize()
{
    core_ = GetCoreCallback();

    // empty the Rx serial buffer before sending command
    ClearPort();

    // check status first (test for communication protocol)
    int ret = CheckDeviceStatus();
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    ret = GetVersion(version_);
    if (ret != DEVICE_OK)
       return ret;
    CPropertyAction* pAct = new CPropertyAction(this, &TIRF::OnVersion);
    CreateProperty("Version", version_.c_str(), MM::String, true, pAct);

    // get the firmware version data from cached value
    versionData_ = ParseVersionString(version_);

    ret = GetCompileDate(compileDate_);
    if (ret != DEVICE_OK)
    {
        return ret;
    }
    pAct = new CPropertyAction(this, &TIRF::OnCompileDate);
    CreateProperty("CompileDate", "", MM::String, true, pAct);

    // if really old firmware then don't get build name
    // build name is really just for diagnostic purposes anyway
    // I think it was present before 2010 but this is easy way

    // previously compared against compile date (2010, 1, 1)
    if (versionData_.IsVersionAtLeast(8, 8, 'a'))
    {
        ret = GetBuildName(buildName_);
        if (ret != DEVICE_OK)
        {
            return ret;
        }
        pAct = new CPropertyAction(this, &TIRF::OnBuildName);
        CreateProperty("BuildName", "", MM::String, true, pAct);
    }

    pAct = new CPropertyAction(this, &TIRF::OnAngle);
    CreateProperty("TIRFAngle(deg)", "0.0", MM::Float, false, pAct);
    SetPropertyLimits("TIRFAngle(deg)", -90, 90);

    initialized_ = true;
    return DEVICE_OK;
}

int TIRF::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool TIRF::Busy()
{
    // empty the Rx serial buffer before sending command
    ClearPort();

    // send status command
    std::string answer;
    int ret = QueryCommand("/", answer);
    if (ret != DEVICE_OK)
    {
        return false;
    }

    return !answer.empty() && answer.front() == 'B';
}

double TIRF::GetAngle()
{
    // empty the Rx serial buffer before sending command
    ClearPort();

    std::ostringstream command;
    command << "W " << axis_;

    std::string answer;
    // query the device
    int ret = QueryCommand(command.str().c_str(), answer);
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
    {
        int errNo = atoi(answer.substr(2).c_str());
        return ERR_OFFSET + errNo;
    }
    else if (answer.length() > 0)
    {
        double position;
        char head[64];
        char iBuf[256];
        strcpy(iBuf, answer.c_str());
        (void)sscanf(iBuf, "%s %lf\r\n", head, &position);

        return asin(position / (scaleFactor_ * unitFactor_)) * 180 / 3.141592653589793;
    }
    return 0.0;
}

int TIRF::SetAngle(double angle)
{
    // empty the Rx serial buffer before sending command
    ClearPort();

    std::ostringstream command;
    command << std::fixed << "M " << axis_ << "=" << scaleFactor_ * unitFactor_ * sin(angle / 180 * 3.141592653589793);

    std::string answer;
    // query the device
    int ret = QueryCommand(command.str().c_str(), answer);
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
    {
        return DEVICE_OK;
    }
    // deal with error later
    else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
    {
        int errNo = atoi(answer.substr(4).c_str());
        return ERR_OFFSET + errNo;
    }
    return ERR_UNRECOGNIZED_ANSWER;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int TIRF::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(port_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        if (initialized_)
        {
            // revert
            pProp->Set(port_.c_str());
            return ERR_PORT_CHANGE_FORBIDDEN;
        }
        pProp->Get(port_);
    }
    return DEVICE_OK;
}

int TIRF::OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(axis_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(axis_);
    }
    return DEVICE_OK;
}

int TIRF::OnAngle(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(GetAngle());
    }
    else if (eAct == MM::AfterSet)
    {
        double angle;
        pProp->Get(angle);
        SetAngle(angle);
    }
    return DEVICE_OK;
}

int TIRF::OnScaleFactor(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(scaleFactor_);
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        scaleFactor_ = val;
    }
    return DEVICE_OK;
}
