/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASIZStage.h"

ZStage::ZStage() :
    ASIBase(this, "1H"), // LX-4000 prefix
    axis_("Z"),
    axisNr_(4),
    stepSizeUm_(0.1),
    answerTimeoutMs_(1000),
    sequenceable_(false),
    runningFastSequence_(false),
    hasRingBuffer_(false),
    nrEvents_(0),
    curSteps_(0),
    maxSpeed_(7.5),
    motorOn_(true),
    supportsLinearSequence_(false),
    linearSequenceIntervalUm_(0.0),
    linearSequenceLength_(0),
    linearSequenceTimeoutMs_(10000),
    // init cached properties
    speed_(0),
    waitCycles_(0),
    backlash_(0),
    error_(0),
    acceleration_(0),
    finishError_(0),
    overShoot_(0)
{
    InitializeDefaultErrorMessages();

    // create pre-initialization properties
    // ------------------------------------

    // Name
    CreateProperty(MM::g_Keyword_Name, g_ZStageDeviceName, MM::String, true);

    // Description
    CreateProperty(MM::g_Keyword_Description, g_ZStageDeviceDescription, MM::String, true);

    // Port
    CPropertyAction* pAct = new CPropertyAction(this, &ZStage::OnPort);
    CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

    // Axis
    pAct = new CPropertyAction(this, &ZStage::OnAxis);
    CreateProperty("Axis", "Z", MM::String, false, pAct, true);
    AddAllowedValue("Axis", "F");
    AddAllowedValue("Axis", "P");
    AddAllowedValue("Axis", "Z");
}

ZStage::~ZStage()
{
    Shutdown();
}

void ZStage::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_ZStageDeviceName);
}

bool ZStage::SupportsDeviceDetection()
{
    return true;
}

MM::DeviceDetectionStatus ZStage::DetectDevice()
{
    return ASIDetectDevice(*this, *GetCoreCallback(), port_, answerTimeoutMs_);
}

int ZStage::Initialize()
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

    // needs to be called first since it sets hasRingBuffer_
    GetControllerInfo();

    stepSizeUm_ = 0.1; //res;

    ret = GetPositionSteps(curSteps_);
    // if command fails, try one more time,
    // other devices may have send crud to this serial port during device detection
    if (ret != DEVICE_OK)
    {
        ret = GetPositionSteps(curSteps_);
    }

    ret = GetVersion(version_);
    if (ret != DEVICE_OK)
       return ret;
    CPropertyAction* pAct = new CPropertyAction(this, &ZStage::OnVersion);
    CreateProperty("Version", version_.c_str(), MM::String, true, pAct);

    // get the firmware version data from cached value
    versionData_ = ParseVersionString(version_);

    ret = GetCompileDate(compileDate_);
    if (ret != DEVICE_OK)
    {
        return ret;
    }
    pAct = new CPropertyAction(this, &ZStage::OnCompileDate);
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
        pAct = new CPropertyAction(this, &ZStage::OnBuildName);
        CreateProperty("BuildName", "", MM::String, true, pAct);
    }

    if (HasRingBuffer() && nrEvents_ == 0)
    {
        // we couldn't detect size of the ring buffer automatically so create property
        //   to allow user to change it
        pAct = new CPropertyAction(this, &ZStage::OnRingBufferSize);
        CreateProperty("RingBufferSize", "50", MM::Integer, false, pAct);
        AddAllowedValue("RingBufferSize", "50");
        AddAllowedValue("RingBufferSize", "250");
        nrEvents_ = 50;  // modified in action handler
    }
    else
    {
        std::ostringstream tmp;
        tmp.str("");
        tmp << nrEvents_;  // initialized in GetControllerInfo() if we got here
        CreateProperty("RingBufferSize", tmp.str().c_str(), MM::String, true);
    }

    if (HasRingBuffer())
    {
        pAct = new CPropertyAction(this, &ZStage::OnSequence);
        const char* spn = "Use Sequence";
        CreateProperty(spn, "No", MM::String, false, pAct);
        AddAllowedValue(spn, "No");
        AddAllowedValue(spn, "Yes");

        pAct = new CPropertyAction(this, &ZStage::OnFastSequence);
        spn = "Use Fast Sequence";
        CreateProperty(spn, "No", MM::String, false, pAct);
        AddAllowedValue(spn, "No");
        AddAllowedValue(spn, "Armed");
    }

    // The timeout for linear Z stacks (ZS F=)
    pAct = new CPropertyAction(this, &ZStage::OnLinearSequenceTimeout);
    CreateFloatProperty("LinearSequenceResetTimeout(ms)", linearSequenceTimeoutMs_, false, pAct);

    // Speed (sets both x and y)
    if (HasCommand("S " + axis_ + "?"))
    {
       ret = GetSpeed(speed_);
       if (ret != DEVICE_OK)
          return ret;
       pAct = new CPropertyAction(this, &ZStage::OnSpeed);
       CreateProperty("Speed-S", std::to_string(speed_).c_str(), MM::Float, false, pAct);
       // Maximum Speed that can be set in Speed-S property
       char max_speed[MM::MaxStrLength];
       GetMaxSpeed(max_speed);
       CreateProperty("Maximum Speed (Do Not Change)", max_speed, MM::Float, true);
    }

    // Backlash (sets both x and y)
    if (HasCommand("B " + axis_ + "?"))
    {
        ret = GetBacklash(backlash_);
        if (ret != DEVICE_OK)
           return ret;
        pAct = new CPropertyAction(this, &ZStage::OnBacklash);
        CreateProperty("Backlash-B", std::to_string(backlash_).c_str(), MM::Float, false, pAct);
    }

    // Error (sets both x and y)
    if (HasCommand("E " + axis_ + "?"))
    {
       ret = GetError(error_);
       if (ret != DEVICE_OK)
          return ret;
       pAct = new CPropertyAction(this, &ZStage::OnError);
       CreateProperty("Error-E(nm)", std::to_string(error_).c_str(), MM::Float, false, pAct);
    }

    // acceleration (sets both x and y)
    if (HasCommand("AC " + axis_ + "?"))
    {
       ret = GetAcceleration(acceleration_);
       if (ret != DEVICE_OK)
          return ret;
       pAct = new CPropertyAction(this, &ZStage::OnAcceleration);
       CreateProperty("Acceleration-AC(ms)", std::to_string(acceleration_).c_str(), MM::Integer, false, pAct);
    }

    // Finish Error (sets both x and y)
    if (HasCommand("PC " + axis_ + "?"))
    {
       ret = GetFinishError(finishError_);
       if (ret != DEVICE_OK)
          return ret;
       pAct = new CPropertyAction(this, &ZStage::OnFinishError);
       CreateProperty("FinishError-PCROS(nm)", std::to_string(finishError_).c_str(), MM::Float, false, pAct);
    }

    // OverShoot (sets both x and y)
    if (HasCommand("OS " + axis_ + "?"))
    {
       ret = GetOverShoot(overShoot_);
       if (ret != DEVICE_OK)
          return ret;
       pAct = new CPropertyAction(this, &ZStage::OnOverShoot);
       CreateProperty("OverShoot(um)", std::to_string(overShoot_).c_str(), MM::Float, false, pAct);
    }

    // MotorCtrl (works on both x and y)
    pAct = new CPropertyAction(this, &ZStage::OnMotorCtrl);
    CreateProperty("MotorOnOff", "On", MM::String, false, pAct);
    AddAllowedValue("MotorOnOff", "On");
    AddAllowedValue("MotorOnOff", "Off");

    // Wait cycles
    if (HasCommand("WT " + axis_ + "?"))
    {
       ret = GetWait(waitCycles_);
       if (ret != DEVICE_OK)
          return ret;
       pAct = new CPropertyAction(this, &ZStage::OnWait);
       CreateProperty("Wait_Cycles", std::to_string(waitCycles_).c_str(), MM::Integer, false, pAct);
        // SetPropertyLimits("Wait_Cycles", 0, 255);  // don't artificially restrict range
    }

    if (HasCommand("VE " + axis_ + "=0"))
    {
        pAct = new CPropertyAction(this, &ZStage::OnVector);
        CreateProperty("VectorMove-VE(mm/s)", "0", MM::Float, false, pAct);
        char orig_speed[MM::MaxStrLength];
        ret = GetProperty("Speed-S", orig_speed);
        double mspeed;
        if (ret != DEVICE_OK)
        {
            mspeed = 8;
        }
        else
        {
            mspeed = atof(orig_speed);
        }

        SetPropertyLimits("VectorMove-VE(mm/s)", mspeed * -1, mspeed);
    }

    initialized_ = true;
    return DEVICE_OK;
}

int ZStage::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool ZStage::Busy()
{
    if (runningFastSequence_)
    {
        return false;
    }

    // empty the Rx serial buffer before sending command
    ClearPort();

    std::string answer;
    int ret = QueryCommand("/", answer);
    if (ret != DEVICE_OK)
    {
        return false;
    }

    return !answer.empty() && answer.front() == 'B';
}

int ZStage::SetPositionUm(double pos)
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    // empty the Rx serial buffer before sending command
    ClearPort();

    std::ostringstream command;
    command << std::fixed << "M " << axis_ << "=" << pos / stepSizeUm_; // in 10ths of micros
    std::string answer;

    // query the device
    int ret = QueryCommand(command.str().c_str(), answer);
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
    {
        this->OnStagePositionChanged(pos);
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

int ZStage::GetPositionUm(double& pos)
{
    // empty the Rx serial buffer before sending command
    ClearPort();

    std::ostringstream command;
    command << "W " << axis_;
    std::string answer;

    // query command
    int ret = QueryCommand(command.str().c_str(), answer);
    if (ret != DEVICE_OK)
        return ret;

    if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
    {
        int errNo = atoi(answer.substr(2).c_str());
        return ERR_OFFSET + errNo;
    }
    else if (answer.length() > 0)
    {
        char head[64];
        float zz;
        char iBuf[256];
        strcpy(iBuf, answer.c_str());
        (void)sscanf(iBuf, "%s %f\r\n", head, &zz);

        pos = zz * stepSizeUm_;
        curSteps_ = (long)zz;

        return DEVICE_OK;
    }
    return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::SetRelativePositionUm(double d)
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    // empty the Rx serial buffer before sending command
    ClearPort();

    std::ostringstream command;
    command << std::fixed << "R " << axis_ << "=" << d / stepSizeUm_; // in 10th of micros

    std::string answer;
    // query the device
    int ret = QueryCommand(command.str().c_str(), answer);
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
    {
        // we don't know the updated position to call this
        //this->OnStagePositionChanged(pos);
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

int ZStage::SetPositionSteps(long pos)
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    std::ostringstream command;
    command << "M " << axis_ << "=" << pos; // in 10th of micros

    std::string answer;
    // query command
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

int ZStage::GetPositionSteps(long& steps)
{
    // empty the Rx serial buffer before sending command
    ClearPort();

    std::ostringstream command;
    command << "W " << axis_;
    std::string answer;

    // query command
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
        char head[64];
        float zz;
        char iBuf[256];
        strcpy(iBuf, answer.c_str());
        (void)sscanf(iBuf, "%s %f\r\n", head, &zz);

        steps = (long)zz;
        curSteps_ = (long)steps;
        return DEVICE_OK;
    }
    return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::SetOrigin()
{
    // empty the Rx serial buffer before sending command
    ClearPort();

    std::ostringstream os;
    os << "H " << axis_;
    std::string answer;

    // query command
    int ret = QueryCommand(os.str().c_str(), answer); // use command HERE, zero (z) zero all x,y,z
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
    {
        return DEVICE_OK;
    }
    else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
    {
        int errNo = atoi(answer.substr(2, 4).c_str());
        return ERR_OFFSET + errNo;
    }
    return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::Calibrate() {

    return DEVICE_OK;;
}

int ZStage::GetLimits(double& /*min*/, double& /*max*/)
{
    return DEVICE_UNSUPPORTED_COMMAND;
}

bool ZStage::HasRingBuffer() const
{
    return hasRingBuffer_;
}

int ZStage::StartStageSequence()
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    std::string answer;

    if (sequence_.empty() && supportsLinearSequence_)
    {
        std::ostringstream os;
        os.precision(0);
        os << std::fixed << "ZS X=" << 10 * linearSequenceIntervalUm_ <<
            " Y=" << linearSequenceLength_ <<
            " Z=0" << " F=" << linearSequenceTimeoutMs_;
        int ret = QueryCommand(os.str().c_str(), answer);
        if (ret != DEVICE_OK)
        {
            return ret;
        }
        if (answer.compare(0, 2, ":A") != 0 || answer.compare(1, 2, ":A") != 0)
        {
            return ERR_UNRECOGNIZED_ANSWER;
        }

        ret = QueryCommand("TTL X=4", answer);
        if (ret != DEVICE_OK)
        {
            return ret;
        }
        if (answer.compare(0, 2, ":A") != 0 || answer.compare(1, 2, ":A") != 0)
        {
            return ERR_UNRECOGNIZED_ANSWER;
        }
        return DEVICE_OK;
    }

    // ensure that ringbuffer pointer points to first entry and
    // that we only trigger the desired axis
    std::ostringstream os;
    os << "RM Y=" << axisNr_ << " Z=0";

    int ret = QueryCommand(os.str().c_str(), answer);
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
    {
        ret = QueryCommand("TTL X=1", answer); // switches on TTL triggering
        if (ret != DEVICE_OK)
        {
            return ret;
        }

        if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
        {
            return DEVICE_OK;
        }
    }
    return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::StopStageSequence()
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    std::string answer;
    int ret = QueryCommand("TTL X=0", answer); // switches off TTL triggering
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
    {
        return DEVICE_OK;
    }
    return DEVICE_OK;
}

int ZStage::SendStageSequence()
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    // first clear the buffer in the device
    std::string answer;
    int ret = QueryCommand("RM X=0", answer); // clears the ringbuffer
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
    {
        for (unsigned i = 0; i < sequence_.size(); i++)
        {
            std::ostringstream os;
            os.precision(0);
            // previously compared against compile date (2015, 10, 23)
            if (versionData_.IsVersionAtLeast(9, 2, 'i'))
            {
                os << std::fixed << "LD " << axis_ << "=" << sequence_[i] * 10;  // 10 here is for unit multiplier/1000
                ret = QueryCommand(os.str().c_str(), answer);
                if (ret != DEVICE_OK)
                {
                    return ret;
                }
            }
            else
            {
                // For WhizKid the LD reply originally was :A without <CR><LF> so
                //   send extra "empty command"  and get back :N-1 which we ignore
                // basically we are trying to compensate for the controller's faults here
                // but as of 2015-10-23 the firmware to "properly" responds with <CR><LF>
                os << std::fixed << "LD " << axis_ << "=" << sequence_[i] * 10 << "\r\n";
                ret = QueryCommand(os.str().c_str(), answer);
                if (ret != DEVICE_OK)
                {
                    return ret;
                }

                // the answer will also have a :N-1 in it, ignore.
                if (!(answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0))
                {
                    return ERR_UNRECOGNIZED_ANSWER;
                }
            }
        }
    }
    return DEVICE_OK;
}

int ZStage::ClearStageSequence()
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    sequence_.clear();

    // clear the buffer in the device
    std::string answer;
    int ret = QueryCommand("RM X=0", answer); // clears the ringbuffer
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.compare(0, 2, ":A") == 0 || answer.compare(1, 2, ":A") == 0)
    {
        return DEVICE_OK;
    }
    return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::AddToStageSequence(double position)
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    sequence_.push_back(position);
    return DEVICE_OK;
}

int ZStage::SetStageLinearSequence(double dZ_um, long nSlices)
{
    if (runningFastSequence_)
    {
        return DEVICE_OK;
    }

    int ret = ClearStageSequence();
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    linearSequenceIntervalUm_ = dZ_um;
    linearSequenceLength_ = nSlices;

    return DEVICE_OK;
}

/*
 * This function checks what is available in this controller
 * It should really be part of a Hub Device
 */
int ZStage::GetControllerInfo()
{
    std::string answer;
    int ret = QueryCommand("BU X", answer);
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    std::istringstream iss(answer);
    std::string token;
    while (getline(iss, token, '\r'))
    {
        std::string ringBuffer = "RING BUFFER";
        if (0 == token.compare(0, ringBuffer.size(), ringBuffer))
        {
            hasRingBuffer_ = true;
            if (token.size() > ringBuffer.size())
            {
                // tries to read ring buffer size, this works since 2013-09-03
                // change to firmware which prints max size
                int rsize = atoi(token.substr(ringBuffer.size()).c_str());
                if (rsize > 0)
                {
                    // only used in GetStageSequenceMaxLength as defined in .h file
                    nrEvents_ = rsize;
                }
            }
        }
        std::string ma = "Motor Axes: ";
        if (token.substr(0, ma.length()) == ma)
        {
            std::istringstream axes(token.substr(ma.length(), std::string::npos));
            std::string thisAxis;
            int i = 1;
            while (getline(axes, thisAxis, ' '))
            {
                if (thisAxis == axis_)
                {
                    axisNr_ = i;
                }
                i = i << 1;
            }
        }
        // TODO: add in tests for other capabilities/devices
    }

    LogMessage(answer.c_str(), false);

    // Determine if our axis is the "active Z-focus axis" (which allows linear sequence)
    ret = QueryCommand("UL F?", answer);
    if (ret != DEVICE_OK)
    {
        return ret;
    }

    if (answer.length() > 5 && answer.compare(0, 5, ":A F=") == 0)
    {
        int focusIndex = atoi(answer.substr(5).c_str());

        std::ostringstream cmd;
        cmd << "Z2B " << axis_ << "?";
        ret = QueryCommand(cmd.str().c_str(), answer);
        if (answer.length() > 5 && answer.compare(0, 3, ":A ") == 0)
        {
            int axisIndex = atoi(answer.substr(5).c_str());
            supportsLinearSequence_ = (focusIndex == axisIndex);
        }
    }
    return DEVICE_OK;
}

bool ZStage::HasCommand(const std::string& command)
{
    std::string answer;
    // query the device
    int ret = QueryCommand(command.c_str(), answer);
    if (ret != DEVICE_OK)
    {
        return false;
    }

    if (answer.compare(0, 2, ":A") == 0)
    {
        return true;
    }

    if (answer.compare(0, 4, ":N-1") == 0)
    {
        return false;
    }

    // if we do not get an answer, or any other answer, this is probably OK
    return true;
}

// Action handlers

int ZStage::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int ZStage::OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int ZStage::OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        if (sequenceable_)
        {
            pProp->Set("Yes");
        }
        else
        {
            pProp->Set("No");
        }
    }
    else if (eAct == MM::AfterSet)
    {
        std::string prop;
        pProp->Get(prop);
        sequenceable_ = false;
        if (prop == "Yes")
        {
            sequenceable_ = true;
        }
    }
    return DEVICE_OK;
}

int ZStage::OnFastSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret;

    if (eAct == MM::BeforeGet)
    {
        if (runningFastSequence_)
        {
            pProp->Set("Armed");
        }
        else
        {
            pProp->Set("No");
        }
    }
    else if (eAct == MM::AfterSet)
    {
        std::string prop;
        pProp->Get(prop);

        // only let user do fast sequence if regular one is enabled
        if (!sequenceable_) {
            pProp->Set("No");
            return DEVICE_OK;
        }

        if (prop.compare("Armed") == 0)
        {
            runningFastSequence_ = false;
            ret = SendStageSequence();
            if (ret)
            {
                return ret;  // same as RETURN_ON_MM_ERROR
            }
            ret = StartStageSequence();
            if (ret)
            {
                return ret;  // same as RETURN_ON_MM_ERROR
            }
            runningFastSequence_ = true;
        }
        else
        {
            runningFastSequence_ = false;
            ret = StopStageSequence();
            if (ret)
            {
                return ret;  // same as RETURN_ON_MM_ERROR
            }
        }
    }

    return DEVICE_OK;
}

int ZStage::OnRingBufferSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(nrEvents_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(nrEvents_);
    }
    return DEVICE_OK;
}

int ZStage::OnLinearSequenceTimeout(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(linearSequenceTimeoutMs_);
    }
    else if (eAct == MM::AfterSet)
    {
        double v;
        pProp->Get(v);
        if (v < 0)
        {
            v = 0.0;
        }
        linearSequenceTimeoutMs_ = long(ceil(v));
    }
    return DEVICE_OK;
}

int ZStage::GetWait(long& waitCycles)
{
   // To simplify our life we only read out waitcycles for the X axis, but set for both
   std::ostringstream command;
   command << "WT " << axis_ << "?";
   std::string answer;
   // query command
   int ret = QueryCommand(command.str().c_str(), answer);
   if (ret != DEVICE_OK)
      return ret;

   if (answer.compare(0, 2, ":" + axis_) == 0)
   {
      return ParseResponseAfterPosition(answer, 3, waitCycles);
   }
   // deal with error later
   else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
   {
      int errNo = atoi(answer.substr(3).c_str());
      return ERR_OFFSET + errNo;
   }
   return ERR_UNRECOGNIZED_ANSWER;
}

// This sets the number of waitcycles
int ZStage::OnWait(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
       pProp->Set(waitCycles_);
    }
    else if (eAct == MM::AfterSet)
    {
        long waitCycles;
        pProp->Get(waitCycles);

        // enforce positive
        if (waitCycles < 0)
        {
            waitCycles = 0;
        }

        // if firmware date is 2009+  then use msec/int definition of WaitCycles
        // would be better to parse firmware (8.4 and earlier used unsigned char)
        // and that transition occurred ~2008 but this is easier than trying to
        // parse version strings

        // previously compared against compile date (2009, 1, 1)
        if (versionData_.IsVersionAtLeast(8, 6, 'd'))
        {
            // don't enforce upper limit
        }
        else  // enforce limit for 2008 and earlier firmware or
        {     // if getting compile date wasn't successful
            if (waitCycles > 255)
            {
                waitCycles = 255;
            }
        }

        std::ostringstream command;
        command << "WT " << axis_ << "=" << waitCycles;
        std::string answer;
        // query command
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
            return ret;
        ret = ResponseStartsWithColonA(answer);
        if (ret != DEVICE_OK)
            return ret;
        waitCycles_ = waitCycles;
    }
    return DEVICE_OK;
}

int ZStage::GetBacklash(double& backlash)
{
   // To simplify our life we only read out waitcycles for the X axis, but set for both
   std::ostringstream command;
   command << "B " << axis_ << "?";
   std::string answer;
   // query command
   int ret = QueryCommand(command.str().c_str(), answer);
   if (ret != DEVICE_OK)
      return ret;

   if (answer.compare(0, 2, ":" + axis_) == 0)
   {
      return ParseResponseAfterPosition(answer, 3, 8, backlash);
   }
   // deal with error later
   else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
   {
      int errNo = atoi(answer.substr(3).c_str());
      return ERR_OFFSET + errNo;
   }
   return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::OnBacklash(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
       pProp->Set(backlash_);
    }
    else if (eAct == MM::AfterSet)
    {
        double backlash;
        pProp->Get(backlash);
        if (backlash < 0.0)
        {
            backlash = 0.0;
        }
        std::ostringstream command;
        command << "B " << axis_ << "=" << backlash;
        std::string answer;
        // query command
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
            return ret;
        ret = ResponseStartsWithColonA(answer);
        if (ret != DEVICE_OK)
            return ret;
        backlash_ = backlash;
    }
    return DEVICE_OK;
}

int ZStage::GetFinishError(double& finishError)
{
   // To simplify our life we only read out waitcycles for the X axis, but set for both
   std::ostringstream command;
   command << "PC " << axis_ << "?";
   std::string answer;
   // query command
   int ret = QueryCommand(command.str().c_str(), answer);
   if (ret != DEVICE_OK)
      return ret;

   if (answer.compare(0, 2, ":" + axis_) == 0)
   {
      double tmp = 0.0;
      const int code = ParseResponseAfterPosition(answer, 3, 8, tmp);
      finishError = 1000000 * tmp;
      return code;
  }
  if (answer.compare(0, 2, ":A") == 0)
  {
      // Answer is of the form :A X=0.00003
      double tmp = 0.0;
      const int code = ParseResponseAfterPosition(answer, 5, 8, tmp);
      finishError = 1000000 * tmp;
      return code;
  }
  // deal with error later
  else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
  {
      int errNo = atoi(answer.substr(3).c_str());
      return ERR_OFFSET + errNo;
  }
  return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::OnFinishError(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
       pProp->Set(finishError_);
    }
    else if (eAct == MM::AfterSet)
    {
        double error;
        pProp->Get(error);
        if (error < 0.0)
        {
            error = 0.0;
        }
        double tmp = error / 1000000;
        std::ostringstream command;
        command << "PC " << axis_ << "=" << tmp;
        std::string answer;
        // query command
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
            return ret;
        ret = ResponseStartsWithColonA(answer);
        if (ret != DEVICE_OK)
            return ret;
    }
    return DEVICE_OK;
}

int ZStage::GetAcceleration(long& acceleration)
{
   // To simplify our life we only read out acceleration for the X axis, but set for both
   std::ostringstream command;
   command << "AC " + axis_ + "?";
   std::string answer;

  // query command
  int ret = QueryCommand(command.str().c_str(), answer);
  if (ret != DEVICE_OK)
  {
      return ret;
  }

  if (answer.compare(0, 2, ":" + axis_) == 0)
  {
      double tmp = 0.0;
      const int code = ParseResponseAfterPosition(answer, 3, 8, tmp);
      acceleration = (long) tmp;
      return code;
  }
  // deal with error later
  else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
  {
      int errNo = atoi(answer.substr(3).c_str());
      return ERR_OFFSET + errNo;
  }
  return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
       pProp->Set(acceleration_);
    }
    else if (eAct == MM::AfterSet)
    {
        long accel;
        pProp->Get(accel);
        if (accel < 0)
        {
            accel = 0;
        }
        std::ostringstream command;
        command << "AC " << axis_ << "=" << accel;
        std::string answer;
        // query command
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
            return ret;
        ret = ResponseStartsWithColonA(answer);
        if (ret != DEVICE_OK)
            return ret;
        acceleration_ = accel;
    }

    return DEVICE_OK;
}

int ZStage::GetOverShoot(double& overShoot)
{
   std::ostringstream command;
   command << "OS " + axis_ + "?";
   std::string answer;
   // query command
   int ret = QueryCommand(command.str().c_str(), answer);
   if (ret != DEVICE_OK)
      return ret;

   if (answer.compare(0, 2, ":A") == 0)
   {
      return ParseResponseAfterPosition(answer, 5, 8, overShoot);
   }
   // deal with error herer
   else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
   {
       int errNo = atoi(answer.substr(3).c_str());
       return ERR_OFFSET + errNo;
   }
   return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::OnOverShoot(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
       pProp->Set(overShoot_);
    }
    else if (eAct == MM::AfterSet)
    {
        double overShoot;
        pProp->Get(overShoot);
        if (overShoot < 0.0)
        {
            overShoot = 0.0;
        }
        overShoot = overShoot / 1000.0;
        std::ostringstream command;
        command << std::fixed << "OS " << axis_ << "=" << overShoot;
        std::string answer;
        // query the device
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
            return ret;
        ret = ResponseStartsWithColonA(answer);
        if (ret != DEVICE_OK)
            return ret;
        overShoot_ = overShoot;
    }
    return DEVICE_OK;
}

int ZStage::GetError(double& error)
{
   // To simplify our life we only read out waitcycles for the X axis, but set for both
   std::ostringstream command;
   command << "E " + axis_ + "?";
   std::string answer;
   // query command
   int ret = QueryCommand(command.str().c_str(), answer);
   if (ret != DEVICE_OK)
       return ret;

   if (answer.compare(0, 2, ":" + axis_) == 0)
   {
       double tmp = 0.0;
       const int code = ParseResponseAfterPosition(answer, 3, 8, tmp);
       error  = tmp * 1000000.0;
       return code;
   }
   // deal with error later
   else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
   {
      int errNo = atoi(answer.substr(3).c_str());
      return ERR_OFFSET + errNo;
   }
   return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::OnError(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
       pProp->Set(error_);
    }
    else if (eAct == MM::AfterSet)
    {
        double error;
        pProp->Get(error);
        if (error < 0.0)
        {
            error = 0.0;
        }
        double tmp = error / 1000000.0;
        std::ostringstream command;
        command << std::fixed << "E " << axis_ << "=" << tmp;
        std::string answer;
        // query the device
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
            return ret;
        ret = ResponseStartsWithColonA(answer);
        if (ret != DEVICE_OK)
            return ret;
        error_ = error;
    }
    return DEVICE_OK;
}

int ZStage::GetMaxSpeed(char* maxSpeedStr)
{
    double origMaxSpeed = maxSpeed_;
    char orig_speed[MM::MaxStrLength];
    int ret = GetProperty("Speed-S", orig_speed);
    if (ret != DEVICE_OK)
    {
        return ret;
    }
    maxSpeed_ = 10001;
    SetProperty("Speed-S", "10000");
    ret = GetProperty("Speed-S", maxSpeedStr);
    maxSpeed_ = origMaxSpeed;  // restore in case we return early
    if (ret != DEVICE_OK)
    {
        return ret;
    }
    ret = SetProperty("Speed-S", orig_speed);
    if (ret != DEVICE_OK)
    {
        return ret;
    }
    return DEVICE_OK;
}

int ZStage::GetSpeed(double& speed)
{
   // To simplify our life we only read out waitcycles for the X axis, but set for both
   std::ostringstream command;
   command << "S " + axis_ + "?";
   std::string answer;
   // query command
   int ret = QueryCommand(command.str().c_str(), answer);
   if (ret != DEVICE_OK)
      return ret;

   if (answer.compare(0, 2, ":A") == 0)
   {
      return ParseResponseAfterPosition(answer, 5, speed);
   }
   // deal with error later
   else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
   {
      int errNo = atoi(answer.substr(3).c_str());
      return ERR_OFFSET + errNo;
   }
   return ERR_UNRECOGNIZED_ANSWER;
}

int ZStage::OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
       pProp->Set(speed_);
    }
    else if (eAct == MM::AfterSet)
    {
        double speed;
        pProp->Get(speed);
        if (speed < 0.0)
        {
            speed = 0.0;
        }
        else if (speed > maxSpeed_)
        {
            // Note: max speed may differ depending on pitch screw
            speed = maxSpeed_;
        }
        std::ostringstream command;
        command << std::fixed << "S " << axis_ << "=" << speed;
        std::string answer;
        // query the device
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
            return ret;
        ret = ResponseStartsWithColonA(answer);
        if (ret != DEVICE_OK)
            return ret;
        speed_ = speed;
    }
    return DEVICE_OK;
}

int ZStage::OnMotorCtrl(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        // The controller can not report whether or not the motors are on.  Cache the value
        if (motorOn_)
        {
            pProp->Set("On");
        }
        else
        {
            pProp->Set("Off");
        }
        return DEVICE_OK;
    }
    else if (eAct == MM::AfterSet)
    {
        std::string motorOn;
        std::string value;
        pProp->Get(motorOn);
        if (motorOn == "On")
        {
            motorOn_ = true;
            value = "+";
        }
        else
        {
            motorOn_ = false;
            value = "-";
        }
        std::ostringstream command;
        command << "MC " << axis_ << value;
        std::string answer;

        // query command
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
        {
            return ret;
        }
        return ResponseStartsWithColonA(answer);
    }
    return DEVICE_OK;
}

int ZStage::OnVector(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        // To simplify our life we only read out acceleration for the X axis, but set for both

        std::ostringstream command;
        command << "VE " + axis_ + "?";
        std::string answer;
        // query command
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
        {
            return ret;
        }

        // if (answer.compare(0, 2, ":" + axis_) == 0)
        if (answer.compare(0, 5, ":A " + axis_ + "=") == 0)
        {
            double speed = 0.0;
            const int code = ParseResponseAfterPosition(answer, 6, 13, speed);
            pProp->Set(speed);
            return code;
        }
        // deal with error later
        else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
        {
            int errNo = atoi(answer.substr(3).c_str());
            return ERR_OFFSET + errNo;
        }
        return ERR_UNRECOGNIZED_ANSWER;
    }
    else if (eAct == MM::AfterSet)
    {
        double vector;
        pProp->Get(vector);

        std::ostringstream command;
        command << "VE " << axis_ << "=" << vector;
        std::string answer;
        // query command
        int ret = QueryCommand(command.str().c_str(), answer);
        if (ret != DEVICE_OK)
        {
            return ret; 
        }
        return ResponseStartsWithColonA(answer);
    }
    return DEVICE_OK;
}
