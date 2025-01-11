#include "DC40.h"

const char* g_operatingMode = "Operating Mode";
const char* g_operatingModeCW = "CW";
const char* g_operatingModeTTL = "TTL";
const char* g_operatingModeMOD = "MOD";


DC40::DC40(const char* serialNr) :
    initialized_(false),
    current_(0.0),
    currentLimit_(4000.0),
    state_(false),
    operatingMode_("CW"),
    instrumentHandle_(0),
    serialNr_(serialNr),
    name_("DC40LED")
{
    // Create properties
    CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true); 
}

DC40::~DC40()
{
    Shutdown();
}


int DC40::Initialize()
{
    if (initialized_)
        return DEVICE_OK;
    /*
    // Find and connect to the first available device
    ViUInt32 numDevices;
    int error = TLDC_findRsrc(VI_NULL, &numDevices);
    if (error) return HandleError(error);

    if (numDevices < 1)
        return  DEVICE_NOT_CONNECTED;

    ViChar resourceName[TLDC_BUFFER_SIZE];
    error = TLDC_getRsrcName(VI_NULL, 0, resourceName);
    if (error) return HandleError(error);
    */

    // Initialize the device
    char* name = new char[deviceName_.length() + 1];
    strcpy(name, deviceName_.c_str());
    int error = TLDC_init(name, VI_OFF, VI_OFF, &instrumentHandle_);
    if (error) return HandleError(error);

    error = TLDC_close(instrumentHandle_);
    if (error) return HandleError(error);

    // Initialize the device
    error = TLDC_init(name, VI_OFF, VI_OFF, &instrumentHandle_);
    if (error) return HandleError(error);
    delete[] name;

    ViStatus err = VI_SUCCESS;
    ViChar ledName[30] = { "nothing read" };
    ViChar ledSerialNumber[30] = { "nothing read" };
    ViReal64 ledCurrentLimit = -1.0;
    ViReal64 ledForwardVoltage = -1.0;
    ViReal64 ledWavelength = 0.0;

    printf("Get LED Info ...\n");
    err = TLDC_getLedInfo(instrumentHandle_, ledName, ledSerialNumber, &ledCurrentLimit, &ledForwardVoltage, &ledWavelength);
    if (!err) printf("LED name: %s\n", ledName);

    // Configure initial device settings
    error = TLDC_setLedMode(instrumentHandle_, TLDC_LED_MODE_CW);
    if (error) return HandleError(error);

    /*
    error = TLDC_setLedCurrentSetpoint(instrumentHandle_, 0.0);
    if (error) return HandleError(error);

    error = TLDC_setLedCurrentLimitUser(instrumentHandle_, currentLimit_);
    if (error) return HandleError(error);
    */


    // TODO: Create MM properrty OnOperatingMode populate with three modes
    // Initialize the PWM current action
    CPropertyAction* pAct = new CPropertyAction(this, &DC40::OnOperatingMode);
    int nRet = CreateProperty(g_operatingMode, operatingMode_.c_str(), MM::String, false, pAct);
    AddAllowedValue(g_operatingMode, g_operatingModeCW);
    AddAllowedValue(g_operatingMode, g_operatingModeTTL);
    AddAllowedValue(g_operatingMode, g_operatingModeMOD);
    if (DEVICE_OK != nRet)	
        return nRet;

    // LED current setpoint
    pAct = new CPropertyAction(this, &DC40::OnCurrent);
    CreateProperty("Current", "0.0", MM::Float, false, pAct);
    SetPropertyLimits("Current", 0.0, 4000.0);

    // LED current limit
    pAct = new CPropertyAction(this, &DC40::OnCurrentLimit);
    CreateProperty("CurrentLimit", "4000.0", MM::Float, false, pAct);
    SetPropertyLimits("CurrentLimit", 100.0, 4000.0);

    // LED state (On/Off)
    pAct = new CPropertyAction(this, &DC40::OnState);
    CreateProperty("State", "0", MM::Integer, false, pAct);
    AddAllowedValue("State", "0"); // Off
    AddAllowedValue("State", "1"); // On

    initialized_ = true;
    return DEVICE_OK;
}

int DC40::Shutdown()
{
    if (initialized_ && instrumentHandle_ != 0)
    {
        // Turn off LED
        TLDC_switchLedOutput(instrumentHandle_, VI_FALSE);

        // Close device
        TLDC_close(instrumentHandle_);
        instrumentHandle_ = 0;
        initialized_ = false;
    }
    return DEVICE_OK;
}

void DC40::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

bool DC40::Busy()
{
    return false;
}

int DC40::OnOperatingMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(operatingMode_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string mode;
        pProp->Get(mode);

        ViUInt32 modeNr = TLDC_LED_MODE_CW;
        if (mode == g_operatingModeMOD)
            modeNr = TLDC_LED_MODE_MOD;
        else if (mode == g_operatingModeTTL)
            modeNr = TLDC_LED_MODE_TTL;

        int error = TLDC_setLedMode(instrumentHandle_, modeNr);
        if (error) 
            return HandleError(error);

        operatingMode_ = mode;
    }
    return DEVICE_OK;
}

int DC40::OnCurrent(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(current_);
    }
    else if (eAct == MM::AfterSet)
    {
        double current;
        pProp->Get(current);

        int error = TLDC_setLedCurrentSetpoint(instrumentHandle_, current);
        if (error) return HandleError(error);

        current_ = current;
    }
    return DEVICE_OK;
}

int DC40::OnCurrentLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(currentLimit_);
    }
    else if (eAct == MM::AfterSet)
    {
        double limit;
        pProp->Get(limit);

        int error = TLDC_setLedCurrentLimitUser(instrumentHandle_, limit);
        if (error) return HandleError(error);

        currentLimit_ = limit;
    }
    return DEVICE_OK;
}

int DC40::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(state_ ? 1L : 0L);
    }
    else if (eAct == MM::AfterSet)
    {
        long state;
        pProp->Get(state);

        int error = TLDC_switchLedOutput(instrumentHandle_, state == 1 ? VI_TRUE : VI_FALSE);
        if (error) return HandleError(error);

        state_ = (state == 1);
    }
    return DEVICE_OK;
}

int DC40::HandleError(int error)
{
    if (error == 0) return DEVICE_OK;

    ViChar errorMessage[256];
    TLDC_errorMessage(instrumentHandle_, error, errorMessage);
    LogMessage(errorMessage);
    return DEVICE_ERR;
}