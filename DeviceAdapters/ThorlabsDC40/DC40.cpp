#include "DC40.h"

const char* g_serialNumber = "Serial Number";
const char* g_operatingMode = "Operating Mode";
const char* g_operatingModeCW = "CW";
const char* g_operatingModeTTL = "TTL";
const char* g_operatingModeMOD = "MOD";


DC40::DC40(const char* deviceName) :
    initialized_(false),
    current_(0.0),
    currentLimit_(4000.0),
    state_(false),
    operatingMode_("CW"),
    instrumentHandle_(0),
    serialNr_("NA"),
    deviceName_(deviceName)
{
    // Create properties
    CreateProperty(MM::g_Keyword_Name, deviceName_.c_str(), MM::String, true); 

    CPropertyAction* pAct = new CPropertyAction(this, &DC40::OnSerialNumber);
    int nRet = CreateProperty(g_serialNumber, serialNr_.c_str(), MM::String, false, pAct, true);
    if (nRet != DEVICE_OK)
       return;

    ViUInt32 numDevices;
    int error = TLDC_findRsrc(VI_NULL, &numDevices);
    if (error) return;

    for (ViUInt32 i = 0; i < numDevices; i++) {
        ViChar resourceName[TLDC_BUFFER_SIZE];
        ViChar serNr[TLDC_BUFFER_SIZE];
        ViPBoolean available = false;
        error = TLDC_getRsrcInfo(0, i, resourceName, serNr, VI_NULL, available);
        if (error) return;
        AddAllowedValue(g_serialNumber, serNr);
    }
}

DC40::~DC40()
{
    Shutdown();
}


int DC40::Initialize()
{
    if (initialized_)
        return DEVICE_OK;
    

    // Find the correct device based on serial number
    ViUInt32 numDevices;
    ViChar rsrcDescr[TLDC_BUFFER_SIZE];
    for (uint32_t i = 0; i < TLDC_BUFFER_SIZE; i++)
      rsrcDescr[i] = '\0';
    bool found = false;
    int error = TLDC_findRsrc(VI_NULL, &numDevices);
    if (error) return DEVICE_NOT_CONNECTED;

    for (ViUInt32 i = 0; i < numDevices; i++) {
        ViChar resourceName[TLDC_BUFFER_SIZE];
        ViChar serNr[TLDC_BUFFER_SIZE];
        ViPBoolean available = false;
        error = TLDC_getRsrcInfo(0, i, resourceName, serNr, VI_NULL, available);
        if (error) return DEVICE_NOT_CONNECTED;  // TODO: LOG!!!
        if (serialNr_ == serNr) {
            error = TLDC_getRsrcName(0, i, rsrcDescr);
            if (error) return DEVICE_NOT_CONNECTED;
            found = true;
            break;
        }
    }

    if (!found)
        return DEVICE_NOT_CONNECTED;


    error = TLDC_init(rsrcDescr, VI_OFF, VI_OFF, &instrumentHandle_);
    if (error) return HandleError(error);

    
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
    SetPropertyLimits("Current", 0.0, 1.225); // TODO: get this from the LED

    // TODO: LED current limit
    // pAct = new CPropertyAction(this, &DC40::OnCurrentLimit);
    // CreateProperty("CurrentLimit", "1.225", MM::Float, false, pAct);
    // SetPropertyLimits("CurrentLimit", 0.5, 4.0);

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
    CDeviceUtils::CopyLimitedString(name, deviceName_.c_str());
}

bool DC40::Busy()
{
    return false;
}

int DC40::OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(serialNr_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(serialNr_);
    }
    return DEVICE_OK;
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

        if (mode != operatingMode_)
        {
            bool open = false;
            GetOpen(open);
            if (open)
               SetOpen(false);

            ViUInt32 modeNr = TLDC_LED_MODE_CW;
            if (mode == g_operatingModeMOD)
               modeNr = TLDC_LED_MODE_MOD;
            else if (mode == g_operatingModeTTL)
               modeNr = TLDC_LED_MODE_TTL;

            int error = TLDC_setLedMode(instrumentHandle_, modeNr);
            if (error)
               return HandleError(error);

            if (open)
               SetOpen(true);
            operatingMode_ = mode;
        }
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

int DC40::SetOpen(bool state)
{
    const char* parm = state ? "1" : "0";
    return SetProperty(MM::g_Keyword_State, parm);
}

int DC40::GetOpen(bool& state) {
    long result;
    int ret = GetProperty(MM::g_Keyword_State, result);
    state = result == 1;
    return ret;
}

int DC40::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
       ViBoolean ledOutputState;
       int error = TLDC_getLedOutputState(instrumentHandle_, &ledOutputState);
       if (error)
          return DEVICE_ERR;

       state_ = ledOutputState;

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