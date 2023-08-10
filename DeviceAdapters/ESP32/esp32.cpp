/* 
 =====================================================================
 
 This Adapter is modified from Arduino32.cpp & PIEZOCONCEPT.cpp:

 Original authors and copyright as below: 
 COPYRIGHT:     University of California, San Francisco, 2008
 LICENSE:       LGPL
 AUTHOR:        Original Author Arduino Device Adapter:
				  Nico Stuurman, nico@cmp.ucsf.edu, 11/09/2008
 
                Automatic device detection by Karl Hoover
 
                Author 32-Bit-Boards adaptation: 
                Bonno Meddens, 30/07/2019
 
 =====================================================================
 
 This uManager Device Adapter is called "ESP32.cpp"
 It requires associated firmware to be uploaded to an ESP family microcontroller.
 The microcontroller is programmed using Arduino software.
 "ESP32_COM_BT_WiFi.INO"

 AUTHORS:      Justin E. Molloy & Nicholas J. Carter
               University of Warwick, UK 2023

 The firmware code is heavily commented so should be pretty easy to modify. 
 It has a combined interface to Bluetooth, WiFI & USB/RS232. It also supports an 
 attached OLED screen which helps with debugging.
 
 Efforts have been made to comment any new code. However, much of the original code
 remains uncommented except for some guesses which may be incorrect - I apologise.
 
 Main changes:
 1) All commands are now sent as ASCII so we can debug/test using PuTTY and monitor command 
    transactions on the OLED screen. The "byte count" overhead is trivial compared to USB bus 
    latency times and total packet size for single COM <-> transactions - most transactions are
    not time-critical
 2) Most commands are now simplex (no duplex handshaking)
 3) x,y,z stage control has been added
 
 Hint: to make the code easier to navigate in Visual Studio press "ctrl M" then "ctrl O"
 to collapse the code.
 
 Things to improve:
 1) Need to fully comment the code so it is easy to modify and debug.
 2) Port communications are messy and should be conducted using a single function.
 3) String handling is inconsistent.
    (Note: The "ASI-hub" has some very nice utility functions - need to port to here)
 5) In this version the "shutter" is a switch and the switch is a shutter which is confusing
 6) The input monitor thread (timed "L" calls) don't seem to work quite as I would expect
 7) Need to read back all settings especially: temperature, x,y,z stage posn, optical trap, 
    magnetic tweezer position and various logic state switches so they can be saved with the
    movie data as Tiff Tags
 8) MDA LED switching cycle is not working exactly as it should 
    The first channel in the series turns on at end of cycle (during the dark phase)

*/

#include "ESP32.h"
#include "ModuleInterface.h"
#include <math.h>

//string and IO handling
#include <string>
#include <sstream>
#include <cstdio>
using namespace std;

// Name the Devices something short and sensible
const char* g_DeviceNameESP32Hub =     "ESP32-Hub";
const char* g_DeviceNameESP32Switch =  "ESP32-Switch";
const char* g_DeviceNameESP32Shutter = "ESP32-Shutter";
const char* g_DeviceNameESP32PWM0 =    "ESP32-PWM0";
const char* g_DeviceNameESP32PWM1 =    "ESP32-PWM1";
const char* g_DeviceNameESP32PWM2 =    "ESP32-PWM2";
const char* g_DeviceNameESP32PWM3 =    "ESP32-PWM3";
const char* g_DeviceNameESP32PWM4 =    "ESP32-PWM4";
const char* g_DeviceNameStage =        "ZStage";
const char* g_DeviceNameXYStage =      "XYStage";
const char* g_DeviceNameESP32Input =   "ESP32-Input";

const char* g_PropertyMinUm =          "Z Stage Low Posn(um)";
const char* g_PropertyMaxUm =          "Z Stage High Posn(um)";

const char* g_PropertyXMinUm =         "X Stage Min Posn(um)";
const char* g_PropertyXMaxUm =         "X Stage Max Posn(um)";

const char* g_PropertyYMinUm =         "Y Stage Min Posn(um)";
const char* g_PropertyYMaxUm =         "Y Stage Max Posn(um)";

// Global info about the state of the ESP32.  This should be folded into a class
const int g_Min_MMVersion = 1;
const int g_Max_MMVersion = 100;
const char* g_versionProp =         "Version";
const char* g_normalLogicString =   "Normal";
const char* g_invertedLogicString = "Inverted";

const char* g_On =  "On";
const char* g_Off = "Off";

// static lock
MMThreadLock CESP32Hub::lock_;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
    // The HUB and other devices are registered with the MMcore code
    // I think there are currently ~16 recognised Device "types" including:
    // Hub, StateDevice, ShutterDevice, GenericDevice, SignalIODevice, StageDevice(focus?), 
    // XYStageDevice, AutofocusDevice, GalvoDevice.. There seems to be some ambiguity/overlap
    // in what the different Device types actually do.. and the shutter function is used to 
    // change the LED output pattern.

    RegisterDevice(g_DeviceNameESP32Hub, MM::HubDevice,         "Hub (required)");

    // Note: The "StateDevice" is used to "blank" the SignalIODevice by setting the output to zero.
    RegisterDevice(g_DeviceNameESP32Switch, MM::StateDevice,    "Switch on/off channels 0 to 10");
    RegisterDevice(g_DeviceNameESP32Shutter, MM::ShutterDevice, "Shutter");
    RegisterDevice(g_DeviceNameESP32PWM0, MM::SignalIODevice,   "PWM channel 0");
    RegisterDevice(g_DeviceNameESP32PWM1, MM::SignalIODevice,   "PWM channel 1");
    RegisterDevice(g_DeviceNameESP32PWM2, MM::SignalIODevice,   "PWM channel 2");
    RegisterDevice(g_DeviceNameESP32PWM3, MM::SignalIODevice,   "PWM channel 3");
    RegisterDevice(g_DeviceNameESP32PWM4, MM::SignalIODevice,   "PWM channel 4");
    RegisterDevice(g_DeviceNameStage, MM::StageDevice,          "Z Stage");
    RegisterDevice(g_DeviceNameXYStage, MM::XYStageDevice,      "XY Stage");
    RegisterDevice(g_DeviceNameESP32Input, MM::GenericDevice,   "ADC");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    // Justin's Notes: After registering the devices we now "create them" and give them another name.
    // I admit I find it hard to keep track of the different names - especially when
    // things get renamed again within the config file. Just keep your hair on and go
    // with it!

    if ( deviceName == 0 ) return 0;

    if ( strcmp(deviceName, g_DeviceNameESP32Hub) == 0 )          return new CESP32Hub;
    else if ( strcmp(deviceName, g_DeviceNameESP32Switch) == 0 )  return new CESP32Switch;
    else if ( strcmp(deviceName, g_DeviceNameESP32Shutter) == 0 ) return new CESP32Shutter;
    else if ( strcmp(deviceName, g_DeviceNameESP32PWM0) == 0 )    return new CESP32DA(0); // channel 0
    else if ( strcmp(deviceName, g_DeviceNameESP32PWM1) == 0 )    return new CESP32DA(1); // channel 1
    else if ( strcmp(deviceName, g_DeviceNameESP32PWM2) == 0 )    return new CESP32DA(2); // channel 2
    else if ( strcmp(deviceName, g_DeviceNameESP32PWM3) == 0 )    return new CESP32DA(3); // channel 3
    else if ( strcmp(deviceName, g_DeviceNameESP32PWM4) == 0 )    return new CESP32DA(4); // channel 4
    else if ( strcmp(deviceName, g_DeviceNameStage) == 0 )        return new CESP32Stage();
    else if ( strcmp(deviceName, g_DeviceNameXYStage) == 0 )      return new CESP32XYStage();
    else if ( strcmp(deviceName, g_DeviceNameESP32Input) == 0 )   return new CESP32Input;
    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}


/* CESP32HUb implementation:
 The HUB is the master "Device" it gives a convenient way to communicate with several
 physical or "nominal" devices connected via a single cable to the PC. The hub heirarchy makes things 
 a little more complicated than adapters written for single, stand-alone, physical devices.

 Although the ESP32 microcontroller is just a single physical device, here
 it functions as four or five separate "MM_Devices". The firmware running on the ESP can 
 control LED light sources, move an XY stage, a focusing device, open/close a physical shutter, 
 control a laser system etc.

 The ESP firmware allows Bluetooth, Wifi, USB(RS232) control "Commanders"
 We should report back to micromanager what different commanders are doing. 
 (I don't do that at present!). 

 LED channels:
 We have 10-bit res on 4 GPIOs (50kHz PWM) for LED control - easy to interface FETs 
 with a PWM signal which then allows control of high-current LEDs.
 
 Laser Control:
 Suggest ESP32 firmware does all the device-specific RS232 stuff by "echoing" appropriate
 commands down its secondary RxTx (pins) with splitter (e.g. MAX 3221 + Max399 to 
 create a 3-port RS2323 hub)? Then we can control all our devices without writing individual
 device adapaters for MM.
 E.G.  Our Toptica laser requires only three commands via RS232 to do what we need:
 "la on", "la off" and set power ("ch 1 pow "+(floating point mW value)) we can do that
 over the secondary (hardware) RxTx on ESP32.
 We assign it to an available DAC line (here channel "5")
 E.g. MM sends:     "O,5,0" to turn the laser OFF
                    "O,5,27.5" to turn laser on to 27.5mW
###
ARDUINO PSEUDO-CODE:
#include <HardwareSerial.h>
void setup() {
    Serial.begin(115200);  // connected to computer
    Serial2.begin(115200); // connected to laser
}

void loop() {
    if (Serial.available()) {
        ...read and process commands sent by uManager via primary USB(RS232)
        .. if it's "O"utput to "channel 5" (the laser)  e.g.  "O,5,27.5" 
        ...then echo the device-specific command via secondary RS232 [GPIO16] & [GPIO17]
        Serial2.println("ch 1 pow 27.5");
        Serial2.println("la on");
        Serial.println("O"); // echo back to micromanager that command has been received and done.
    }
}
###
Using the above approach the laser would just appear as another DAC channel: adjust power, turn on/off
sequence it do what you like.
*/

/*
 HUB Summary:
 ============
 The Hub code sends 2 commands or "requests" to the ESP device.
 => "V" "V"ersion number & Firmware ID? - expected response is "MM-ESP32,5" or greater 
 => "U,0-2" "U"nderstand the XYZ stage ranges (X=0, Y=1, Z=2) 
      => "U,0" expects ESP32 response e.g. "U,200" (X-axis range)
      => "U,2" expects ESP32 response e.g. "U,100" (Z-axis range)
      => if no response or range=0..  MM thinks there's no stage
 
 => There are 3 or 4 functions/methods that do clever things to DetectDevices
    Probably critical for the Hardware Wizard to work.
    => Nice to have these functions documented here.. I have no idea of the mechanics
*/

//CESP32HUb implementation
CESP32Hub::CESP32Hub() :
    initialized_(false),
    switchState_(0),
    shutterState_(0),
    hasZStage_(false),
    hasXYStage_(false)
{
    portAvailable_ = false;
    invertedLogic_ = false;
    timedOutputActive_ = false;


// Create Error messages
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_PORT_OPEN_FAILED, "Failed opening ESP32 USB device");
    SetErrorText(ERR_BOARD_NOT_FOUND, "Did not find an ESP32 board with the correct firmware.  Is the ESP32 board connected to this serial port?");
    SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The ESP32 Hub device is needed to create this device");
    std::ostringstream errorText;
    errorText << "The firmware version on the ESP32 is not compatible with this adapter.  Please use firmware version ";
    errorText << g_Min_MMVersion << " to " << g_Max_MMVersion;
    SetErrorText(ERR_VERSION_MISMATCH, errorText.str().c_str());

// Create some pre-initialisation properties:
// Port:
    CPropertyAction* pAct = new CPropertyAction(this, &CESP32Hub::OnPort);
    CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

// Set default logic property to "inverted" (1 is 0) ?
// comments==0 
    pAct = new CPropertyAction(this, &CESP32Hub::OnLogic);
    CreateProperty("Logic", g_invertedLogicString, MM::String, false, pAct, true);

    AddAllowedValue("Logic", g_invertedLogicString);
    AddAllowedValue("Logic", g_normalLogicString);
}

CESP32Hub::~CESP32Hub()
{
    Shutdown();
}

void CESP32Hub::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameESP32Hub);
}

bool CESP32Hub::Busy()
{
    return false;
}

// Commands: 30 & 31 - We now just send "V" (to confirm ESP firmware name and version)
int CESP32Hub::GetControllerVersion(int& version)
{
    // Here we use a char array to "send" and a string for "receive"
    // Makes sense because we don't know how long the receive message will be
    int ret = DEVICE_OK;
    char command[50];
    string answer;

    LogMessage("Check ESP32 firmware name and version (>=1?)", false);

    // Flush the I/O buffer
    PurgeComPort(port_.c_str());

    // construct the command char array (limit to 50 characters).. here it's just one character
    snprintf(( char* ) command, 50, "V"); 
    
    ret = SendSerialCommand(port_.c_str(), command, "\r\n");
    if ( ret != DEVICE_OK ) return ret;

    // get the answer.. a String .. hence need to INCLUDE <string> and <sstream>
    ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
    // Flush the I/O buffer
    PurgeComPort(port_.c_str());

    if ( ret != DEVICE_OK ) return ret;

    if (answer.find("MM-ESP32") > 0 )  return ERR_BOARD_NOT_FOUND;

    // convert the "answer" string to a char array so we can parse it with sscanf
    const char* anschar = answer.c_str();
    sscanf(( const char* ) anschar, "%8s,%d", command, &version);

    // version number should be => 1
    std::ostringstream os;
    os << "Send/Receive:"<< port_.c_str() << " -> FirmwareQuery: " << answer;
    LogMessage(os.str().c_str(), false);

    LogMessage("InitialisingStageRanges X,Y,Z", false);

    // now the XYZ stage setup

    double travelX, travelY, travelZ;

    ret = GetAxisInfo(0, travelX);
    ret = GetAxisInfo(1, travelY);
    if ( ( travelX > 0 ) && ( travelY > 0 ) )  hasXYStage_ = true;

    ret = GetAxisInfo(2, travelZ);
    if ( travelZ > 0 ) hasZStage_ = true;

    return ret;
}

// Command: "U" obtain stageRange for X, Y and Z axes
int CESP32Hub::GetAxisInfo(int axis, double& travel)
{

    if ( axis < 0 || axis > 2 ) return DEVICE_ERR;

    std::stringstream cmd;
    cmd.clear();

    switch ( axis ) {
    case 0:
        cmd << "U,0"; // X-axis
        break;
    case 1:
        cmd << "U,1"; // Y-axis
        break;
    case 2:
        cmd << "U,2"; // Z-axis
        break;
    }

    // Flush the I/O buffer
    PurgeComPort(port_.c_str());

    int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r\n");
    if ( ret != DEVICE_OK ) return ret;

    std::string answer;
    ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
    if ( ret != DEVICE_OK ) return ret;

    // another way to play with string.. 
    std::stringstream ss(answer);
    std::string type, trav;
    getline(ss, type, ',');
    getline(ss, trav, ',');

    if ( 0 != strcmp(type.c_str(), "U") ) return ERR_UNKNOWN_AXIS;

    std::stringstream sstravel(trav);
    sstravel >> travel;

    return DEVICE_OK;
}

bool CESP32Hub::SupportsDeviceDetection(void)
{
    return true;
}

MM::DeviceDetectionStatus CESP32Hub::DetectDevice(void)
{
    if ( initialized_ )  return MM::CanCommunicate;

    // all conditions must be satisfied...
    MM::DeviceDetectionStatus result = MM::Misconfigured;
    char answerTO[MM::MaxStrLength];

    try
    {
        std::string portLowerCase = port_;
        for ( std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its )
        {
            *its = ( char ) tolower(*its);
        }
        if ( 0 < portLowerCase.length() && 0 != portLowerCase.compare("undefined") && 0 != portLowerCase.compare("unknown") )
        {
            result = MM::CanNotCommunicate;
            // record the default answer time out
            GetCoreCallback()->GetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);

            // device specific default communication parameters
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_Handshaking, g_Off);
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "115200");
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_StopBits, "1");
            // ESP32 needs quite a long AnswerTimeout because it checks fo WiFi etc...
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", "5000.0");
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), "DelayBetweenCharsMs", "0");
            MM::Device* pS = GetCoreCallback()->GetDevice(this, port_.c_str());

            pS->Initialize();

            // The first second or so after opening the serial port, the ESP32 is waiting for firmwareupgrades.  Simply sleep 2 seconds.
            CDeviceUtils::SleepMs(5000);
            MMThreadGuard myLock(lock_);
            PurgeComPort(port_.c_str());
            int v = 0;
            int ret = GetControllerVersion(v);

            if ( DEVICE_OK != ret )  LogMessageCode(ret, false);
            else result = MM::CanCommunicate;

            pS->Shutdown();

            // always restore the AnswerTimeout to the default
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);

        }
    }
    catch ( ... )
    {
        LogMessage("Exception in DetectDevice!", false);
    }

    return result;
}

int CESP32Hub::Initialize()
{
    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameESP32Hub, MM::String, true);
    if ( DEVICE_OK != ret )
        return ret;

    // Sleep for 2 seconds while ESP32 restarts. 
    CDeviceUtils::SleepMs(2000);

    MMThreadGuard myLock(lock_);

    // Check that we have a controller:
    PurgeComPort(port_.c_str());
    ret = GetControllerVersion(version_);
    if ( DEVICE_OK != ret )
        return ret;

    if ( version_ < g_Min_MMVersion )
        return ERR_VERSION_MISMATCH;

    CPropertyAction* pAct = new CPropertyAction(this, &CESP32Hub::OnVersion);
    std::ostringstream sversion;
    sversion << version_;
    CreateProperty(g_versionProp, sversion.str().c_str(), MM::Integer, true, pAct);

    ret = UpdateStatus();
    if ( ret != DEVICE_OK ) return ret;

    // turn off verbose serial debug messages
    // GetCoreCallback()->SetDeviceProperty(port_.c_str(), "Verbose", "0");

    initialized_ = true;
    return DEVICE_OK;
}

int CESP32Hub::DetectInstalledDevices()
{
    if ( MM::CanCommunicate == DetectDevice() )
    {
        std::vector<std::string> peripherals;
        peripherals.clear();
        peripherals.push_back(g_DeviceNameESP32Switch);
        peripherals.push_back(g_DeviceNameESP32Shutter);
        peripherals.push_back(g_DeviceNameESP32PWM0);
        peripherals.push_back(g_DeviceNameESP32PWM1);
        peripherals.push_back(g_DeviceNameESP32PWM2);
        peripherals.push_back(g_DeviceNameESP32PWM3);
        peripherals.push_back(g_DeviceNameESP32PWM4);
        peripherals.push_back(g_DeviceNameStage);
        peripherals.push_back(g_DeviceNameXYStage);
        peripherals.push_back(g_DeviceNameESP32Input);

        for ( size_t i = 0; i < peripherals.size(); i++ )
        {
            MM::Device* pDev = ::CreateDevice(peripherals[i].c_str());
            if ( pDev ) AddInstalledDevice(pDev);
        }
    }

    return DEVICE_OK;
}

int CESP32Hub::Shutdown()
{
    initialized_ = false;
    return DEVICE_OK;
}

int CESP32Hub::OnPort(MM::PropertyBase* pProp, MM::ActionType pAct)
{
    if ( pAct == MM::BeforeGet )
    {
        pProp->Set(port_.c_str());
    }
    else if ( pAct == MM::AfterSet )
    {
        pProp->Get(port_);
        portAvailable_ = true;
    }
    return DEVICE_OK;
}

int CESP32Hub::OnVersion(MM::PropertyBase* pProp, MM::ActionType pAct)
{
    if ( pAct == MM::BeforeGet )
    {
        pProp->Set(( long ) version_);
    }
    return DEVICE_OK;
}

int CESP32Hub::OnLogic(MM::PropertyBase* pProp, MM::ActionType pAct)
{
    if ( pAct == MM::BeforeGet )
    {
        if ( invertedLogic_ )
            pProp->Set(g_invertedLogicString);
        else
            pProp->Set(g_normalLogicString);
    }
    else if ( pAct == MM::AfterSet )
    {
        std::string logic;
        pProp->Get(logic);
        if ( logic.compare(g_invertedLogicString) == 0 )
            invertedLogic_ = true;
        else invertedLogic_ = false;
    }
    return DEVICE_OK;
}

/* CESP32Switch implementation:
 Justin's Notes:
 The "Switch Device" is registered with MM as a "State Device"
 I don't know the full command set used by the MM config file which is
 documented by "example" - I haven't found a detailed listing for that.

 Different switch states are attached to different "channels" - 
 That is done here, again in the config file, also in the ESP32 firmware
 and finally the physical wiring of the GPIO pins. Very flexible but 
 the complexity can lead to confusion.
*/

// CESP32Switch implementation
CESP32Switch::CESP32Switch() :
    nrPatternsUsed_(0),
    currentDelay_(0),
    sequenceOn_(false),
    blanking_(false),
    initialized_(false),
    numPos_(12),
    busy_(false)
{


    InitializeDefaultErrorMessages();

    // add custom error messages
    SetErrorText(ERR_UNKNOWN_POSITION,  "Invalid position (state) specified");
    SetErrorText(ERR_INITIALIZE_FAILED, "Initialization of the device failed");
    SetErrorText(ERR_WRITE_FAILED,      "Failed to write data to the device");
    SetErrorText(ERR_CLOSE_FAILED,      "Failed closing the device");
    SetErrorText(ERR_COMMUNICATION,     "Error in communication with ESP32 board");
    SetErrorText(ERR_NO_PORT_SET,       "Hub Device not found.  The ESP32 Hub device is needed to create this device");

    for ( unsigned int i = 0; i < NUMPATTERNS; i++ )
        pattern_[i] = 0;

    // Description
    int ret = CreateProperty(MM::g_Keyword_Description, "ESP32 digital output driver", MM::String, true);
    assert(DEVICE_OK == ret);

    // Name
    ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameESP32Switch, MM::String, true);
    assert(DEVICE_OK == ret);

    // parent ID display
    CreateHubIDProperty();
}

CESP32Switch::~CESP32Switch()
{
    Shutdown();
}

void CESP32Switch::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameESP32Switch);
}

int CESP32Switch::Initialize()
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() ) {
        return ERR_NO_PORT_SET;
    }
    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    // set property list
    // -----------------

    // In our ESP32 firmware we have "nDAC" switchable devices 
    // Currently, 4 LEDS.
    // so 2^4 = 16 bit-mapped options 
    // We will define up to 256 char labels for 256 options !
    // 0="0", 2="2"...... 255="255"
    // The ASCII labels are the decimal numbers representing each bit-mapped switch pattern
    // (see below for explanation)
    
    //Create the text labels that we will use in the config file 
    const int bufSize = 4;
    char buf[bufSize];
    for ( long i = 0; i < 256; i++ ) {
        snprintf(( char* ) buf, bufSize, "%d", ( unsigned ) i);
        SetPositionLabel(i, buf);
    }

    /* E.g. if you have different LED light sources powered by the different ESP32 O/P pins
      B,  G,  Y,  R   < colour of the LED
      3,  2,  1,  0   < bit position
      8,  4,  2,  1   < decimal number when set to "1" at that bit position
      1,  0,  0,  0,   = "Blue only" requires this bit pattern = 16 Decimal 
    And if you want RGB simultaneously you send:
      1,  1,  0,  1    = 8+4+1 = 13 Decimal
    To make this easier you list the options you want available in the Config file like this:

*** example config file: The comma-delimited fields need to be documented.. add HTTP: here..
Label,ESP32-Switch,0,All_OFF
Label,ESP32-Switch,1,Red
Label,ESP32-Switch,2,Yellow
Label,ESP32-Switch,4,Green
Label,ESP32-Switch,8,Blue
Label,ESP32-Switch,13,RedGreenBlue

ConfigGroup,Channel,Red_LED,ESP32-Switch,Label,Red
ConfigGroup,Channel,Yellow_LED,ESP32-Switch,Label,Yellow
ConfigGroup,Channel,Green_LED,ESP32-Switch,Label,Green
ConfigGroup,Channel,Blue_LED,ESP32-Switch,Label,Blue
ConfigGroup,Channel,Red_Green_Blue_LEDs,ESP32-Switch,Label,RedGreenBlue
*** 

    */
    // State
    CPropertyAction* pAct = new CPropertyAction(this, &CESP32Switch::OnState);
    int nRet = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if ( nRet != DEVICE_OK ) return nRet;
    SetPropertyLimits(MM::g_Keyword_State, 0, 255);

    // Label
    pAct = new CPropertyAction(this, &CStateBase::OnLabel);
    nRet = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if ( nRet != DEVICE_OK )
        return nRet;

    pAct = new CPropertyAction(this, &CESP32Switch::OnSequence);
    nRet = CreateProperty("Sequence", g_On, MM::String, false, pAct);
    if ( nRet != DEVICE_OK )
        return nRet;
    AddAllowedValue("Sequence", g_On);
    AddAllowedValue("Sequence", g_Off);

    // Starts "blanking" mode: goal is to synchronize laser light with camera exposure
    std::string blankMode = "Blanking Mode";
    pAct = new CPropertyAction(this, &CESP32Switch::OnBlanking);
    nRet = CreateProperty(blankMode.c_str(), "Idle", MM::String, false, pAct);
    if ( nRet != DEVICE_OK ) return nRet;

    AddAllowedValue(blankMode.c_str(), g_On);
    AddAllowedValue(blankMode.c_str(), g_Off);

    // Blank on TTL high or low
    pAct = new CPropertyAction(this, &CESP32Switch::OnBlankingTriggerDirection);
    nRet = CreateProperty("Blank On", "Low", MM::String, false, pAct);
    if ( nRet != DEVICE_OK )
        return nRet;
    AddAllowedValue("Blank On", "Low");
    AddAllowedValue("Blank On", "High");

    /*
    // Some original comments:
    // but SADLY, the code itself has been commented out
    // In fact, looks like a useful thing to include...To Do.
    // //////////////////////////////////////////////////////
    // Starts producing timed digital output patterns
    // Parameters that influence the pattern are 'Repeat Timed Pattern', 'Delay', 'State'
    // where the latter two are manipulated with the Get and SetPattern functions

    std::string timedOutput = "Timed Output Mode";
    pAct = new CPropertyAction(this, &CESP32Switch::OnStartTimedOutput);
    nRet = CreateProperty(timedOutput.c_str(), "Idle", MM::String, false, pAct);
    if (nRet != DEVICE_OK)
       return nRet;
    AddAllowedValue(timedOutput.c_str(), "Stop");
    AddAllowedValue(timedOutput.c_str(), "Start");
    AddAllowedValue(timedOutput.c_str(), "Running");
    AddAllowedValue(timedOutput.c_str(), "Idle");

    // Sets a delay (in ms) to be used in timed output mode
    // This delay will be transferred to the ESP32 using the Get and SetPattern commands
    pAct = new CPropertyAction(this, &CESP32Switch::OnDelay);
    nRet = CreateProperty("Delay (ms)", "0", MM::Integer, false, pAct);
    if (nRet != DEVICE_OK)
       return nRet;
    SetPropertyLimits("Delay (ms)", 0, 65535);

    // Repeat the timed Pattern this many times:
    pAct = new CPropertyAction(this, &CESP32Switch::OnRepeatTimedPattern);
    nRet = CreateProperty("Repeat Timed Pattern", "0", MM::Integer, false, pAct);
    if (nRet != DEVICE_OK)
       return nRet;
    SetPropertyLimits("Repeat Timed Pattern", 0, 255);
    */

    nRet = UpdateStatus();
    if ( nRet != DEVICE_OK ) return nRet;

    initialized_ = true;

    return DEVICE_OK;
}

int CESP32Switch::Shutdown()
{
    initialized_ = false;
    return DEVICE_OK;
}

// I am assuming when this function (method) is called the passed parameter "value"
// is the bitmap pattern that is to be written to the switches (if bit is 0 the output is turned off
// (electronically gated) if the bit is 1 then the output goes to it's preset value.
 
// command: 1 now "S" Set current digital output pattern - The "CESP32Shutter" NO LONGER uses this command!
int CESP32Switch::WriteToPort(long value)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;

    // keep the low byte only.. we might want up to 8 lines later
    // Top bit (128) is the master shutter
    value = 255 & value; 
    if ( hub->IsLogicInverted() ) value = ~value;

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    char command[50];
    unsigned int leng;

    leng = snprintf(( char* ) command, 50, "S,%d\r\n", value);
    ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
    if ( ret != DEVICE_OK )  return ret;

    // Purge the IO buffer in case ESP has sent a response!
    hub->PurgeComPortH();
    hub->SetTimedOutput(false);

    std::ostringstream os;
    os << "Switch::WriteToPort Command= " << command;
    LogMessage(os.str().c_str(), false);

    return DEVICE_OK;
}

// Commands: 5 & 6 now "P" and "N" = store new "P"atterns and "N"umber of patterns
int CESP32Switch::LoadSequence(unsigned size, unsigned char* seq)
{
 
    std::ostringstream os;
    os << "Switch::LoadSequence size= " << size <<" Seq= " << seq;
    LogMessage(os.str().c_str(), false);
 
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    // preamble for all port reads and writes
    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    char command[50];
    unsigned int leng;

    for ( unsigned i = 0; i < size; i++ )
    {
        unsigned char value = seq[i];

        value = 255 & value;
        if ( hub->IsLogicInverted() ) value = ~value;

        leng = snprintf(( char* ) command, 50, "P,%d,%d\r\n", i, value);
        ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
        if ( ret != DEVICE_OK )  return ret;
    }

    leng = snprintf(( char* ) command, 50, "N,%d\r\n", size);
    ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
    if ( ret != DEVICE_OK )  return ret;

    // Purge the IO buffer in case ESP has sent a response!
    hub->SetTimedOutput(false);
    hub->PurgeComPortH();

    return DEVICE_OK;
}

// Action handlers

// Commands: 8 & 9 now "R" and "E" Run and End Trigger mode resp.
int CESP32Switch::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    std::ostringstream os;
    os << "Switch::OnState_";
    LogMessage(os.str().c_str(), false);

    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    // Some comments here would be helpful
    if ( eAct == MM::BeforeGet )
    {
        // nothing to do, let the caller use cached property  ???
    }
    else if ( eAct == MM::AfterSet )
    {

 //  **** a comment here would be very helpful... 
 // where are we GETting "pos" value from?
        long pos;
        pProp->Get(pos);

// this is obscure - I can guess what it does.. but, I'm not going to say
        hub->SetSwitchState(pos);

        if ( hub->GetShutterState() > 0 )    // I don't like this because of confusion with "shutterDevice"
            os << "_Pos= " << pos <<" WriteToPort(pos)?";
            LogMessage(os.str().c_str(), false);
            return WriteToPort(pos);         // I understand this!
    }
    else if ( eAct == MM::IsSequenceable )
    {
        if ( sequenceOn_ )
            pProp->SetSequenceable(NUMPATTERNS);
        else
            pProp->SetSequenceable(0);
    }
    else if ( eAct == MM::AfterLoadSequence )
    {
        std::vector<std::string> sequence = pProp->GetSequence();

        if ( sequence.size() > NUMPATTERNS ) return DEVICE_SEQUENCE_TOO_LARGE;

        unsigned char* seq = new unsigned char[sequence.size()];
        for ( unsigned int i = 0; i < sequence.size(); i++ )
        {
            std::istringstream ios(sequence[i]);
            int val;
            ios >> val;
            seq[i] = ( unsigned char ) val;
        }

        int ret = LoadSequence(( unsigned ) sequence.size(), seq);
        if ( ret != DEVICE_OK )
            return ret;

        delete[ ] seq;
    }
    else if ( eAct == MM::StartSequence )
    {
        MMThreadGuard myLock(hub->GetLock());

// preamble for all port reads and writes
        hub->PurgeComPortH();
        int ret = DEVICE_OK;
        char command[50];
        unsigned int leng;

        leng = snprintf(( char* ) command, 50, "R\r\n");
        ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
        if ( ret != DEVICE_OK )  return ret;

    }
    else if ( eAct == MM::StopSequence )
    {
        MMThreadGuard myLock(hub->GetLock());

        // preamble for all port reads and writes
        hub->PurgeComPortH();
        int ret = DEVICE_OK;
        char command[50];
        unsigned int leng;

        leng = snprintf(( char* ) command, 50, "E\r\n");
        ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
        if ( ret != DEVICE_OK )  return ret;

        // This code needs to be improved!
        // We expect a reply like "E,23456\r\n"

        MM::MMTime startTime = GetCurrentMMTime();
        unsigned long bytesRead = 0;
        unsigned char answer[10];

        // Now chug through the I/O buffer, byte-by-byte and stop at cr or lf.... yuk!
        while ( ( bytesRead < 10 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
            unsigned long br;
            ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
            if ( answer[bytesRead] == '\r' )  break;
            bytesRead += br;
        }

        answer[bytesRead] = 0; // string terminator

        hub->SetTimedOutput(false);
        hub->PurgeComPortH();
        if ( ret != DEVICE_OK ) return ret;

        int num;
        char com[1];
        sscanf(( const char* ) answer, "%1s,%d", com, &num);
        // should give com[0]='E' num = 23456 (or something!)

        if ( com[0] != 'E' )  return ERR_COMMUNICATION;

        os << "Sequence_Transitions: " << num;
        LogMessage(os.str().c_str(), false);
    }

    return DEVICE_OK;
}
int CESP32Switch::OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if ( eAct == MM::BeforeGet )
    {
        if ( sequenceOn_ )
            pProp->Set(g_On);
        else
            pProp->Set(g_Off);
    }
    else if ( eAct == MM::AfterSet )
    {
        std::string state;
        pProp->Get(state);
        if ( state == g_On )
            sequenceOn_ = true;
        else
            sequenceOn_ = false;
    }

    std::ostringstream os;
    os << "Switch::OnSequence_SeqOn= " << sequenceOn_;
    LogMessage(os.str().c_str(), false);


    return DEVICE_OK;
}

// Commands: 12 & 9 now = "G" and "E" 
int CESP32Switch::OnStartTimedOutput(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    std::ostringstream os;
    os << "Switch::OnStartTimedOutput ";
    LogMessage(os.str().c_str(), false);

    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    if ( eAct == MM::BeforeGet ) {
        if ( hub->IsTimedOutputActive() ) pProp->Set("Running");
        else                              pProp->Set("Idle");
    }
    else if ( eAct == MM::AfterSet )
    {
        MMThreadGuard myLock(hub->GetLock());

        std::string prop;
        pProp->Get(prop);

        if ( prop == "Start" ) {
            // preamble for port read and write
            hub->PurgeComPortH();
            int ret = DEVICE_OK;
            char command[50];
            unsigned int leng;

            leng = snprintf(( char* ) command, 50, "G\r\n");
            ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
            if ( ret != DEVICE_OK )  return ret;

            hub->SetTimedOutput(true);         // Check this ***********

        }
        else {
            // preamble for port read and write
            hub->PurgeComPortH();
            int ret = DEVICE_OK;
            char command[50];
            unsigned int leng;
            leng = snprintf(( char* ) command, 50, "E\r\n");
            ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
            if ( ret != DEVICE_OK )  return ret;

            hub->SetTimedOutput(false);
        }
    }

    return DEVICE_OK;
}

// Commands: 20 & 21 now = "B,1" "B,0" now Blanking on(1) or Blanking off (0)
int CESP32Switch::OnBlanking(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::ostringstream os;
    os << "Switch::OnBlanking ";
    LogMessage(os.str().c_str(), false);

    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );

    if ( !hub || !hub->IsPortAvailable() ) return ERR_NO_PORT_SET;

    if ( eAct == MM::BeforeGet ) {
        if ( blanking_ ) pProp->Set(g_On);
        else pProp->Set(g_Off);
    }
    else if ( eAct == MM::AfterSet )
    {
        MMThreadGuard myLock(hub->GetLock());

        std::string prop;
        pProp->Get(prop);

        if ( prop == g_On && !blanking_ ) {
            hub->PurgeComPortH();
            int ret = DEVICE_OK;
            char command[50];
            unsigned int leng;

            leng = snprintf(( char* ) command, 50, "B,1\r\n");
            ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
            if ( ret != DEVICE_OK )  return ret;

            blanking_ = true;
            hub->SetTimedOutput(false);
            LogMessage("Switched blanking on", false);

        }
        else if ( prop == g_Off && blanking_ ) {
            hub->PurgeComPortH();
            int ret = DEVICE_OK;
            char command[50];
            unsigned int leng;

            leng = snprintf(( char* ) command, 50, "B,0\r\n");
            ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
            if ( ret != DEVICE_OK )  return ret;

            blanking_ = false;
            hub->SetTimedOutput(false);
            LogMessage("Switched blanking off", false);
        }
    }
    return DEVICE_OK;
}

// Command: 22 now = "F,0" or "F,1" Flip polarity of trigger signal
int CESP32Switch::OnBlankingTriggerDirection(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    std::ostringstream os;
    os << "Switch::OnBlankingTriggerDirection";
    LogMessage(os.str().c_str(), false);
    
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );

    if ( !hub || !hub->IsPortAvailable() ) return ERR_NO_PORT_SET;

    // Perhaps shorten to one line and clarify logic?
    // Only execute if eAct has already been got and set.
    // if ((eAct != MM::BeforeGet) && (eAct == MM::AfterSet)) {

    if ( eAct == MM::BeforeGet ) {
        // nothing to do, let the caller use cached property
    }
    else if ( eAct == MM::AfterSet )
    {

        MMThreadGuard myLock(hub->GetLock());

        std::string direction;
        pProp->Get(direction);
        hub->PurgeComPortH();
        int ret = DEVICE_OK;
        char command[50];
        unsigned int leng;

        int dir = 0;
        if ( direction == "Low" ) dir = 1;
        leng = snprintf(( char* ) command, 50, "F,%d\r\n", dir);
        ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
        if ( ret != DEVICE_OK )  return ret;

        hub->SetTimedOutput(false);
    }
    return DEVICE_OK;
}
int CESP32Switch::OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::ostringstream os;
    os << "Switch::OnDelay ";
    LogMessage(os.str().c_str(), false);

    if ( eAct == MM::BeforeGet ) {
        pProp->Set(( long int ) currentDelay_);
    }
    else if ( eAct == MM::AfterSet )
    {
        long prop;
        pProp->Get(prop);
        currentDelay_ = ( int ) prop;
    }

    return DEVICE_OK;
}

// Command: 11 now = "I" set number of "I"terations
int CESP32Switch::OnRepeatTimedPattern(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::ostringstream os;
    os << "Switch::OnRepeatTimedPattern ";
    LogMessage(os.str().c_str(), false);

    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;

    // Perhaps shorten and clarify logic to one line?
    // if ((eAct != MM::BeforeGet) && (eAct == MM::AfterSet)) {

    if ( eAct == MM::BeforeGet ) {
    }
    else if ( eAct == MM::AfterSet )
    {
        MMThreadGuard myLock(hub->GetLock());

        long prop;
        pProp->Get(prop);

        hub->PurgeComPortH();
        int ret = DEVICE_OK;
        char command[50];
        unsigned int leng;
        leng = snprintf(( char* ) command, 50, "I,%d\r\n", prop);
        ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
        if ( ret != DEVICE_OK )  return ret;

        hub->SetTimedOutput(false);
    }

    return DEVICE_OK;
}

// CESP32Shutter implementation
// Justin Notes: I'm a bit mystified by the "shutter" and when it is called. Currently it issues a "Switch" 
// command - I find that confusing.

// CESP32Shutter implementation
CESP32Shutter::CESP32Shutter() : initialized_(false), name_(g_DeviceNameESP32Shutter)
{
    InitializeDefaultErrorMessages();
    EnableDelay();

    SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The ESP32 Hub device is needed to create this device");

    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameESP32Shutter, MM::String, true);
    assert(DEVICE_OK == ret);

    // Description
    ret = CreateProperty(MM::g_Keyword_Description, "ESP32 shutter driver", MM::String, true);
    assert(DEVICE_OK == ret);

    // parent ID display
    CreateHubIDProperty();
}
CESP32Shutter::~CESP32Shutter()
{
    Shutdown();
}
void CESP32Shutter::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameESP32Shutter);
}
bool CESP32Shutter::Busy()
{
    MM::MMTime interval = GetCurrentMMTime() - changedTime_;

    return interval < MM::MMTime::fromMs(GetDelayMs());
}
int CESP32Shutter::Initialize()
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() ) {
        return ERR_NO_PORT_SET;
    }
    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    // set property list
    // -----------------

    // removed comment from OnOff
    // set shutter into the off state  ???wot
    // WriteToPort(0);

    // OnOff
    CPropertyAction* pAct = new CPropertyAction(this, &CESP32Shutter::OnOnOff);
    int ret = CreateProperty("OnOff", "0", MM::Integer, false, pAct);
    if ( ret != DEVICE_OK ) return ret;

    std::vector<std::string> vals;
    vals.push_back("0");
    vals.push_back("1");
    ret = SetAllowedValues("OnOff", vals);
    if ( ret != DEVICE_OK ) return ret;

    ret = UpdateStatus();
    if ( ret != DEVICE_OK ) return ret;

    changedTime_ = GetCurrentMMTime();
    initialized_ = true;

    return DEVICE_OK;
}
int CESP32Shutter::Shutdown()
{
    if ( initialized_ )
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}
int CESP32Shutter::SetOpen(bool open)
{
    std::ostringstream os;
    os << "Shutter::SetOpen open= " << open;
    LogMessage(os.str().c_str(), false);

    if ( open ) return SetProperty("OnOff", "1");
    else      return SetProperty("OnOff", "0");
}
int CESP32Shutter::GetOpen(bool& open)
{
    std::ostringstream os;
    os << "Shutter::GetOpen open= " << open;
    LogMessage(os.str().c_str(), false);

    char buf[MM::MaxStrLength];
    int ret = GetProperty("OnOff", buf);
    if ( ret != DEVICE_OK ) return ret;

    long pos = atol(buf);
    pos > 0 ? open = true : open = false;

    return DEVICE_OK;
}
int CESP32Shutter::Fire(double /*deltaT*/)
{
    return DEVICE_UNSUPPORTED_COMMAND;
}

// e.g. "x0101100" turns off channels 0,1,5,7.. 
// Note: channels 2,3,6 will switch on at their preset output level.. "shutter" is messing with the switches!
// must try to separate church and state.

// Command: 1 Now "H,1/0" Set s"H"utter open or closed (this can be hardware or electronic shuttering) or both!          
int CESP32Shutter::WriteToPort(long value)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    MMThreadGuard myLock(hub->GetLock());

    // keep the lowest 8 bits -
    value = value & 255;
    if ( hub->IsLogicInverted() )  value = ~value;

    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    // create command buffer
    char command[50];
    unsigned int leng;

    // load command buffer with the command string "lf" terminated.
    leng = snprintf(( char* ) command, 50, "H,%d\r\n", value);
    ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
    if ( ret != DEVICE_OK )  return ret;

    hub->SetTimedOutput(false);

    std::ostringstream os;
    os << "Shutter::WriteToPort Command= " << command;
    LogMessage(os.str().c_str(), false);

    return DEVICE_OK;
}

// I think this method should be part of the "Switch" Class and should call Switch::WritetoPort
// This method is called in an inapproriate manner during MDA.. need to sort it out
// OnOnOff should be moved to "Switch::" : it's called during MDA and operates the "switch" indirectly
// by issuing the "S" command

// Action handlers - don't seem to work correctly at present. I've modded it and it now fails!
int CESP32Shutter::OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( eAct == MM::BeforeGet )
    {
        // use cached state
        pProp->Set(( long ) hub->GetShutterState());
    }
    else if ( eAct == MM::AfterSet )
    {
        long pos;
        pProp->Get(pos);
        int ret;
        if ( pos == 0 )
            ret = WriteToPort(0); // Shutter closed (and/or write zeros to all LEDs)
        else
            //ret = WriteToPort(1); // Shutter open (and/or restore LED O/P pattern) 
                                    // Set top bit to open shutter
           // ret = WriteToPort(pos | 128);
            ret = WriteToPort((hub->GetSwitchState()) | 128); // restore old setting NO - let the ESP32 do this logic!

        if ( ret != DEVICE_OK )  return ret;

        hub->SetShutterState(pos);
        changedTime_ = GetCurrentMMTime();
    }

    std::ostringstream os;
    os <<  "Shutter::OnOnOff hub->GetSwitchState()= " << hub->GetSwitchState();
    LogMessage(os.str().c_str(), false);

    return DEVICE_OK;
}

// CESP32DA implementation:
// Justin's Notes:
// This "Device" only sends one command "O,chan,volts"
// Summary:
// => OnVolts Action Handler is called by an event handler in MM.
// It subsequently calls:
//      => SetSignal which optionally "gates" the O/P to "zero Volts"
//      => WriteSignal scales the value to 12 bits (but now does nothing!)
//      => WritetoPort - Sends "O,(int) chan,(float) volts" to ESP32
// => OnMaxVolts & => OnChannel just keep Volts and Channels within bounds
// 
// If we need anything fast or device-specific do it on the ESP32. 
// USB transmit/receive latency will dominate.

// CESP32DA implementation
CESP32DA::CESP32DA(int channel) :
    busy_(false),
    minV_(0.0),
    maxV_(100.0),
    volts_(0.0),
    gatedVolts_(0.0),
    channel_(channel),
    maxChannel_(7),
    gateOpen_(true)
{
    InitializeDefaultErrorMessages();

    // add custom error messages
    SetErrorText(ERR_UNKNOWN_POSITION, "Invalid position (state) specified");
    SetErrorText(ERR_INITIALIZE_FAILED, "Initialization of the device failed");
    SetErrorText(ERR_WRITE_FAILED, "Failed to write data to the device");
    SetErrorText(ERR_CLOSE_FAILED, "Failed closing the device");
    SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The ESP32 Hub device is needed to create this device");

    /* Channel property is not needed
    CPropertyAction* pAct = new CPropertyAction(this, &CESP32DA::OnChannel);
    CreateProperty("Channel", channel_ == 1 ? "1" : "2", MM::Integer, false, pAct, true);
    for (int i=1; i<= 2; i++){
       std::ostringstream os;
       os << i;
       AddAllowedValue("Channel", os.str().c_str());
    }
    */

    CPropertyAction* pAct = new CPropertyAction(this, &CESP32DA::OnMaxVolt);
    CreateProperty("Power %", "100", MM::Float, false, pAct, true);

    if (channel_ == 0)      name_ =  g_DeviceNameESP32PWM0;
    else if (channel_ == 1) name_ =  g_DeviceNameESP32PWM1;
    else if (channel_ == 2) name_ =  g_DeviceNameESP32PWM2;
    else if (channel_ == 3) name_ =  g_DeviceNameESP32PWM3;
    else if (channel_ == 4) name_ =  g_DeviceNameESP32PWM4;

    //name_ = channel_ == 1 ? g_DeviceNameESP32DA1 : g_DeviceNameESP32DA2;

    // Description
    int nRet = CreateProperty(MM::g_Keyword_Description, "ESP32 DAC driver", MM::String, true);
    assert(DEVICE_OK == nRet);

    // Name
    nRet = CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);
    assert(DEVICE_OK == nRet);

    // parent ID display
    CreateHubIDProperty();
}
CESP32DA::~CESP32DA()
{
    Shutdown();
}
void CESP32DA::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, name_.c_str());
}
int CESP32DA::Initialize()
{
    CESP32Hub* hub = static_cast<CESP32Hub*>(GetParentHub());
    if (!hub || !hub->IsPortAvailable()) {
        return ERR_NO_PORT_SET;
    }
    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    // set property list
    // -----------------

    // State
    // -----
    CPropertyAction* pAct = new CPropertyAction(this, &CESP32DA::OnVolts);
    int nRet = CreateProperty("Volts", "0.0", MM::Float, false, pAct);
    if (nRet != DEVICE_OK)
        return nRet;
    SetPropertyLimits("Volts", minV_, maxV_);

    nRet = UpdateStatus();

    if (nRet != DEVICE_OK) return nRet;

    initialized_ = true;

    return DEVICE_OK;
}
int CESP32DA::Shutdown()
{
    initialized_ = false;
    return DEVICE_OK;
}

// Command: 3 now = "O,chan,value" Output DAC value
int CESP32DA::WriteToPort(double value)
{
    CESP32Hub* hub = static_cast<CESP32Hub*>(GetParentHub());
    if (!hub || !hub->IsPortAvailable())
        return ERR_NO_PORT_SET;

    MMThreadGuard myLock(hub->GetLock());

    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    char command[50];
    unsigned int leng;

    leng = snprintf((char*)command, 50, "O,%d,%3.3f\r\n", ( int ) channel_, value);
    ret = hub->WriteToComPortH((const unsigned char*)command, leng);
    if (ret != DEVICE_OK)  return ret;

    hub->SetTimedOutput(false);

    return DEVICE_OK;
}
int CESP32DA::WriteSignal(double volts)
{
    double value = volts;  // ( (volts - minV_) / maxV_ * 4095);

    std::ostringstream os;
    os << "Volts= " << volts << " MaxVoltage= " << maxV_ << " digitalValue= " << value;
    LogMessage(os.str().c_str(), false);

    return WriteToPort(value);
}
int CESP32DA::SetSignal(double volts)
{
    volts_ = volts;
    if (gateOpen_) {
        gatedVolts_ = volts_;
        return WriteSignal(volts_);
    }
    else {
        gatedVolts_ = 0;
    }

    return DEVICE_OK;
}
int CESP32DA::SetGateOpen(bool open)
{
    if (open) {
        gateOpen_ = true;
        gatedVolts_ = volts_;
        return WriteSignal(volts_);
    }
    gateOpen_ = false;
    gatedVolts_ = 0;
    return WriteSignal(0.0);

}

// Action handlers
int CESP32DA::OnVolts(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        // nothing to do, let the caller use cached property
    }
    else if (eAct == MM::AfterSet)
    {
        double volts;
        pProp->Get(volts);
        return SetSignal(volts);
    }

    return DEVICE_OK;
}
int CESP32DA::OnMaxVolt(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(maxV_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(maxV_);
        if (HasProperty("Volts"))
            SetPropertyLimits("Volts", 0.0, maxV_);

    }
    return DEVICE_OK;
}
int CESP32DA::OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set((long int)channel_);
    }
    else if (eAct == MM::AfterSet)
    {
        long channel;
        pProp->Get(channel);
        if (channel >= 0 && ((unsigned)channel <= maxChannel_))
            channel_ = channel;
    }
    return DEVICE_OK;
}

/*
Below, I have folded-in (and modified) the PIEZOCONCEPT XYZ stage adapter
so we can control focus and move an x-y stage connected to our ESP32 board (see firmware).
So, we have two new blocks of code:

CESP32Stage & CESP32XYStage

Notes:
Stage and Focus noise:
On the ESP side, I gave the XYZ channels 16-bit resolution (which allows only 1kHz PWM (Fpwm) on ESP32).
If we low-pass filter to 1 Hz (Fc) we will still get "Ao * (Fc/Fpwm)"  of p-p PWM ripple noise 
which is too high for most applications.
Suggest using 20-bit DACs (e.g. Analog devices: DAC1220) which can be 'chip selected' using 
the "DAC" GPIO pins (see ESP32 Firmware code) and values sent using the ESP SPI interface. 
Easy to "mod" the ESP32 firmware and leave this code unchanged.
*/

// CESP32Stage Implementation
CESP32Stage::CESP32Stage() :
    stepSizeUm_(0.0001),
    pos_um_(0.0),
    busy_(false),
    initialized_(false),
    lowerLimit_(0.0),
    upperLimit_(100.0)
{
    InitializeDefaultErrorMessages();

    CreateHubIDProperty();
}

CESP32Stage::~CESP32Stage()
{
    Shutdown();
}

void CESP32Stage::GetName(char* Name) const
{
    CDeviceUtils::CopyLimitedString(Name, g_DeviceNameStage);
}

int CESP32Stage::Initialize()
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );

    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;

    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    if ( initialized_ ) return DEVICE_OK;

    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameStage, MM::String, true);
    if ( DEVICE_OK != ret ) return ret;

    // Description
    ret = CreateProperty(MM::g_Keyword_Description, "ESP32 Z stage driver", MM::String, true);
    if ( DEVICE_OK != ret ) return ret;

    double travel;

    ret = hub->GetAxisInfoH(2, travel);
    if ( DEVICE_OK != ret ) return ret;

    upperLimit_ = travel;
    //Should we initialise stepsize as we do for X- Y- axes 
    //Suggest e.g. perhaps 10 digital bits per mouse-wheel click?
    //stepSizeUm_ = travel / 6553.6;

    ret = UpdateStatus();
    if ( ret != DEVICE_OK ) return ret;

    initialized_ = true;

    std::ostringstream os;
    os << "Stage::Initialize (exit)";
    LogMessage(os.str().c_str(), false);

    return DEVICE_OK;
}

int CESP32Stage::Shutdown()
{
    if ( initialized_ ) initialized_ = false;
    return DEVICE_OK;
}

int CESP32Stage::GetPositionUm(double& pos)
{
    pos = pos_um_;
    return DEVICE_OK;
}

int CESP32Stage::SetPositionUm(double pos)
{
    int ret = MoveZ(pos);
    if ( ret != DEVICE_OK ) return ret;
    return OnStagePositionChanged(pos);
}

int CESP32Stage::GetPositionSteps(long& steps)
{
    double posUm;
    int ret = GetPositionUm(posUm);
    if ( ret != DEVICE_OK ) return ret;

    steps = static_cast< long >( posUm / GetStepSize() );
    return DEVICE_OK;
}

int CESP32Stage::SetPositionSteps(long steps)
{
    return SetPositionUm(steps * GetStepSize());
}

// "Z" command - move to new Z-position (Focus up/down)
int CESP32Stage::MoveZ(double pos)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );

    if ( !hub || !hub->IsPortAvailable() ) return ERR_HUB_UNAVAILABLE;

    if ( pos > upperLimit_ ) pos = upperLimit_;
    if ( pos < lowerLimit_ ) pos = lowerLimit_;

    char buf[25];
    int length = sprintf(buf, "Z,%3.3f\r\n", pos);

    std::stringstream ss;
    ss << "Command: " << buf << "  Position set: " << pos;
    LogMessage(ss.str().c_str(), false);

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();

    int ret = hub->WriteToComPortH(( unsigned char* ) buf, length);
    if ( ret != DEVICE_OK ) return ret;

    MM::MMTime startTime = GetCurrentMMTime();
    unsigned long bytesRead = 0;
    unsigned char answer[10];
    while ( ( bytesRead < 10 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br;
        ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
        if ( answer[bytesRead] == '\r' )  break;
        bytesRead += br;
    }

    answer[bytesRead] = 0; // string terminator

    hub->PurgeComPortH();
    if ( ret != DEVICE_OK ) return ret;

    if ( answer[0] != 'Z' )  return ERR_COMMUNICATION;

    hub->SetTimedOutput(false);

    std::ostringstream os;
    os << "MoveZ Z," << pos;
    LogMessage(os.str().c_str(), false);

    pos_um_ = pos;

    return DEVICE_OK;
}

bool CESP32Stage::Busy()
{
    return false;
}

CESP32XYStage::CESP32XYStage() : CXYStageBase<CESP32XYStage>(),
stepSize_X_um_(0.1),
stepSize_Y_um_(0.1),
posX_um_(0.0),
posY_um_(0.0),
busy_(false),
initialized_(false),
lowerLimitX_(0.0),
upperLimitX_(200.0),
lowerLimitY_(0.0),
upperLimitY_(200.0)
{
    InitializeDefaultErrorMessages();

    // parent ID display
    CreateHubIDProperty();

    // step size
    CPropertyAction* pAct = new CPropertyAction(this, &CESP32XYStage::OnXStageMinPos);
    CreateProperty(g_PropertyXMinUm, "0", MM::Float, false, pAct, true);

    pAct = new CPropertyAction(this, &CESP32XYStage::OnXStageMaxPos);
    CreateProperty(g_PropertyXMaxUm, "200", MM::Float, false, pAct, true);

    pAct = new CPropertyAction(this, &CESP32XYStage::OnYStageMinPos);
    CreateProperty(g_PropertyYMinUm, "0", MM::Float, false, pAct, true);

    pAct = new CPropertyAction(this, &CESP32XYStage::OnYStageMaxPos);
    CreateProperty(g_PropertyYMaxUm, "200", MM::Float, false, pAct, true);
}

CESP32XYStage::~CESP32XYStage()
{
    Shutdown();
}

void CESP32XYStage::GetName(char* Name) const
{
    CDeviceUtils::CopyLimitedString(Name, g_DeviceNameXYStage);
}

int CESP32XYStage::Initialize()
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;

    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    if ( initialized_ ) return DEVICE_OK;

    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameXYStage, MM::String, true);
    if ( DEVICE_OK != ret )  return ret;

    // Description
    ret = CreateProperty(MM::g_Keyword_Description, "XY stage driver", MM::String, true);
    if ( DEVICE_OK != ret ) return ret;

    double travelX, travelY;
    ret = hub->GetAxisInfoH(0, travelX);
    if ( DEVICE_OK != ret )  return ret;

    ret = hub->GetAxisInfoH(1, travelY);
    if ( DEVICE_OK != ret )  return ret;

    // here the stepsize is set to 1 digital bit ?..Cut-and-paste from PIEZOCONCEPT.
    upperLimitX_ = travelX;
    stepSize_X_um_ = travelX / 65535;
    upperLimitY_ = travelY;
    stepSize_Y_um_ = travelY / 65535;

    ret = UpdateStatus();
    if ( ret != DEVICE_OK ) return ret;

    initialized_ = true;

    return DEVICE_OK;
}

int CESP32XYStage::Shutdown()
{
    if ( initialized_ ) initialized_ = false;
    return DEVICE_OK;
}

bool CESP32XYStage::Busy()
{
    return false;
}

int CESP32XYStage::GetPositionSteps(long& x, long& y)
{
    x = ( long ) ( posX_um_ / stepSize_X_um_ );
    y = ( long ) ( posY_um_ / stepSize_Y_um_ );

    std::stringstream ss;
    ss << "GetPositionSteps :=" << x << "," << y;
    LogMessage(ss.str(), false);
    return DEVICE_OK;
}

int CESP32XYStage::SetPositionSteps(long x, long y)
{
    double posX = x * stepSize_X_um_;
    double posY = y * stepSize_Y_um_;

    std::stringstream ss;
    ss << "Current position = " << posX_um_ << "," << posY_um_ << " \n Commanded position = " << posX << "," << posY;
    LogMessage(ss.str(), false);

    int ret = DEVICE_OK;

    if ( posX_um_ != posX )
    {
        ret = MoveX(posX);
        if ( ret != DEVICE_OK ) return ret;
    }
    if ( posY_um_ != posY )
    {
        ret = MoveY(posY);
        if ( ret != DEVICE_OK ) return ret;
    }
    return OnXYStagePositionChanged(posX_um_, posY_um_);
}

int CESP32XYStage::SetRelativePositionSteps(long x, long y)
{
    long curX, curY;
    GetPositionSteps(curX, curY);

    return SetPositionSteps(curX + x, curY + y);
}

// "X" command - move to new x-position (stage translate)
int CESP32XYStage::MoveX(double posUm)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() ) return ERR_HUB_UNAVAILABLE;

    if ( posUm < lowerLimitX_ ) posUm = lowerLimitX_;
    if ( posUm > upperLimitX_ ) posUm = upperLimitX_;

    char buf[25];
    int length = sprintf(buf, "X,%3.3f\r\n", posUm);

    std::stringstream ss;
    ss << "Command: " << buf << "  Position set: " << posUm;
    LogMessage(ss.str().c_str(), false);

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();

    int ret = hub->WriteToComPortH(( unsigned char* ) buf, length);
    if ( ret != DEVICE_OK ) return ret;

    MM::MMTime startTime = GetCurrentMMTime();
    unsigned long bytesRead = 0;
    unsigned char answer[10];
    while ( ( bytesRead < 10 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br;
        ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
        if ( answer[bytesRead] == '\r' )  break;
        bytesRead += br;
    }

    answer[bytesRead] = 0; // string terminator

    hub->PurgeComPortH();
    hub->SetTimedOutput(false);

    if ( ret != DEVICE_OK ) return ret;

    if ( answer[0] != 'X' )  return ERR_COMMUNICATION;

    std::ostringstream os;
    os << "X-Answer= " << answer;
    LogMessage(os.str().c_str(), false);

    posX_um_ = posUm;
    return DEVICE_OK;
}

// "Y" command - move to new y-position (stage translate)
int CESP32XYStage::MoveY(double posUm)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );

    if ( !hub || !hub->IsPortAvailable() ) return ERR_HUB_UNAVAILABLE;

    if ( posUm < lowerLimitY_ ) posUm = lowerLimitY_;
    if ( posUm > upperLimitY_ ) posUm = upperLimitY_;

    char buf[25];
    int length = sprintf(buf, "Y,%3.3f\r\n", posUm);

    std::stringstream ss;
    ss << "Command: " << buf << "  Position set: " << posUm;
    LogMessage(ss.str().c_str(), false);

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();

    int ret = hub->WriteToComPortH(( unsigned char* ) buf, length);
    if ( ret != DEVICE_OK ) return ret;

    MM::MMTime startTime = GetCurrentMMTime();
    unsigned long bytesRead = 0;
    unsigned char answer[10];
    while ( ( bytesRead < 10 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br;
        ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
        if ( answer[bytesRead] == '\r' )  break;
        bytesRead += br;
    }

    answer[bytesRead] = 0; // string terminator

    hub->PurgeComPortH();
    hub->SetTimedOutput(false);

    if ( ret != DEVICE_OK ) return ret;

    if ( answer[0] != 'Y' )  return ERR_COMMUNICATION;

    std::ostringstream os;
    os << "Y-Answer= " << answer;
    LogMessage(os.str().c_str(), false);

    posY_um_ = posUm;
    return DEVICE_OK;
}

int CESP32XYStage::OnXStageMinPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    if ( eAct == MM::BeforeGet )
    {
        pProp->Set(lowerLimitX_);
        ret = DEVICE_OK;
    }
    else if ( eAct == MM::AfterSet )
    {
        double limit;
        pProp->Get(limit);
        lowerLimitX_ = limit;

        ret = DEVICE_OK;
    }

    return ret;
}

int CESP32XYStage::OnXStageMaxPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    if ( eAct == MM::BeforeGet )
    {
        pProp->Set(upperLimitX_);
        ret = DEVICE_OK;
    }
    else if ( eAct == MM::AfterSet )
    {
        double limit;
        pProp->Get(limit);
        upperLimitX_ = limit;

        ret = DEVICE_OK;
    }

    return ret;
}

int CESP32XYStage::OnYStageMinPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    if ( eAct == MM::BeforeGet )
    {
        pProp->Set(lowerLimitY_);
        ret = DEVICE_OK;
    }
    else if ( eAct == MM::AfterSet )
    {
        double limit;
        pProp->Get(limit);
        lowerLimitY_ = limit;

        ret = DEVICE_OK;
    }

    return ret;
}

int CESP32XYStage::OnYStageMaxPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    if ( eAct == MM::BeforeGet )
    {
        pProp->Set(upperLimitY_);
        ret = DEVICE_OK;
    }
    else if ( eAct == MM::AfterSet )
    {
        double limit;
        pProp->Get(limit);
        upperLimitY_ = limit;

        ret = DEVICE_OK;
    }

    return ret;
}


// The "Input" functions below need to be commented
CESP32Input::CESP32Input() :
    mThread_(0),
    pin_(0),
    name_(g_DeviceNameESP32Input)
{
    std::string errorText = "To use the Input function you need firmware version 1 or higher";
    SetErrorText(ERR_VERSION_MISMATCH, errorText.c_str());

    CreateProperty("Pin", "All", MM::String, false, 0, true);
    AddAllowedValue("Pin", "All");
    AddAllowedValue("Pin", "0");
    AddAllowedValue("Pin", "1");
    AddAllowedValue("Pin", "2");
    AddAllowedValue("Pin", "3");
    AddAllowedValue("Pin", "4");
    AddAllowedValue("Pin", "5");

    CreateProperty("Pull-Up-Resistor", g_On, MM::String, false, 0, true);
    AddAllowedValue("Pull-Up-Resistor", g_On);
    AddAllowedValue("Pull-Up-Resistor", g_Off);

    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);
    assert(DEVICE_OK == ret);

    // Description
    ret = CreateProperty(MM::g_Keyword_Description, "ESP32 shutter driver", MM::String, true);
    assert(DEVICE_OK == ret);

    // parent ID display
    CreateHubIDProperty();
}

CESP32Input::~CESP32Input()
{
    Shutdown();
}

int CESP32Input::Shutdown()
{
    if ( initialized_ )
        delete( mThread_ );
    initialized_ = false;
    return DEVICE_OK;
}


int CESP32Input::Initialize()
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;

    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    char ver[MM::MaxStrLength] = "0";
    hub->GetProperty(g_versionProp, ver);

    // we should get ASCII "1" (minimally) back from the ESP - convert to (int)
    int version = atoi(ver);
    if ( version < g_Min_MMVersion )  return ERR_VERSION_MISMATCH;


    // This bit of code needs some comments!
    // Are we going to use the GPIO lines for Analogue or Digital; I/O
    // I think the idea is to request setup of the INPUT_PULLUP state 
    // on the GPIO pins on the microcontroller... but it is not clear

    int ret = GetProperty("Pin", pins_);
    if ( ret != DEVICE_OK ) return ret;

    // pin_ = ascii to integer of pins_ (i.e. 0,1,2,3,4,5) unknown value if pins_="All"
    if ( strcmp("All", pins_) != 0 )  pin_ = atoi(pins_);

    ret = GetProperty("Pull-Up-Resistor", pullUp_);
    if ( ret != DEVICE_OK ) return ret;

    // Digital Input
    CPropertyAction* pAct = new CPropertyAction(this, &CESP32Input::OnDigitalInput);
    ret = CreateProperty("DigitalInput", "0", MM::Integer, true, pAct);
    if ( ret != DEVICE_OK ) return ret;

    int start = 0;
    int end = 5;

    //if pins_ != "All" then start=end=pin_ 
    if ( strcmp("All", pins_) != 0 ) {
        start = pin_;
        end = pin_;
    }

    for ( long i = start; i <= end; i++ )
    {

// This does something and it would be very helpful to know what*******
        CPropertyActionEx* pExAct = new CPropertyActionEx(this, &CESP32Input::OnAnalogInput, i);
        std::ostringstream os;
        os << "AnalogInput= " << i;
        ret = CreateProperty(os.str().c_str(), "0.0", MM::Float, true, pExAct);
        if ( ret != DEVICE_OK ) return ret;

        // set pull up resistor state for this pin
        if ( strcmp(g_On, pullUp_) == 0 ) SetPullUp(i, 1);
        else SetPullUp(i, 0);

    }

    mThread_ = new ESP32InputMonitorThread(*this);
    mThread_->Start();

    initialized_ = true;

    return DEVICE_OK;
}

void CESP32Input::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

bool CESP32Input::Busy()
{
    return false;
}

// Justin Notes: "GetDigitalInput" is polled for state change every 0.5sec by the thread below.
// In the original ESP code we were testing the Analogue Inputs for a digital change..
// That's fine.. but we should set this up in a more obvious way.
// suggest we assign some of the GPIO lines as Digital I/O and some as Analogue Input.
// Currently this is unclear to me...I don't know what Analogue inputs we want to monitor.

// Command: 40 now "L" logic in - returns "L,l\r\n" pin High/Low
int CESP32Input::GetDigitalInput(long* state)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    MMThreadGuard myLock(hub->GetLock());

    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    char command[50];
    unsigned int leng;

    int testPin = pin_; //(test pin 0->5)
    if ( strcmp("All", pins_) == 0 ) testPin = 6; // (test if all pins set)

    leng = snprintf(( char* ) command, 50, "L,%d\r\n", testPin);
    ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
    if ( ret != DEVICE_OK )  return ret;

    // we expect a reply like "L,1\r\n" 1 or 0 if testPin is High/Low
    MM::MMTime startTime = GetCurrentMMTime();
    unsigned long bytesRead = 0;
    char answer[10];
    // Chug through the I/O buffer, byte-by-byte and stop at lf!
    while ( ( bytesRead < 10 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br;
        ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
        if ( answer[bytesRead] == '\r' )  break;
        bytesRead += br;
    }

    answer[bytesRead] = 0; // string terminator

    // discard anything left in the IO buffer
    hub->PurgeComPortH();
    hub->SetTimedOutput(false);

    if ( ret != DEVICE_OK ) return ret;

    int num;
    char com[1];
    sscanf(( const char* ) answer, "%1s,%d", com, &num);

    // should give com[0]='L' and num = 1 or 0 

    if ( com[0] != 'L' )  return ERR_COMMUNICATION;

    *state = ( long ) num;

    std::ostringstream os;
    os << "GetDigitalInput_State=" << *state << " testPin = " << testPin;
    LogMessage(os.str().c_str(), false);

    return DEVICE_OK;
}
int CESP32Input::ReportStateChange(long newState)
{
    std::ostringstream os;
    os << newState;
    return OnPropertyChanged("DigitalInput", os.str().c_str());
}

// Action handlers
int CESP32Input::OnDigitalInput(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    // This sets a property listed in the MMDevice Property.h file (very limited commenting)
    // "Set" is an "Action Functor" (handler?)  
    // "eACT" seems common to all "On" functions - so probably an Event listener?
    // Anyway, it wants a (long) value for "state" which becomes a virtual Boolean, apparently! 

    if ( eAct == MM::BeforeGet )
    {
        long state;

        // get the state of the selected pin 0-5 or all pins (which I have made to be "6" in my ESP code)
        int ret = GetDigitalInput(&state);
        if ( ret != DEVICE_OK ) return ret;
        pProp->Set(state);
    }

    return DEVICE_OK;
}

// Commands: 41 now = "A,chan" analogue read channel
int CESP32Input::OnAnalogInput(MM::PropertyBase* pProp, MM::ActionType eAct, long  channel)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() ) return ERR_NO_PORT_SET;

    if ( eAct == MM::BeforeGet )
    {

        MMThreadGuard myLock(hub->GetLock());
        hub->PurgeComPortH();
        int ret = DEVICE_OK;
        char command[50];
        unsigned int leng;

        leng = snprintf(( char* ) command, 50, "A,%d\r\n", ( int ) channel);
        ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
        if ( ret != DEVICE_OK )  return ret;

        // We expect a reply like "A,597\r\n" analogue value on channel
        // I think we need to use the hub to communicate via the rather "ReadFromComPortH" command??
        unsigned long bytesRead = 0;
        char answer[20];
        MM::MMTime startTime = GetCurrentMMTime();
        while ( ( bytesRead < 20 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 1000) ) {
            unsigned long br;
            ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
            if ( answer[bytesRead] == '\r' )  break;
            bytesRead += br;
        }

        answer[bytesRead] = 0; // replace \r with string terminator
        
        hub->PurgeComPortH();
        hub->SetTimedOutput(false);

        if ( ret != DEVICE_OK ) return ret;

        int num;
        char com[1];
        sscanf(( const char* ) answer, "%1s,%d", com, &num);
        // should give com[0]='A' and num = 597 or something!

        std::ostringstream os;
        os << "AnswerTo A," << channel << " = " << answer;
        LogMessage(os.str().c_str(), false);

        if ( com[0] != 'A' )  return ERR_COMMUNICATION;

        pProp->Set(( long ) num);
    }
    return DEVICE_OK;
}

// Commands: 42 - now "D" set digital pull-up ?? needs commenting - not sure why we want to do this dynamically.
int CESP32Input::SetPullUp(int pin, int state)
{
    CESP32Hub* hub = static_cast< CESP32Hub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    char command[50];
    unsigned int leng;

    leng = snprintf(( char* ) command, 50, "D,%d,%d\r\n", ( int ) pin, ( int ) state);
    ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
    if ( ret != DEVICE_OK )  return ret;

    hub->PurgeComPortH();
    hub->SetTimedOutput(false);

    return DEVICE_OK;
}

ESP32InputMonitorThread::ESP32InputMonitorThread(CESP32Input& aInput) :
    state_(0),
    aInput_(aInput)
{
}
ESP32InputMonitorThread::~ESP32InputMonitorThread()
{
    Stop();
    wait();
}

// I think this thread is a 0.5s delay loop that polls the Digital I/O lines and reports changes
// It looks like an infinite background polling-thread...until we hit a comms error
// It has probably been messed-up by my meddling!

int ESP32InputMonitorThread::svc()
{
    while ( !stop_ )
    {
        long state;
        int ret = aInput_.GetDigitalInput(&state);
        if ( ret != DEVICE_OK )
        {
            stop_ = true; // on communication error drop out of loop and don't retry ??
            return ret;
        }

        if ( state != state_ )
        {
            aInput_.ReportStateChange(state);
            state_ = state;
        }
        CDeviceUtils::SleepMs(500);
    }
    return DEVICE_OK;
}
void ESP32InputMonitorThread::Start()
{
    stop_ = false;
    activate();
}