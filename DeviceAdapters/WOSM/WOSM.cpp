///////////////////////////////////////////////////////////////////////////////
// FILE:          WOSM.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for Warwick Open-Source Microscope "WOSM"
//                HTTPS://www.wosmic.org
//                This device Adapter is a modded Arduino Device Adapter
//                
// COPYRIGHT:     University of Warwick, UK 2023
// LICENSE:       LGPL
// AUTHORS:       Justin E. Molloy & Nicholas J. Carter
// 
//                
// 
// Software is modified from Arduino Adapter written by:
// Copyright :    University of California, San Francisco, 2008
//                Nico Stuurman, nico@cmp.ucsf.edu, 11/09/2008
//                automatic device detection by Karl Hoover
//
//   This device Adapter also includes code modified from the PiezoConcept XYZ.
//
///////////////////////////////////////////////////////////////////////////////
// 
// The IP address of the WOSM controller is available on the WOSM LCD screen.
// e.g. "192.168.10.100" use port 23 or 1023 
// Upon connection via TCP/IP the WOSM reports its model ID and networking information 
// and requests the user to enter an "admin password: " (default is "wosm")
// It then enters "command mode" signified by the prompt:
// 
// W>
//
// ============================
//  WOSM Command-Set over-view:
// ============================
// Here, we use a small sub-set of commands to control the WOSM
// The full command-set for the WOSM is at: http://wosmic.org/mcu/commands.php
// 
// Hint: You can enter and test commands using PuTTy.
//
// System Commands:
//          sys_    commands=    time, date, usec, uptime etc....
//                  example:     W>sys_time  ->returns system time in UTC format. 
// 
// Temperature Sensors:
//          temp_   commands=    ser, ref, val etc...
//                  example:     W>temp_val stage  ->temp (oC*256) of sensor called "stage"
// 
// Addressable Serial LEDs:
//          led_    commands=    conf, buff, mask, bright, on
//                  example:     W>led_buff 0xFF00FF 0x000000010  ->make LED number 9 purple
// 
// Counters (2 * 16-bit counters on digital lines "q" & "r"):
//          cnt_    commands=    en, clr, val
//                  example:     W>cnt_clr q  ->clear counter on digtal line "q"
// 
// KeyPad, Rotary encoder (monitor remote controller box - send messages to its screen):
//          kyp_    commands=   val, rot, lcd_txt
//                  examples:   W>kyp_rot  ->read rotary encoder clicks
//                              W>kyp_lcd_screen t=6s \"\n  Micromanager\n  Connected.\"
// 
// Digital I/O lines (shutters, triggers, interlocks etc - 26 lines available ("a-z"):
//                  - 4 lines ("s,t,u,v") are used to gate the High-Current LED drivers (below) 
//          dig_    commands=   in, out, mode, hilo, lohi etc....
//                  example:    W>dig_in f  ->read T/F on digital line "f" (there are 26 lines "a-z")
// 
// Analogue outputs 8 * 16-bit DAC lines (these lines are prefixed with "p" on the "P"ower daughter-board)
//                  - 4 lines ("ps, pt, pu, pv") have high-current MOSFET drivers (for e.g. LED lamps)
//                  - (Note: these outputs are gated by the dig lines "s,t,u,v")
//                  - 4 lines ("pw, px, py, pz") have low-current 16-bit DACs (for e.g. PZT stage control)               
//          dac_    commands=   ref, out, rate, mode
//                  Note:       The output lines are prefixed with "p" for power board and "m" for mother
//                  examples:   W>dac_mode ps 3  ->line "s" as LED driver mode (3 = current mode)
//                              W>dac_out ps 12.5  ->set line s to 12.5mA output
// 
// Motors, Steppers, Servos:
// (not clear how these motors interact with "Stage" and "Analogue" x,y,z                      
//          mot_    commands=   accl, bckl, dest, pos, out.. etc
//                  example:    W>mot_out m3 27.5  ->moves motor to 27.5um absolute
// 
// Stage control: 
// (not clear how this works/interacts with pw,px,py,pz or with motor controls)
//          stg_    command=    out, val, min, max
//                  examples:   W>stg_out_x r+22.345  ->move stage x-axis relative by 22.345um
// 
// Serial Config (configure the serial ports SPI and RS232):
//          ser_    commands:   config, mode, send
//                  example:    W>ser_config d on pdsel=2 stsel=0  ->enable serial bus "d", 8 data, even parity, 1 stop
// 
// Networking:
//          lan_    commands:   host, config, ip, mac, subnet etc...
//                  examples:   W>lan_mac  ->get the lan MAC address.
// 
// Macro commands:
//          wml_    commands:   run, stop, pause, file_new
//                  examples:   W>wml_file_new col3z4  ->load a new macro command file
//                              W>wml_run col3z4  ->run the loaded macro command file
// User Management:
//          usr_    commands: set, del, name
//                  example:    W>usr_name 10  ->returns authenticated user name index 10
// 
// Note: all commands are "cr+lf" terminated 
//////////////////////////////////////////////////////////////////////////////////////

#include "WOSM.h"
#include "ModuleInterface.h"
#include <math.h>

//$$ string and IO handling
#include <string>
#include <sstream>
#include <cstdio>
using namespace std;
//$$

// Name the Devices something short and sensible
const char* g_DeviceNameWOSMHub = "WOSM-Hub";
const char* g_DeviceNameWOSMSwitch = "WOSM-Switch";
const char* g_DeviceNameWOSMShutter = "WOSM-Shutter";
const char* g_DeviceNameWOSMDAC0 = "WOSM-DAC0";
const char* g_DeviceNameWOSMDAC1 = "WOSM-DAC1";
const char* g_DeviceNameWOSMDAC2 = "WOSM-DAC2";
const char* g_DeviceNameWOSMDAC3 = "WOSM-DAC3";
const char* g_DeviceNameWOSMDAC4 = "WOSM-DAC4";
const char* g_DeviceNameStage = "ZStage";
const char* g_DeviceNameXYStage = "XYStage";
const char* g_DeviceNameWOSMInput = "WOSM-Input";

const char* g_PropertyMinUm = "Z Stage Low Posn(um)";
const char* g_PropertyMaxUm = "Z Stage High Posn(um)";
const char* g_PropertyXMinUm = "X Stage Min Posn(um)";
const char* g_PropertyXMaxUm = "X Stage Max Posn(um)";
const char* g_PropertyYMinUm = "Y Stage Min Posn(um)";
const char* g_PropertyYMaxUm = "Y Stage Max Posn(um)";

// Global info about the state of the WOSM.  This should be folded into a class
const int g_Min_MMVersion = 5;
const int g_Max_MMVersion = 100;
const char* g_versionProp = "Version";
const char* g_normalLogicString = "Normal";
const char* g_invertedLogicString = "Inverted";

const char* g_On = "On";
const char* g_Off = "Off";

// static lock
MMThreadLock CWOSMHub::lock_;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
    // Justin's Notes: I think this is where the HUb and other devices are registered with the MMcore code
    // I think there are currently ~16 recognised Device "types" including:
    // Hub, StateDevice, ShutterDevice, GenericDevice, SignalIODevice, StageDevice(focus?), XYStageDevice, AutofocusDevice, GalvoDevice
    // There seems to be some ambiguity/overlap in what the different Device types actually do.. and the shutter function is used to change
    // the LED output pattern - which is not logical to me.
    // Anyway, seems we call 'em what we like!

    RegisterDevice(g_DeviceNameWOSMHub, MM::HubDevice, "Hub (required)");

    // Note: The "StateDevice" is used to "blank" the SignalIODevice by setting the output to zero.
    RegisterDevice(g_DeviceNameWOSMSwitch, MM::StateDevice, "Switch on/off channels 0 to 10");
    RegisterDevice(g_DeviceNameWOSMShutter, MM::ShutterDevice, "Shutter");
    RegisterDevice(g_DeviceNameWOSMDAC0, MM::SignalIODevice, "DAC channel 0");
    RegisterDevice(g_DeviceNameWOSMDAC1, MM::SignalIODevice, "DAC channel 1");
    RegisterDevice(g_DeviceNameWOSMDAC2, MM::SignalIODevice, "DAC channel 2");
    RegisterDevice(g_DeviceNameWOSMDAC3, MM::SignalIODevice, "DAC channel 3");
    RegisterDevice(g_DeviceNameWOSMDAC4, MM::SignalIODevice, "DAC channel 4");
    RegisterDevice(g_DeviceNameStage, MM::StageDevice, "Z Stage");
    RegisterDevice(g_DeviceNameXYStage, MM::XYStageDevice, "XY Stage");
    RegisterDevice(g_DeviceNameWOSMInput, MM::GenericDevice, "ADC");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    // Justin's Notes: After registering the devices we now "create them" and give them another name.
    // I admit I find it hard to keep track of the different names - especially when
    // things get renamed again within the config file. Just keep your hair on and go
    // with it!

    if ( deviceName == 0 )
        return 0;

    if ( strcmp(deviceName, g_DeviceNameWOSMHub) == 0 )
    {
        return new CWOSMHub;
    }
    else if ( strcmp(deviceName, g_DeviceNameWOSMSwitch) == 0 )
    {
        return new CWOSMSwitch;
    }
    else if ( strcmp(deviceName, g_DeviceNameWOSMShutter) == 0 )
    {
        return new CWOSMShutter;
    }
    else if ( strcmp(deviceName, g_DeviceNameWOSMDAC0) == 0 )
    {
        return new CWOSMDA(0); // channel 0
    }
    else if ( strcmp(deviceName, g_DeviceNameWOSMDAC1) == 0 )
    {
        return new CWOSMDA(1); // channel 1
    }
    else if ( strcmp(deviceName, g_DeviceNameWOSMDAC2) == 0 )
    {
        return new CWOSMDA(2); // channel 2
    }
    else if ( strcmp(deviceName, g_DeviceNameWOSMDAC3) == 0 )
    {
        return new CWOSMDA(3); // channel 3
    }
    else if ( strcmp(deviceName, g_DeviceNameWOSMDAC4) == 0 )
    {
        return new CWOSMDA(4); // channel 4
    }
    else if ( strcmp(deviceName, g_DeviceNameStage) == 0 )
    {
        return new CWOSMStage();
    }
    else if ( strcmp(deviceName, g_DeviceNameXYStage) == 0 )
    {
        return new CWOSMXYStage();
    }
    else if ( strcmp(deviceName, g_DeviceNameWOSMInput) == 0 )
    {
        return new CWOSMInput;
    }
    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

/* CWOSMHUb implementation:
 Justin's Notes: The WOSM HUB is the master "Device" it gives a convenient way to communicate with several
 physical or "nominal" devices connected via a single cable to the PC. The hub heirarchy makes things
 a little more complicated than adapters written for single, stand-alone, physical devices.
 The WOSM controller runs four or five separate "MM_Devices". The firmware running on the WOSM can
 control LED light sources, move an XY stage, a focusing device, open/close a physical shutter,
 control a laser system etc.
*/

//CWOSMHUb implementation
CWOSMHub::CWOSMHub() :
    initialized_(false),
    switchState_(0),
    shutterState_(0),
    hasZStage_(false),
    hasXYStage_(false)
{
    portAvailable_ = false;
    invertedLogic_ = false;
    timedOutputActive_ = false;


    // we can guess what this does...
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_PORT_OPEN_FAILED, "Failed opening WOSM TCP/IP device");
    SetErrorText(ERR_WOSM_NOT_FOUND, "Did not find a WOSM with the correct firmware.  Is the WOSM connected to this IP address?");
    SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The WOSM Hub device is needed to create this device");
    std::ostringstream errorText;
    errorText << "The firmware version on the WOSM is not compatible with this adapter.  Please use firmware version ";
    errorText << g_Min_MMVersion << " to " << g_Max_MMVersion;
    SetErrorText(ERR_VERSION_MISMATCH, errorText.str().c_str());

    // It would be very helpful to know exactly what this does...
    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMHub::OnPort);
    CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

    // and this!
    pAct = new CPropertyAction(this, &CWOSMHub::OnLogic);
    CreateProperty("Logic", g_invertedLogicString, MM::String, false, pAct, true);

    AddAllowedValue("Logic", g_invertedLogicString);
    AddAllowedValue("Logic", g_normalLogicString);
}

CWOSMHub::~CWOSMHub()
{
    Shutdown();
}

void CWOSMHub::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameWOSMHub);
}

bool CWOSMHub::Busy()
{
    return false;
}

/* The Hub sends 2 "commands" / "requests" to the WOSM controller.
= > upon TCP / IP connection we must send the login password
= > "wosm" Expected response is command prompt  "W>"
= > We confirm connection with "kyp_lcd_screen t=6s \"\n  Micromanager\n  connected.\"", crlf
    Once connected...we get the XYZ stage ranges
= > "U,0-2" "U"nderstand the XYZ stage ranges(X = 0, Y = 1, Z = 2)
= > "U,0" expects WOSM response e.g. "U,200" ( X - axis range )
= > "U,2" expects WOSM response e.g. "U,100" ( Z - axis range )
= > if no response or range = 0..MM thinks there's no stage
*/

// Commands: "wosm" login, LCD screen message, set xyz modes & get xyz ranges.
int CWOSMHub::GetControllerVersion(int& version)
{

    int ret = DEVICE_OK;

    LogMessage("Login to WOSM", false);

    // Flush the I/O buffer
    PurgeComPort(port_.c_str());
 
    ret = SendSerialCommand(port_.c_str(), " ", "\r\n");
    if ( ret != DEVICE_OK ) return ret;
    LogMessage("SENT space", false);

    // wait for Password request....
    string answer;
    ret = GetSerialAnswer(port_.c_str(), "admin password:", answer);
    if ( ret != DEVICE_OK ) return ERR_WOSM_NOT_FOUND;
;
    ret = SendSerialCommand(port_.c_str(), "wosm", "\r\n");
    ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    if ( ret != DEVICE_OK ) return ERR_WOSM_NOT_FOUND; 

    ret = SendSerialCommand(port_.c_str(), "kyp_lcd_screen t = 5s \"\n  MicroManager\n   connected!\"", "\r\n");
    ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    if ( ret != DEVICE_OK ) return ret;
 
    //completed initialisation okay...from now on.. don't bother with comms error checking!
    version = 99;

    // Below is what happens when you steal code from differnt places and can't be bothered to
    // have a unified approach!
  
    // XYZ stage setup set DAC mode to voltage output (mode 2) (0-10V)
    // Obtain the Stage range (should be 16 bits = 65535)
    double travelX, travelY, travelZ;
    ret = SetAxisMode(0);
    ret = GetAxisInfo(0, travelX);

    ret = SetAxisMode(1);
    ret = GetAxisInfo(1, travelY);

    ret = SetAxisMode(2);
    ret = GetAxisInfo(2, travelZ);

    if ( ( travelX > 0 ) && ( travelY > 0 ) )  hasXYStage_ = true;
    if ( travelZ > 0 ) hasZStage_ = true;

    // now initialise the LED DAC lines (ps->pv) to constant current mode (mode 3)
    ret = SendSerialCommand(port_.c_str(), "dac_mode ps 3", "\r\n"); ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    ret = SendSerialCommand(port_.c_str(), "dac_mode pt 3", "\r\n"); ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    ret = SendSerialCommand(port_.c_str(), "dac_mode pu 3", "\r\n"); ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    ret = SendSerialCommand(port_.c_str(), "dac_mode pv 3", "\r\n"); ret = GetSerialAnswer(port_.c_str(), "W>", answer);

    // now initialise the LED DIG lines (s->v) for gating (mode 12)
    ret = SendSerialCommand(port_.c_str(), "dig_mode s 12", "\r\n"); ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    ret = SendSerialCommand(port_.c_str(), "dig_mode t 12", "\r\n"); ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    ret = SendSerialCommand(port_.c_str(), "dig_mode u 12", "\r\n"); ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    ret = SendSerialCommand(port_.c_str(), "dig_mode v 12", "\r\n"); ret = GetSerialAnswer(port_.c_str(), "W>", answer);
 
    return ret;
}

// Command: "dac_max" obtain stageRange for X, Y and Z axes
int CWOSMHub::GetAxisInfo(int axis, double& travel)
{
    if ( axis < 0 || axis > 2 ) return DEVICE_ERR;

    std::stringstream cmd;
    cmd.clear();

    switch ( axis ) {
    case 0:
        cmd << "dac_max px"; // X-axis
        break;
    case 1:
        cmd << "dac_max py"; // Y-axis
        break;
    case 2:
        cmd << "dac_max pz"; // Z-axis
        break;
    }

    // Flush the I/O buffer then send command
    PurgeComPort(port_.c_str());
    int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r\n");
    if ( ret != DEVICE_OK ) return ret;

    std::string answer;
    ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    if ( ret != DEVICE_OK ) return ERR_UNKNOWN_AXIS;

    std::stringstream ss(answer);
    std::string trav;
    getline(ss, trav, '\r');

    std::stringstream sstravel(trav);
    sstravel >> travel;

    travel = travel / 65535 * 100; // all rather pointless! we know its 16 bits and 100um! 

    return DEVICE_OK;
}

// Command: "dac_mode" make X, Y and Z axes all voltage out
int CWOSMHub::SetAxisMode(int axis)
{
    if ( axis < 0 || axis > 2 ) return DEVICE_ERR;

    std::stringstream cmd;
    cmd.clear();

    switch ( axis ) {
    case 0:
        cmd << "dac_mode px 2"; // X-axis
        break;
    case 1:
        cmd << "dac_mode py 2"; // Y-axis
        break;
    case 2:
        cmd << "dac_mode pz 2"; // Z-axis
        break;
    }

    // Flush the I/O buffer then send command
    PurgeComPort(port_.c_str());
    int ret = SendSerialCommand(port_.c_str(), cmd.str().c_str(), "\r\n");
    if ( ret != DEVICE_OK ) return ret;

    std::string answer;
    ret = GetSerialAnswer(port_.c_str(), "W>", answer);
    if ( ret != DEVICE_OK ) return ERR_UNKNOWN_AXIS;

    return DEVICE_OK;
}
bool CWOSMHub::SupportsDeviceDetection(void)
{
    return true;
}

MM::DeviceDetectionStatus CWOSMHub::DetectDevice(void)
{
    if ( initialized_ )
        return MM::CanCommunicate;

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

            MM::Device* pS = GetCoreCallback()->GetDevice(this, port_.c_str());

            pS->Initialize();

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

int CWOSMHub::Initialize()
{
    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameWOSMHub, MM::String, true);
    if ( DEVICE_OK != ret )
        return ret;

    MMThreadGuard myLock(lock_);

    // Check that we have a controller:
    PurgeComPort(port_.c_str());
    ret = GetControllerVersion(version_);
    if ( DEVICE_OK != ret )
        return ret;

    if ( version_ < g_Min_MMVersion )
        return ERR_VERSION_MISMATCH;

    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMHub::OnVersion);
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

int CWOSMHub::DetectInstalledDevices()
{
    if ( MM::CanCommunicate == DetectDevice() )
    {
        std::vector<std::string> peripherals;
        peripherals.clear();
        peripherals.push_back(g_DeviceNameWOSMSwitch);
        peripherals.push_back(g_DeviceNameWOSMShutter);
        peripherals.push_back(g_DeviceNameWOSMDAC0);
        peripherals.push_back(g_DeviceNameWOSMDAC1);
        peripherals.push_back(g_DeviceNameWOSMDAC2);
        peripherals.push_back(g_DeviceNameWOSMDAC3);
        peripherals.push_back(g_DeviceNameWOSMDAC4);
        peripherals.push_back(g_DeviceNameStage);
        peripherals.push_back(g_DeviceNameXYStage);
        peripherals.push_back(g_DeviceNameWOSMInput);

        for ( size_t i = 0; i < peripherals.size(); i++ )
        {
            MM::Device* pDev = ::CreateDevice(peripherals[i].c_str());
            if ( pDev )
            {
                AddInstalledDevice(pDev);
            }
        }
    }

    return DEVICE_OK;
}

int CWOSMHub::Shutdown()
{
    initialized_ = false;
    return DEVICE_OK;
}

int CWOSMHub::OnPort(MM::PropertyBase* pProp, MM::ActionType pAct)
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

int CWOSMHub::OnVersion(MM::PropertyBase* pProp, MM::ActionType pAct)
{
    if ( pAct == MM::BeforeGet )
    {
        pProp->Set(( long ) version_);
    }
    return DEVICE_OK;
}

int CWOSMHub::OnLogic(MM::PropertyBase* pProp, MM::ActionType pAct)
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

/* CWOSMSwitch implementation:
 Justin's Notes:
 The "Switch Device" is registered with MM as a "State Device"
 I don't know the full command set used by the config file it seems to
 be documented by "example".

 Different switch states are attached to different "channels" -
 That is done here, again in the config file, also in the WOSM firmware
 and finally the physical wiring of the GPIO pins.
*/

// CWOSMSwitch implementation
CWOSMSwitch::CWOSMSwitch() :
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
    SetErrorText(ERR_UNKNOWN_POSITION, "Invalid position (state) specified");
    SetErrorText(ERR_INITIALIZE_FAILED, "Initialization of the device failed");
    SetErrorText(ERR_WRITE_FAILED, "Failed to write data to the device");
    SetErrorText(ERR_CLOSE_FAILED, "Failed closing the device");
    SetErrorText(ERR_COMMUNICATION, "Error in communication with WOSM board");
    SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The WOSM Hub device is needed to create this device");

    for ( unsigned int i = 0; i < NUMPATTERNS; i++ )
        pattern_[i] = 0;

    // Description
    int ret = CreateProperty(MM::g_Keyword_Description, "WOSM digital output driver", MM::String, true);
    assert(DEVICE_OK == ret);

    // Name
    ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameWOSMSwitch, MM::String, true);
    assert(DEVICE_OK == ret);

    // parent ID display
    CreateHubIDProperty();
}

CWOSMSwitch::~CWOSMSwitch()
{
    Shutdown();
}

void CWOSMSwitch::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameWOSMSwitch);
}

int CWOSMSwitch::Initialize()
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() ) {
        return ERR_NO_PORT_SET;
    }
    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    // set property list
    // -----------------

    // In this version of the adpater we will control up to 8 switchable devices 
    // Using WOSM dig_out lines s,t,u,v,w,x,y,z (in this adapter we use x,y,z for stage control)
    // .. so we might "blank" those lines (x,y,z) to prevent accidentally switching to zero! 
    // Anyway....  We will define up to 256 char labels for 256 options !
    // 0="0", 2="2"...... 255="255"
    // you will see later "WriteToPort" method that we need to do some bit shifting to get the 
    // correct control-byte pattern (since in principle we have up to 26 control bits!! 
    // The ASCII labels are just the decimal numbers representing each bit-mapped switch pattern
    // (see below for explanation)

    //Create the text labels that we will use in the config file 
    const int bufSize = 4;
    char buf[bufSize];
    for ( long i = 0; i < 256; i++ ) {
        snprintf(( char* ) buf, bufSize, "%d", ( unsigned ) i);
        SetPositionLabel(i, buf);
    }

    /* E.g. if you have different LED light sources powered by the different WOSM O/P pins
      B,  G,  Y,  R   < colour of the LED
      3,  2,  1,  0   < bit position
      8,  4,  2,  1   < decimal number when set to "1" at that bit position
      1,  0,  0,  0,   = "Blue only" requires this bit pattern = 16 Decimal
    And if you want RGB simultaneously you send:
      1,  1,  0,  1    = 8+4+1 = 13 Decimal
    To make this easier you list the options you want available in the Config file like this:

*** config file:
Label,WOSM-Switch,0,All_OFF
Label,WOSM-Switch,1,Red
Label,WOSM-Switch,2,Yellow
Label,WOSM-Switch,4,Red
Label,WOSM-Switch,8,Blue
Label,WOSM-Switch,13,RedGreenBlue
ConfigGroup,Channel,Red_LED,WOSM-Switch,Label,Red
ConfigGroup,Channel,Yellow_LED,WOSM-Switch,Label,Yellow
ConfigGroup,Channel,Green_LED,WOSM-Switch,Label,Green
ConfigGroup,Channel,Blue_LED,WOSM-Switch,Label,Blue
ConfigGroup,Channel,Red_Yell_UV,WOSM-Switch,Label,RedGreenBlue
***

    */

    // State
    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMSwitch::OnState);
    int nRet = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if ( nRet != DEVICE_OK )
        return nRet;
    SetPropertyLimits(MM::g_Keyword_State, 0, 256 - 1);

    // Label
    pAct = new CPropertyAction(this, &CStateBase::OnLabel);
    nRet = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if ( nRet != DEVICE_OK )
        return nRet;

    pAct = new CPropertyAction(this, &CWOSMSwitch::OnSequence);
    nRet = CreateProperty("Sequence", g_On, MM::String, false, pAct);
    if ( nRet != DEVICE_OK )
        return nRet;
    AddAllowedValue("Sequence", g_On);
    AddAllowedValue("Sequence", g_Off);

    // Starts "blanking" mode: goal is to synchronize laser light with camera exposure
    std::string blankMode = "Blanking Mode";
    pAct = new CPropertyAction(this, &CWOSMSwitch::OnBlanking);
    nRet = CreateProperty(blankMode.c_str(), "Idle", MM::String, false, pAct);
    if ( nRet != DEVICE_OK ) return nRet;

    AddAllowedValue(blankMode.c_str(), g_On);
    AddAllowedValue(blankMode.c_str(), g_Off);

    // Blank on TTL high or low
    pAct = new CPropertyAction(this, &CWOSMSwitch::OnBlankingTriggerDirection);
    nRet = CreateProperty("Blank On", "Low", MM::String, false, pAct);
    if ( nRet != DEVICE_OK )
        return nRet;
    AddAllowedValue("Blank On", "Low");
    AddAllowedValue("Blank On", "High");

    /*
    // Some original comments:
    // but SADLY, the code itself is commented out
    // In fact, looks like a useful thing to include...To Do.
    // ////////////////////////////////////////////////////////////////
    // Starts producing timed digital output patterns
    // Parameters that influence the pattern are 'Repeat Timed Pattern', 'Delay', 'State'
    // where the latter two are manipulated with the Get and SetPattern functions

    std::string timedOutput = "Timed Output Mode";
    pAct = new CPropertyAction(this, &CWOSMSwitch::OnStartTimedOutput);
    nRet = CreateProperty(timedOutput.c_str(), "Idle", MM::String, false, pAct);
    if (nRet != DEVICE_OK)
       return nRet;
    AddAllowedValue(timedOutput.c_str(), "Stop");
    AddAllowedValue(timedOutput.c_str(), "Start");
    AddAllowedValue(timedOutput.c_str(), "Running");
    AddAllowedValue(timedOutput.c_str(), "Idle");

    // Sets a delay (in ms) to be used in timed output mode
    // This delay will be transferred to the WOSM using the Get and SetPattern commands
    pAct = new CPropertyAction(this, &CWOSMSwitch::OnDelay);
    nRet = CreateProperty("Delay (ms)", "0", MM::Integer, false, pAct);
    if (nRet != DEVICE_OK)
       return nRet;
    SetPropertyLimits("Delay (ms)", 0, 65535);

    // Repeat the timed Pattern this many times:
    pAct = new CPropertyAction(this, &CWOSMSwitch::OnRepeatTimedPattern);
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

int CWOSMSwitch::Shutdown()
{
    initialized_ = false;
    return DEVICE_OK;
}
// I am assuming when this function (method) is called the passed parameter "value"
// is the bitmap pattern that is to be written to the switches (if bit is 0 the output is turned off
// (electronically gated) if the bit is 1 then the output goes to it's preset value.

// command: 1 now "dig_out" Set current digital output pattern - The "CWOSMShutter" NO LONGER uses this command!
int CWOSMSwitch::WriteToPort(long value)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;
    
    //                          Masking pattern for the WOSM dig_out lines
    //  -----------  H I G H    W O R D  -----------       -----------  L O W      W O R D  -----------
    // 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00 :: 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00
    //  *  *  *  *  *  *  z  y  x  w  v  u  t  s  r  q ::  p  o  n  m  l  k  j  i  h  g  f  e  d  c  b  a 
    //                    ^  ^  ^  ^  ^  ^  ^  ^ 
    //                     useful control lines
    //                     ...... Mask bits .....
    //                    0  0  0  1  1  0  1  0  = 26 Decimal (= turn on lines w,v & t, all others off!)
    // Keep the low byte and shift up to the High Word position as shown above
    // Note: to make double sure we don't interfere with other lines we mask our mask with 03FC::0000 
    // (i.e. only modify digital lines 's->z')

    value = 255 & value;
    value = value << 18;

    if ( hub->IsLogicInverted() ) value = ~value;

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    char command[50];
    unsigned int leng;

    leng = snprintf(( char* ) command, 50, "dig_out %d 0x03FC0000\r\n", value);
    ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
    if ( ret != DEVICE_OK )  return ret;

    // Purge the IO buffer in case WOSM has sent a response!
    hub->PurgeComPortH();
    hub->SetTimedOutput(false);

    std::ostringstream os;
    os << "Switch::WriteToPort Command= " << command;
    LogMessage(os.str().c_str(), false);

    // wait for usual response from WOSM ("W>")

    MM::MMTime startTime = GetCurrentMMTime();
    unsigned long bytesRead = 0;
    unsigned char answer[50];
    while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br = 0;
        ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
        if ( answer[bytesRead] == '>' )  break;
        bytesRead += br;
    }

    answer[bytesRead] = 0; // string terminator

    hub->PurgeComPortH();
    if ( ret != DEVICE_OK ) return ret;

    //if ( answer[0] != 'W' )  return ERR_COMMUNICATION;

    return DEVICE_OK;
}

// Commands: 5 & 6 now "P" and "N" = store new "P"atterns and "N"umber of patterns
int CWOSMSwitch::LoadSequence(unsigned size, unsigned char* seq)
{

    std::ostringstream os;
    os << "Switch::LoadSequence size= " << size << " Seq= " << seq;
    LogMessage(os.str().c_str(), false);

    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
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

    // Purge the IO buffer in case WOSM has sent a response!
    hub->SetTimedOutput(false);
    hub->PurgeComPortH();

    return DEVICE_OK;
}

// Action handlers

// Commands: 8 & 9 now "R" and "E" Run and End Trigger mode resp.
int CWOSMSwitch::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    std::ostringstream os;
    os << "Switch::OnState_";
    LogMessage(os.str().c_str(), false);

    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    // Some comments here would be helpful
    if ( eAct == MM::BeforeGet )
    {
        // nothing to do, let the caller use cached property  ???
    }
    else if ( eAct == MM::AfterSet )
    {
        //  **** a comment here is essential !.. 
        // where are we getting the "pos" value from?
        long pos;
        pProp->Get(pos);

        // this is obscure - I can guess what it does.. but, I'm not going to say
        hub->SetSwitchState(pos);

        if ( hub->GetShutterState() > 0 )    // I don't like this because of confusion with "shutterDevice"
            os << "_Pos= " << pos << " WriteToPort(pos)?";
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
        unsigned char answer[50];

        // Now chug through the I/O buffer, byte-by-byte and stop at cr or lf.... yuk!
        while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
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
int CWOSMSwitch::OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
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
int CWOSMSwitch::OnStartTimedOutput(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    std::ostringstream os;
    os << "Switch::OnStartTimedOutput ";
    LogMessage(os.str().c_str(), false);

    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    if ( eAct == MM::BeforeGet ) {
        if ( hub->IsTimedOutputActive() ) pProp->Set("Running");
        else                            pProp->Set("Idle");
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
int CWOSMSwitch::OnBlanking(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::ostringstream os;
    os << "Switch::OnBlanking ";
    LogMessage(os.str().c_str(), false);

    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );

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
int CWOSMSwitch::OnBlankingTriggerDirection(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    std::ostringstream os;
    os << "Switch::OnBlankingTriggerDirection";
    LogMessage(os.str().c_str(), false);

    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );

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
int CWOSMSwitch::OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
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
int CWOSMSwitch::OnRepeatTimedPattern(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::ostringstream os;
    os << "Switch::OnRepeatTimedPattern ";
    LogMessage(os.str().c_str(), false);

    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
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

// CWOSMShutter implementation
// Justin Notes: I'm a bit mystified by the "shutter" and when it is called. Currently it issues a "Switch" 
// command - I find that confusing.

// CWOSMShutter implementation
CWOSMShutter::CWOSMShutter() : initialized_(false), name_(g_DeviceNameWOSMShutter)
{
    InitializeDefaultErrorMessages();
    EnableDelay();

    SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The WOSM Hub device is needed to create this device");

    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameWOSMShutter, MM::String, true);
    assert(DEVICE_OK == ret);

    // Description
    ret = CreateProperty(MM::g_Keyword_Description, "WOSM shutter driver", MM::String, true);
    assert(DEVICE_OK == ret);

    // parent ID display
    CreateHubIDProperty();
}
CWOSMShutter::~CWOSMShutter()
{
    Shutdown();
}
void CWOSMShutter::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameWOSMShutter);
}
bool CWOSMShutter::Busy()
{
    MM::MMTime interval = GetCurrentMMTime() - changedTime_;

    return interval < MM::MMTime::fromMs(GetDelayMs());
}
int CWOSMShutter::Initialize()
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
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
    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMShutter::OnOnOff);
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
int CWOSMShutter::Shutdown()
{
    if ( initialized_ )
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}
int CWOSMShutter::SetOpen(bool open)
{
    std::ostringstream os;
    os << "Shutter::SetOpen open= " << open;
    LogMessage(os.str().c_str(), false);

    if ( open ) return SetProperty("OnOff", "1");
    else      return SetProperty("OnOff", "0");
}
int CWOSMShutter::GetOpen(bool& open)
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
int CWOSMShutter::Fire(double /*deltaT*/)
{
    return DEVICE_UNSUPPORTED_COMMAND;
}

// e.g. "x0101100" turns off channels 0,1,5,7.. 
// Note: channels 2,3,6 will switch on at their preset output level.. "shutter" is messing with the switches!
// must try to separate church and state.

// Command: "dig_out r 0/1" Shutter open or closed on channel "r"       
int CWOSMShutter::WriteToPort(long value)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    MMThreadGuard myLock(hub->GetLock());

    // Just keep the lowest bit -  a single shutter it's either on or off (open or closed)
    value = 255 & value;
    value = value << 18;

    if ( hub->IsLogicInverted() )  value = ~value;

    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    // create command buffer
    char command[50];
    unsigned int leng;

        leng = snprintf(( char* ) command, 50, "dig_out %d 0x3FC0000\r\n", value);
        ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
        if ( ret != DEVICE_OK )  return ret;

        MM::MMTime startTime = GetCurrentMMTime();
        unsigned long bytesRead = 0;
        unsigned char answer[50];
        while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
            unsigned long br = 0;
            ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
            if ( answer[bytesRead] == '>' )  break;
            bytesRead += br;
        }
        answer[bytesRead] = 0; // string terminator

        hub->PurgeComPortH();
        if ( ret != DEVICE_OK ) return ret;

        //if ( answer[bytesRead-1] != 'W' )  return ERR_COMMUNICATION;

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

// Action handlers - don't seem to work correctly at present. I've modded it so, it now fails!
int CWOSMShutter::OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
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
            ret = WriteToPort(0); // Shutter closed (write zeros to all LEDs)
        else
            ret = WriteToPort(hub->GetSwitchState()); // restore old setting

        if ( ret != DEVICE_OK )  return ret;

        hub->SetShutterState(pos);
        changedTime_ = GetCurrentMMTime();
    }

    std::ostringstream os;
    os << "Shutter::OnOnOff hub->GetSwitchState()= " << hub->GetSwitchState();
    LogMessage(os.str().c_str(), false);

    return DEVICE_OK;
}

// CWOSMDA implementation:
// Justin's Notes:
// This "Device" only sends one command "O,chan,volts"
// Summary:
// => OnVolts Action Handler is called by an event handler in MM.
// It subsequently calls:
//      => SetSignal which optionally "gates" the O/P to "zero Volts"
//      => WriteSignal scales the value to 12 bits (but now does nothing!)
//      => WritetoPort - Sends "O,(int) chan,(float) volts" to WOSM
// => OnMaxVolts & => OnChannel just keep Volts and Channels within bounds
// 
// If we need anything fast or device-specific do it on the WOSM. 
// USB transmit/receive latency will dominate.

// CWOSMDA implementation
CWOSMDA::CWOSMDA(int channel) :
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
    SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The WOSM Hub device is needed to create this device");

    /* Channel property is not needed
    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMDA::OnChannel);
    CreateProperty("Channel", channel_ == 1 ? "1" : "2", MM::Integer, false, pAct, true);
    for (int i=1; i<= 2; i++){
       std::ostringstream os;
       os << i;
       AddAllowedValue("Channel", os.str().c_str());
    }
    */

    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMDA::OnMaxVolt);
    CreateProperty("Power %", "100", MM::Float, false, pAct, true);

    if ( channel_ == 0 )
    {
        name_ = g_DeviceNameWOSMDAC0;
    }
    else if ( channel_ == 1 )
    {
        name_ = g_DeviceNameWOSMDAC1;
    }
    else if ( channel_ == 2 )
    {
        name_ = g_DeviceNameWOSMDAC2;
    }
    else if ( channel_ == 3 )
    {
        name_ = g_DeviceNameWOSMDAC3;
    }
    else if ( channel_ == 4 )
    {
        name_ = g_DeviceNameWOSMDAC4;
    }

    //name_ = channel_ == 1 ? g_DeviceNameWOSMDA1 : g_DeviceNameWOSMDA2;

    // Description
    int nRet = CreateProperty(MM::g_Keyword_Description, "WOSM DAC driver", MM::String, true);
    assert(DEVICE_OK == nRet);

    // Name
    nRet = CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);
    assert(DEVICE_OK == nRet);

    // parent ID display
    CreateHubIDProperty();
}
CWOSMDA::~CWOSMDA()
{
    Shutdown();
}
void CWOSMDA::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, name_.c_str());
}
int CWOSMDA::Initialize()
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() ) {
        return ERR_NO_PORT_SET;
    }
    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    // set property list
    // -----------------

    // State
    // -----
    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMDA::OnVolts);
    int nRet = CreateProperty("Volts", "0.0", MM::Float, false, pAct);
    if ( nRet != DEVICE_OK )
        return nRet;
    SetPropertyLimits("Volts", minV_, maxV_);

    nRet = UpdateStatus();

    if ( nRet != DEVICE_OK ) return nRet;

    initialized_ = true;

    return DEVICE_OK;
}
int CWOSMDA::Shutdown()
{
    initialized_ = false;
    return DEVICE_OK;
}

// Command: "dac_out ps 24.5"
int CWOSMDA::WriteToPort(double value)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )
        return ERR_NO_PORT_SET;

    LogMessage("Into write dac_out", false);

    MMThreadGuard myLock(hub->GetLock());

    hub->PurgeComPortH();
    int ret = DEVICE_OK;
    char command[50];
    unsigned int leng;
    char chan[2];
    if ( ( channel_ >= 0 ) && ( channel_ < 4 ) ) {
        chan[0] = 's' + static_cast<char>(channel_) ; chan[1] = 0;     // set s,t,u or v lines on WOSM

        leng = snprintf(( char* ) command, 50, "dac_out p%s %3.3f\r\n", ( char* ) chan, value);

        std::stringstream ss;
        ss << "Sending...Command: " << command ;
        LogMessage(ss.str().c_str(), false);

        ret = hub->WriteToComPortH(( const unsigned char* ) command, leng);
        if ( ret != DEVICE_OK )  return ret;

        MM::MMTime startTime = GetCurrentMMTime();
        unsigned long bytesRead = 0;
        unsigned char answer[50];
        while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
            unsigned long br=0;
            ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
            if ( answer[bytesRead] == '>' )  break;
            bytesRead += br;
        }

        answer[bytesRead] = 0; // string terminator

        hub->PurgeComPortH();
        if ( ret != DEVICE_OK ) return ret;

        //if ( answer[0] != 'W' )  return ERR_COMMUNICATION;
    }
    hub->SetTimedOutput(false);

    return DEVICE_OK;
}
int CWOSMDA::WriteSignal(double volts)
{
    double value = ( double ) volts;  // ( (volts - minV_) / maxV_ * 4095);

    std::ostringstream os;
    os << "Volts= " << volts << " MaxVoltage= " << maxV_ << " digitalValue= " << value;
    LogMessage(os.str().c_str(), false);

    return WriteToPort(value);
}
int CWOSMDA::SetSignal(double volts)
{
    volts_ = volts;
    if ( gateOpen_ ) {
        gatedVolts_ = volts_;
        return WriteSignal(volts_);
    }
    else {
        gatedVolts_ = 0;
    }

    return DEVICE_OK;
}
int CWOSMDA::SetGateOpen(bool open)
{
    if ( open ) {
        gateOpen_ = true;
        gatedVolts_ = volts_;
        return WriteSignal(volts_);
    }
    gateOpen_ = false;
    gatedVolts_ = 0;
    return WriteSignal(0.0);

}

// Action handlers
int CWOSMDA::OnVolts(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if ( eAct == MM::BeforeGet )
    {
        // nothing to do, let the caller use cached property
    }
    else if ( eAct == MM::AfterSet )
    {
        double volts;
        pProp->Get(volts);
        return SetSignal(volts);
    }

    return DEVICE_OK;
}
int CWOSMDA::OnMaxVolt(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if ( eAct == MM::BeforeGet )
    {
        pProp->Set(maxV_);
    }
    else if ( eAct == MM::AfterSet )
    {
        pProp->Get(maxV_);
        if ( HasProperty("Volts") )
            SetPropertyLimits("Volts", 0.0, maxV_);

    }
    return DEVICE_OK;
}
int CWOSMDA::OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if ( eAct == MM::BeforeGet )
    {
        pProp->Set(( long int ) channel_);
    }
    else if ( eAct == MM::AfterSet )
    {
        long channel;
        pProp->Get(channel);
        if ( channel >= 0 && ( ( unsigned ) channel <= maxChannel_ ) )
            channel_ = channel;
    }
    return DEVICE_OK;
}

/*
Below, I have folded-in (and modified) the PIEZOCONCEPT XYZ stage adapter
so we can control focus and move an x-y stage connected to our WOSM board (see firmware).
So, we have two new blocks of code:

CWOSMStage & CWOSMXYStage

Notes:
Stage and Focus noise:
On the WOSM side, the XYZ channels have 16-bit resolution

*/

// CWOSMStage Implementation
CWOSMStage::CWOSMStage() :
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

CWOSMStage::~CWOSMStage()
{
    Shutdown();
}

void CWOSMStage::GetName(char* Name) const
{
    CDeviceUtils::CopyLimitedString(Name, g_DeviceNameStage);
}

int CWOSMStage::Initialize()
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );

    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;

    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    if ( initialized_ ) return DEVICE_OK;

    // Name
    int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameStage, MM::String, true);
    if ( DEVICE_OK != ret ) return ret;

    // Description
    ret = CreateProperty(MM::g_Keyword_Description, "WOSM Z stage driver", MM::String, true);
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

int CWOSMStage::Shutdown()
{
    if ( initialized_ ) initialized_ = false;
    return DEVICE_OK;
}

int CWOSMStage::GetPositionUm(double& pos)
{
    pos = pos_um_;
    return DEVICE_OK;
}

int CWOSMStage::SetPositionUm(double pos)
{
    int ret = MoveZ(pos);
    if ( ret != DEVICE_OK ) return ret;
    return OnStagePositionChanged(pos);
}

int CWOSMStage::GetPositionSteps(long& steps)
{
    double posUm;
    int ret = GetPositionUm(posUm);
    if ( ret != DEVICE_OK ) return ret;

    steps = static_cast< long >( posUm / GetStepSize() );
    return DEVICE_OK;
}

int CWOSMStage::SetPositionSteps(long steps)
{
    return SetPositionUm(steps * GetStepSize());
}

// "Z" command - move to new Z-position (Focus up/down)
int CWOSMStage::MoveZ(double pos)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );

    LogMessage("Into moveZ" , false);

    if ( !hub || !hub->IsPortAvailable() ) return ERR_HUB_UNAVAILABLE;

    if ( pos > upperLimit_ ) pos = upperLimit_;
    if ( pos < lowerLimit_ ) pos = lowerLimit_;

    LogMessage("after limits", false);

    char buf[50];
    int length = sprintf(buf, "dac_out pz %3.3f\r\n", pos);

    std::stringstream ss;
    ss << "Command: " << buf << "  Position set: " << pos;
    LogMessage(ss.str().c_str(), false);

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();

    int ret = hub->WriteToComPortH(( unsigned char* ) buf, length);
    if ( ret != DEVICE_OK ) return ret;

    MM::MMTime startTime = GetCurrentMMTime();
    unsigned long bytesRead = 0;
    unsigned char answer[50];
    while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br;
        ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
        if ( answer[bytesRead] == '>' )  break;
        bytesRead += br;
    }

    answer[bytesRead] = 0; // string terminator

    hub->PurgeComPortH();
    if ( ret != DEVICE_OK ) return ret;

    //if ( answer[0] != 'W' )  return ERR_COMMUNICATION;

    hub->SetTimedOutput(false);

    std::ostringstream os;
    os << "MoveZ Z," << pos;
    LogMessage(os.str().c_str(), false);

    pos_um_ = pos;

    return DEVICE_OK;
}

bool CWOSMStage::Busy()
{
    return false;
}

CWOSMXYStage::CWOSMXYStage() : CXYStageBase<CWOSMXYStage>(),
stepSize_X_um_(0.1),
stepSize_Y_um_(0.1),
posX_um_(0.0),
posY_um_(0.0),
busy_(false),
initialized_(false),
lowerLimitX_(0.0),
upperLimitX_(100.0),
lowerLimitY_(0.0),
upperLimitY_(100.0)
{
    InitializeDefaultErrorMessages();

    // parent ID display
    CreateHubIDProperty();

    // step size
    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMXYStage::OnXStageMinPos);
    CreateProperty(g_PropertyXMinUm, "0", MM::Float, false, pAct, true);

    pAct = new CPropertyAction(this, &CWOSMXYStage::OnXStageMaxPos);
    CreateProperty(g_PropertyXMaxUm, "100", MM::Float, false, pAct, true);

    pAct = new CPropertyAction(this, &CWOSMXYStage::OnYStageMinPos);
    CreateProperty(g_PropertyYMinUm, "0", MM::Float, false, pAct, true);

    pAct = new CPropertyAction(this, &CWOSMXYStage::OnYStageMaxPos);
    CreateProperty(g_PropertyYMaxUm, "100", MM::Float, false, pAct, true);
}

CWOSMXYStage::~CWOSMXYStage()
{
    Shutdown();
}

void CWOSMXYStage::GetName(char* Name) const
{
    CDeviceUtils::CopyLimitedString(Name, g_DeviceNameXYStage);
}

int CWOSMXYStage::Initialize()
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
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

int CWOSMXYStage::Shutdown()
{
    if ( initialized_ ) initialized_ = false;
    return DEVICE_OK;
}

bool CWOSMXYStage::Busy()
{
    return false;
}

int CWOSMXYStage::GetPositionSteps(long& x, long& y)
{
    x = ( long ) ( posX_um_ / stepSize_X_um_ );
    y = ( long ) ( posY_um_ / stepSize_Y_um_ );

    std::stringstream ss;
    ss << "GetPositionSteps :=" << x << "," << y;
    LogMessage(ss.str(), false);
    return DEVICE_OK;
}

int CWOSMXYStage::SetPositionSteps(long x, long y)
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

int CWOSMXYStage::SetRelativePositionSteps(long x, long y)
{
    long curX, curY;
    GetPositionSteps(curX, curY);

    return SetPositionSteps(curX + x, curY + y);
}

// "X" command - move to new x-position (stage translate)
int CWOSMXYStage::MoveX(double posUm)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() ) return ERR_HUB_UNAVAILABLE;

    if ( posUm < lowerLimitX_ ) posUm = lowerLimitX_;
    if ( posUm > upperLimitX_ ) posUm = upperLimitX_;

    char buf[25];
    int length = sprintf(buf, "dac_out px %1.3f\r\n", posUm);

    std::stringstream ss;
    ss << "Command: " << buf << "  Position set: " << posUm;
    LogMessage(ss.str().c_str(), false);

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();

    int ret = hub->WriteToComPortH(( unsigned char* ) buf, length);
    if ( ret != DEVICE_OK ) return ret;

    MM::MMTime startTime = GetCurrentMMTime();
    unsigned long bytesRead = 0;
    unsigned char answer[50];
    while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br;
        ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
        if ( answer[bytesRead] == '>' )  break;
        bytesRead += br;
    }

    answer[bytesRead] = 0; // string terminator

    hub->PurgeComPortH();
    hub->SetTimedOutput(false);

    if ( ret != DEVICE_OK ) return ret;

    //if ( answer[0] != 'W' )  return ERR_COMMUNICATION;

    std::ostringstream os;
    os << "X-Answer= " << answer;
    LogMessage(os.str().c_str(), false);

    posX_um_ = posUm;
    return DEVICE_OK;
}

// "Y" command - move to new y-position (stage translate)
int CWOSMXYStage::MoveY(double posUm)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );

    if ( !hub || !hub->IsPortAvailable() ) return ERR_HUB_UNAVAILABLE;

    if ( posUm < lowerLimitY_ ) posUm = lowerLimitY_;
    if ( posUm > upperLimitY_ ) posUm = upperLimitY_;

    char buf[25];
    int length = sprintf(buf, "dac_out py %1.3f\r\n", posUm);

    std::stringstream ss;
    ss << "Command: " << buf << "  Position set: " << posUm;
    LogMessage(ss.str().c_str(), true);

    MMThreadGuard myLock(hub->GetLock());
    hub->PurgeComPortH();

    int ret = hub->WriteToComPortH(( unsigned char* ) buf, length);
    if ( ret != DEVICE_OK ) return ret;

    MM::MMTime startTime = GetCurrentMMTime();
    unsigned long bytesRead = 0;
    unsigned char answer[50];
    while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br;
        ret = hub->ReadFromComPortH(( unsigned char* ) answer + bytesRead, 1, br);
        if ( answer[bytesRead] == '>' )  break;
        bytesRead += br;
    }

    answer[bytesRead] = 0; // string terminator

    hub->PurgeComPortH();
    hub->SetTimedOutput(false);

    if ( ret != DEVICE_OK ) return ret;

    //if ( answer[0] != 'W' )  return ERR_COMMUNICATION;

    std::ostringstream os;
    os << "Y-Answer= " << answer;
    LogMessage(os.str().c_str(), false);

    posY_um_ = posUm;
    return DEVICE_OK;
}

int CWOSMXYStage::OnXStageMinPos(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CWOSMXYStage::OnXStageMaxPos(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CWOSMXYStage::OnYStageMinPos(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CWOSMXYStage::OnYStageMaxPos(MM::PropertyBase* pProp, MM::ActionType eAct)
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
CWOSMInput::CWOSMInput() :
    mThread_(0),
    pin_(0),
    name_(g_DeviceNameWOSMInput)
{
    std::string errorText = "To use the Input function you need firmware version 5 or higher";
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
    ret = CreateProperty(MM::g_Keyword_Description, "WOSM shutter driver", MM::String, true);
    assert(DEVICE_OK == ret);

    // parent ID display
    CreateHubIDProperty();
}

CWOSMInput::~CWOSMInput()
{
    Shutdown();
}

int CWOSMInput::Shutdown()
{
    if ( initialized_ )
        delete( mThread_ );
    initialized_ = false;
    return DEVICE_OK;
}

int CWOSMInput::Initialize()
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
    if ( !hub || !hub->IsPortAvailable() )  return ERR_NO_PORT_SET;

    char hubLabel[MM::MaxStrLength];
    hub->GetLabel(hubLabel);
    SetParentID(hubLabel); // for backward comp.

    char ver[MM::MaxStrLength] = "0";
    hub->GetProperty(g_versionProp, ver);

    // we should get ASCII "5" back from the WOSM - convert to (int)
    int version = atoi(ver);
    if ( version < g_Min_MMVersion )  return ERR_VERSION_MISMATCH;

    // I think the idea is to request setup of the INPUT_PULLUP state 
    // on the GPIO pins on the microcontroller... but it is not clear

    int ret = GetProperty("Pin", pins_);
    if ( ret != DEVICE_OK ) return ret;

    // pin_ = ascii to integer of pins_ (i.e. 0,1,2,3,4,5) unknown value if pins_="All"
    if ( strcmp("All", pins_) != 0 )  pin_ = atoi(pins_);

    ret = GetProperty("Pull-Up-Resistor", pullUp_);
    if ( ret != DEVICE_OK ) return ret;

    // Digital Input
    CPropertyAction* pAct = new CPropertyAction(this, &CWOSMInput::OnDigitalInput);
    ret = CreateProperty("DigitalInput", "0", MM::Integer, true, pAct);
    if ( ret != DEVICE_OK ) return ret;

    int start = 0;
    int end = 5;
    if ( strcmp("All", pins_) != 0 ) {
        start = pin_;
        end = pin_;
    }

    for ( long i = start; i <= end; i++ )
    {
        CPropertyActionEx* pExAct = new CPropertyActionEx(this, &CWOSMInput::OnAnalogInput, i);
        std::ostringstream os;
        os << "AnalogInput= " << i;

        ret = CreateProperty(os.str().c_str(), "0.0", MM::Float, true, pExAct);
        if ( ret != DEVICE_OK ) return ret;

        // set pull up resistor state for this pin
        if ( strcmp(g_On, pullUp_) == 0 ) SetPullUp(i, 1);
        else SetPullUp(i, 0);

    }

    mThread_ = new WOSMInputMonitorThread(*this);
    mThread_->Start();

    initialized_ = true;

    return DEVICE_OK;
}

void CWOSMInput::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

bool CWOSMInput::Busy()
{
    return false;
}

// Justin Notes: "GetDigitalInput" is polled every 0.5sec by the timed thread see below.
// In the original WOSM code we were testing the Analogue Inputs for a digital change..
// That's fine.. but we should set this up in a more obvious way.
// suggest we assign a some of the GPIO lines as Digital I/O and some as Analogue Input.
// Currently this is unclear to me...I don't know what Analogue inputs we want to monitor.

// Command: 40 now "L" logic in - returns "L,l\r\n" pin High/Low
int CWOSMInput::GetDigitalInput(long* state)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
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
    char answer[50];
    // Chug through the I/O buffer, byte-by-byte and stop at lf!
    while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 250 ) ) {
        unsigned long br=0;
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
    // sscanf(( const char* ) answer, "%1s,%d", com, &num);
    // EDITED HERE**************************************************
    num = 1;
    // should give com[0]='L' and num = 1 or 0 

    if ( com[0] != 'L' )  return ERR_COMMUNICATION;

    *state = ( long ) num;

    return DEVICE_OK;
}
int CWOSMInput::ReportStateChange(long newState)
{
    std::ostringstream os;
    os << newState;
    return OnPropertyChanged("DigitalInput", os.str().c_str());
}

// Action handlers
int CWOSMInput::OnDigitalInput(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    // This sets a property listed in the MMDevice Property.h file (very limited commenting)
    // "Set" is an "Action Functor" (handler?)  
    // "eACT" seems common to all "On" functions - so probably an Event listener?
    // Anyway, it wants a (long) value for "state" which becomes a virtual Boolean, apparently! 

    if ( eAct == MM::BeforeGet )
    {
        long state;

        // get the state of the selected pin 0-5 or all pins (which I have made to be "6" in my WOSM code)
        int ret = GetDigitalInput(&state);
        if ( ret != DEVICE_OK ) return ret;
        pProp->Set(state);
    }

    return DEVICE_OK;
}

// Commands: 41 now = "A,chan" analogue read channel
int CWOSMInput::OnAnalogInput(MM::PropertyBase* pProp, MM::ActionType eAct, long  channel)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
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
        char answer[50];
        MM::MMTime startTime = GetCurrentMMTime();
        while ( ( bytesRead < 50 ) && ( ( GetCurrentMMTime() - startTime ).getMsec() < 1000 ) ) {
            unsigned long br=0;
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
        //sscanf(( const char* ) answer, "%1s,%d", com, &num);
        // should give com[0]='A' and num = 597 or something!
        //EDITED HERE **********************************
        num = 1;
        std::ostringstream os;
        os << "AnswerToA= " << answer << " bytesRead= " << bytesRead << " com= " << com << " num= " << num;
        LogMessage(os.str().c_str(), false);

        //if ( com[0] != 'A' )  return ERR_COMMUNICATION;

        pProp->Set(( long ) num);
    }
    return DEVICE_OK;
}

// Commands: 42 - now "D" set digital pull-up ?? needs commenting - not sure why we want to do this dynamically.
int CWOSMInput::SetPullUp(int pin, int state)
{
    CWOSMHub* hub = static_cast< CWOSMHub* >( GetParentHub() );
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

WOSMInputMonitorThread::WOSMInputMonitorThread(CWOSMInput& aInput) :
    state_(0),
    aInput_(aInput)
{
}
WOSMInputMonitorThread::~WOSMInputMonitorThread()
{
    Stop();
    wait();
}

// I think this "free-running" thread is a 0.5s delay loop that polls the Digital I/O lines and reports changes
// It is perhaps not "thread safe" because the "L" and "A" commands seems to interfere... I haven't debugged this.
// It has probably been messed-up by my meddling!

int WOSMInputMonitorThread::svc()
{
    while ( !stop_ )
    {
        long state;
        int ret = aInput_.GetDigitalInput(&state);
        if ( ret != DEVICE_OK )
        {
            stop_ = true;
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
void WOSMInputMonitorThread::Start()
{
    stop_ = false;
    activate();
}
