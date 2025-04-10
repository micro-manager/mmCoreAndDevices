///////////////////////////////////////////////////////////////////////////////
// FILE:          WPI.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for WPI AL-XXX syringe pumps
//                
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes
//
// YEAR:          2025
//                
// VERSION:       1.0
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE   LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
//LAST UPDATE:    09.04.2025 LK

#include "WPI.h"

#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceThreads.h"
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include <string>
#include <map>
#include <algorithm>
#include <stdint.h>
#include <future>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
//  Global constants
///////////////////////////////////////////////////////////////////////////////

const char* g_WPIHubName = "WPI Pump Hub";
const char* g_WPIPumpName = "WPI Pump";
const char* NoHubError = "Parent Hub not defined.";
const char* CR = "\r";
const char* ETX = "\x03";
const double pi = 3.14159265359;

// All values are obtained from the data sheet https://www.wpi-europe.com/downloads/content/AL-4000_IM1.pdf
const double g_Diameter_min = 0.1;
const double g_Diameter_max = 50;
const double g_Speed_min = 0.00002299; // mm/s, converted from 0.008276531 cm/hr
const double g_Speed_max = 3.01339285; // mm/s, converted from 18.08035714 cm/min

enum PumpDirections {
    INFUSE,
    WITHDRAW
};

/**
* Convert the linear velocity of the syringe pump plus the diameter of the 
* syringe to the flowrate in uL/s.
* @param speed - linear velocity of the syringe pump in mm/s
* @param diameter - diameter of the syringe in mm.
*/
double calculate_flowrate(double speed, double diameter) {
    return 0.25 * speed * pi * diameter * diameter;
}

///////////////////////////////////////////////////////////////////////////////
//  MMDevice API
///////////////////////////////////////////////////////////////////////////////


MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_WPIHubName, MM::HubDevice, "Hub for WPI pumps.");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (!deviceName)
    {
        return 0; // Trying to create nothing, return nothing
    }
    else if (strcmp(deviceName, g_WPIHubName) == 0)
    {
        return new WPIPumpHub(); // Create Hub
    }
    // 8 is the length of the g_DemoPumpName name
    else if (strcmp(((string)deviceName).substr(0, 8).c_str(), g_WPIPumpName) == 0)
    {
        return new WPIPump(stoi(((string)deviceName).substr(8))); // Create pump
    }
    return 0; // If an unexpected name is provided, return nothing
}

MODULE_API void DeleteDevice(MM::Device* device)
{
    delete device;
}


///////////////////////////////////////////////////////////////////////////////
// WPIPumpHub class
// Hub for WPIPump devices
///////////////////////////////////////////////////////////////////////////////

WPIPumpHub::WPIPumpHub() :
    initialized_(false),
    busy_(false),
    nPumps_(0),
    port_("Undefined")
{
    // Assign COM-port
    CPropertyAction* pAct;
    pAct = new CPropertyAction(this, &WPIPumpHub::OnPort);
    CreateStringProperty("Port", port_.c_str(), false, pAct, true);

    // Assign number of pumps
    pAct = new CPropertyAction(this, &WPIPumpHub::OnNPumps);
    CreateIntegerProperty("Number of pumps", nPumps_, false, pAct, true);
}

WPIPumpHub::~WPIPumpHub() {
    Shutdown();
}

///////////////////////////////////////////////////////////////////////////////
// WPIPumpHub class
// MMDevice API
///////////////////////////////////////////////////////////////////////////////

int WPIPumpHub::Initialize()
{
    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, g_WPIHubName, true);
    if (DEVICE_OK != ret)
        return ret;

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Hub for WPI pumps.", true);
    if (DEVICE_OK != ret)
        return ret;

    // Verify connection
    ret = VerifyConnection(0);
    if (ret != DEVICE_OK) { return ret; }

    initialized_ = true;
    return DEVICE_OK;
}

int WPIPumpHub::Shutdown() {
    if (!initialized_) {
        return DEVICE_OK;
    }

    // Shut down all of the sub-pumps.
    // The pumps should handle their own safe shutdown.
    for (unsigned int i = 0; i < GetNumberOfInstalledDevices(); i++) {
        MM::Device* temp = GetInstalledDevice(i);
        if (temp->GetType() != MM::HubDevice) {
            temp->Shutdown();
        }
    }
    return DEVICE_OK;
}

bool WPIPumpHub::Busy() {
    return busy_;
}

void WPIPumpHub::GetName(char* name) const
{
    // Return the name used to refer to this device adapter
    CDeviceUtils::CopyLimitedString(name, g_WPIHubName);
}

///////////////////////////////////////////////////////////////////////////////
// WPIPumpHub class
// MMHub API
///////////////////////////////////////////////////////////////////////////////

/**
* In this function the iterator "i" is also used as the id of the device, i.e.
* this index can be included in the serial command (or API) to address that
* specific pump. The implementation should be adapter to match the serial/API
* requirements of your specific pump.
*/
int WPIPumpHub::DetectInstalledDevices()
{
    ClearInstalledDevices();

    // make sure this method is called before we look for available devices
    InitializeModuleData();

    char hubName[MM::MaxStrLength];
    GetName(hubName); // Name of the hub
    for (int i = 0; i < nPumps_; i++)
    {
        if (VerifyConnection(i) != DEVICE_OK) {
            LogMessage("Verification of pump " + to_string(i) + " unsuccessful");
            nPumps_ = i - 1;
            break;
        }
        LogMessage("Verification of pump " + to_string(i) + " worked!");
        string deviceName = (string)g_WPIPumpName + to_string(i);
        MM::Device* pDev = CreateDevice(deviceName.c_str());
        AddInstalledDevice(pDev);

        char temp[MM::MaxStrLength];
        pDev->GetName(temp);
        LogMessage("Created pump: " + (string)temp);
    }
    return nPumps_ > 0 ? DEVICE_OK : DEVICE_ERR;
}

///////////////////////////////////////////////////////////////////////////////
// WPIPumpHub class
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int WPIPumpHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct) {
    switch (eAct) {
    case MM::BeforeGet:
        pProp->Set(port_.c_str());
        break;
    case MM::AfterSet:
        pProp->Get(port_);
        break;
    }
    return DEVICE_OK;
}

int WPIPumpHub::OnNPumps(MM::PropertyBase* pProp, MM::ActionType eAct) {
    string temp = "";

    switch (eAct) {
    case MM::BeforeGet:
        temp = to_string(nPumps_);
        pProp->Set(temp.c_str());
        break;
    case MM::AfterSet:
        pProp->Get(temp);
        nPumps_ = stoi(temp);
        break;
    }
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// WPIPumpHub class
// Utility methods
///////////////////////////////////////////////////////////////////////////////

/**
* Pinging pump at address 0 (first pump in daisy chain), to check if any pump
* is connected.
*/
int WPIPumpHub::VerifyConnection(int idx) {
    string response = "";
    string cmd = to_string(idx) + " VER";
    SendSerialCommand(port_.c_str(), cmd.c_str(), CR);
    GetSerialAnswer(port_.c_str(), ETX, response);
    if (response[0] == '?') {
        LogMessage("Hub could not connect to port: " + port_ +
            " at address: " + to_string(idx));
        return DEVICE_SERIAL_COMMAND_FAILED;
    }
    else if (response[2] == 'A') {
        LogMessage("Hub connected successfully to port: " + port_ +
            ", but received an unexpected ping response.");
        return DEVICE_SERIAL_INVALID_RESPONSE;
    }
    LogMessage("Hub connected successfully to port: " + port_ +
        " at address: " + to_string(idx));
    return DEVICE_OK;
}

int WPIPumpHub::GetPort(string& port) {
    port = port_;
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// WPIPump class
//  Hub for WPI pumps
///////////////////////////////////////////////////////////////////////////////

WPIPump::WPIPump(int idx) :
    initialized_(false),
    busy_(false),
    port_("Undefined"),
    minVolumeUl_(0),
    maxVolumeUl_(1000),
    volumeUl_(0),
    diameter_(0),
    flowrateUlperSecond_(0),
    direction_(1)
{
    // Set pump id and name
    id_ = idx;
    name_ = ((string)g_WPIPumpName) + to_string(id_);
    thd_ = new PumpThread(this);

    // parent ID display
    CreateHubIDProperty();
}

WPIPump::~WPIPump() {
    free(thd_);
    Shutdown();
};

///////////////////////////////////////////////////////////////////////////////
// WPIPump class
// MMDevice API
///////////////////////////////////////////////////////////////////////////////

int WPIPump::Initialize() {
    if (initialized_)
        return DEVICE_OK;

    // Link to parent Hub
    WPIPumpHub* pHub = static_cast<WPIPumpHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel); // for backward comp.
    }
    else
        LogMessage(NoHubError);

    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, name_.c_str(), true);
    if (DEVICE_OK != ret)
        return ret;

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "WPI syringe pump", true);
    if (DEVICE_OK != ret)
        return ret;

    // Set COM Port
    pHub->GetPort(port_);

    // Set communication parameters (obtained from Hub)
    CPropertyAction* pAct;

    // Set maxVolume
    pAct = new CPropertyAction(this, &WPIPump::OnMaxVolume);
    ret = CreateFloatProperty("Max Volume uL", maxVolumeUl_, false, pAct);

    // Set currentVolume
    pAct = new CPropertyAction(this, &WPIPump::OnCurrentVolume);
    ret = CreateFloatProperty("Current Volume uL", volumeUl_, false, pAct);

    // Set diameter
    GetDiameter(diameter_);
    pAct = new CPropertyAction(this, &WPIPump::OnDiameter);
    ret = CreateFloatProperty("Diameter mm", diameter_, false, pAct);
    SetPropertyLimits("Diameter mm", g_Diameter_min, g_Diameter_max);

    // Set direction
    vector<string> allowedDirections = { "1", "-1" };
    pAct = new CPropertyAction(this, &WPIPump::OnDirection);
    ret = CreateIntegerProperty("Direction", direction_, false, pAct);
    SetAllowedValues("Direction", allowedDirections);

    // Set flowrate
    GetFlowrateUlPerSecond(flowrateUlperSecond_);
    pAct = new CPropertyAction(this, &WPIPump::OnFlowrate);
    ret = CreateFloatProperty("Flow rate uL/sec", flowrateUlperSecond_, false, pAct);
    SetPropertyLimits("Flow rate uL/sec", -calculate_flowrate(g_Speed_max, diameter_), calculate_flowrate(g_Speed_max, diameter_));

    // Start dispense
    vector<string> allowedRunValues = { "1", "0" };
    pAct = new CPropertyAction(this, &WPIPump::OnRun);
    ret = CreateIntegerProperty("Start", run_, false, pAct);
    SetAllowedValues("Start", allowedRunValues);

    initialized_ = true;
    return DEVICE_OK;
}

int WPIPump::Shutdown() {
    if (!initialized_) {
        return DEVICE_OK;
    }

    if (IsPumping()) {
        Stop();
    }

    initialized_ = false;
    return DEVICE_OK;
}

bool WPIPump::Busy() {
    return false;
}

void WPIPump::GetName(char* name) const {
    // Return the name used to refer to this device adapter
    CDeviceUtils::CopyLimitedString(name, name_.c_str());
}

///////////////////////////////////////////////////////////////////////////////
// WPIPump class
// MMPump API
///////////////////////////////////////////////////////////////////////////////

int WPIPump::GetPort(string& port) {
    port = port_;
    return DEVICE_OK;
}

// Alladin pumps don't support an obvious way to home the pump, besides
// crashing the pump and catching the error...
// Hence, just set the currentVolume to 0.
int WPIPump::Home() {
    if (IsPumping()) {
        return DEVICE_PUMP_IS_RUNNING;
    }

    MMThreadGuard g(this->currentVolumeLock_);
    volumeUl_ = 0;
    return DEVICE_OK;
}

int WPIPump::Stop() {
    // If pump is not running, no need to stop it.
    if (!IsPumping())
        return DEVICE_OK;

    thd_->Stop();

    Send(to_string(id_) + " STP");
    return DEVICE_OK;
}

int WPIPump::GetMaxVolumeUl(double& volUl) {
    volUl = maxVolumeUl_;
    return DEVICE_OK;
}

int WPIPump::SetMaxVolumeUl(double volUl) {
    if (IsPumping())
        return DEVICE_PUMP_IS_RUNNING;

    maxVolumeUl_ = volUl;
    SetPropertyLimits("Current Volume uL", minVolumeUl_, maxVolumeUl_);
    return DEVICE_OK;
}

int WPIPump::GetVolumeUl(double& volUl) {
    MMThreadGuard g(this->currentVolumeLock_);
    volUl = volumeUl_;
    return DEVICE_OK;
}

int WPIPump::SetVolumeUl(double volUl) {
    if (IsPumping())
        return DEVICE_PUMP_IS_RUNNING;

    MMThreadGuard g(this->currentVolumeLock_);
    volumeUl_ = volUl;
    return DEVICE_OK;
}

int WPIPump::IsDirectionInverted(bool& invert) {
    invert = (direction_ == -1);
    return DEVICE_OK;
}

int WPIPump::InvertDirection(bool invert) {
    if (IsPumping())
        return DEVICE_PUMP_IS_RUNNING;
    direction_ = (invert) ? -1 : 1;
    return DEVICE_OK;
}

int WPIPump::GetDiameter(double& diam) {
    string response = "";
    Send(to_string(id_) + " DIA");
    ReceiveOneLine(response);
    if (response[2] == 'A') {
        return DEVICE_SERIAL_INVALID_RESPONSE;
    }
    diam = stod(response.substr(4));
    return DEVICE_OK;
}

int WPIPump::SetDiameter(double diam) {
    if (IsPumping()) {
        return DEVICE_PUMP_IS_RUNNING;
    }
    if (diam < g_Diameter_min || diam > g_Diameter_max) {
        return DEVICE_INVALID_PROPERTY_VALUE;
    }

    stringstream msg;
    msg << to_string(id_) + " DIA " << setprecision(4) << diam;
    int ret = Send(msg.str());

    if (ret == DEVICE_OK)
        diameter_ = diam;

    SetPropertyLimits("Flow rate uL/sec", -calculate_flowrate(g_Speed_max, diam), calculate_flowrate(g_Speed_max, diam));

    return (ret == DEVICE_OK) ? DEVICE_OK : DEVICE_SERIAL_COMMAND_FAILED;
}

int WPIPump::GetFlowrateUlPerSecond(double& flowrate) {
    // Inquire direction
    string temp = "";
    Send(to_string(id_) + " DIR");
    ReceiveOneLine(temp);
    long infWith = (temp.substr(4) == "INF") ? 1 : -1;

    // Inquire flowrate
    Send(to_string(id_) + " RAT");
    ReceiveOneLine(temp);
    flowrate = direction_ * infWith * PumpFlowrateTouL(temp.substr(4));
    return DEVICE_OK;
}

int WPIPump::SetFlowrateUlPerSecond(double flowrate) {
    // Cannot change flowrate while pumping, so temporarily stop pump
    bool isPumping = IsPumping();
    if (isPumping) {
       // Don't use Start/Stop here, as it will reset the dispense timer
       Send(to_string(id_) + " STP");
       CDeviceUtils::SleepMs(5);
    }

    if (abs(flowrate) < calculate_flowrate(g_Speed_min, diameter_)) {
        LogMessage("Flowrate: " + to_string(flowrate) + " is too low");
        return DEVICE_INVALID_PROPERTY_VALUE;
    }
    if (abs(flowrate) > calculate_flowrate(g_Speed_max, diameter_)) {
        LogMessage("Flowrate: " + to_string(flowrate) + " is too high. " +
            "The maximum flowrate is: " +
            to_string(calculate_flowrate(g_Speed_max, diameter_)));
        return DEVICE_INVALID_PROPERTY_VALUE;
    }

    if (flowrate > 0 && direction_ == 1) {
        Send(to_string(id_) + " DIR INF");
    }
    else {
        Send(to_string(id_) + " DIR WDR");
    }
        
    AdjustUnits(abs(flowrate));

    stringstream msg;
    msg << to_string(id_) + " RAT C ";
    msg << setprecision(4) << uLToPumpFlowrate(abs(flowrate), flowrate_unit_);
    msg << GetUnitString();
    int ret = Send(msg.str());
    if (ret == DEVICE_OK)
        flowrateUlperSecond_ = flowrate;

    // Restart pump if it was pumping before
    // Don't use Start/Stop here, as it will reset the dispense timer
    if (isPumping) {
        CDeviceUtils::SleepMs(5);
        Send(to_string(id_) + " RUN");
    }
    return (ret == DEVICE_OK) ? DEVICE_OK : DEVICE_SERIAL_COMMAND_FAILED;
}

int WPIPump::Start() {
    if (IsPumping())
        return DEVICE_PUMP_IS_RUNNING;

	double seconds = 0;
    { // Scope for threadguard
        MMThreadGuard g(this->currentVolumeLock_);
        if (flowrateUlperSecond_ >= 0)
            seconds = volumeUl_ / flowrateUlperSecond_;
        else
            seconds = (maxVolumeUl_ - volumeUl_) / abs(flowrateUlperSecond_);
    }

    DispenseDurationSeconds(seconds);
    return DEVICE_OK;
}

int WPIPump::DispenseVolumeUl(double volUl) {
    if (IsPumping()) {
        return DEVICE_PUMP_IS_RUNNING;
    }

    if (volUl < 0) {
        volUl = -volUl;
        LogMessage("Negative volume was provided. Flowrate is used to indicate infusion/withdrawal. The absolute value was used.");
    }

    // Calculate duration based on volume and flowrate
    double seconds = abs(volUl / flowrateUlperSecond_);
    DispenseDurationSeconds(seconds);
    return DEVICE_OK;
}

int WPIPump::DispenseDurationSeconds(double seconds) {
    if (IsPumping())
        return DEVICE_PUMP_IS_RUNNING;

    duration_ = seconds;
    if (duration_ < 0) {
        LogMessage("Negative dispense/withdraw duration. Check the sign of the flowrate and volume");
        return DEVICE_ERR;
    }

    // Set correct unit
    double volToBeDispensed = abs(seconds * flowrateUlperSecond_);
    if (volToBeDispensed > 1000) {
        Send(to_string(id_) + " VOL ML");
    }
    else {
        Send(to_string(id_) + " VOL UL");
    }

    // Set volume to be dispensed/withdrawn
    stringstream msg;
    msg << to_string(id_) + " VOL ";
    if (volToBeDispensed > 1000)
        msg << setprecision(4) << volToBeDispensed / 1000;
    else
        msg << setprecision(4) << volToBeDispensed;
    Send(msg.str());
    Send(to_string(id_) + " RUN");

    // Run dispense/withdraw
    {
        MMThreadGuard g(this->currentVolumeLock_);
        startVolume_ = volumeUl_;
    }
    thd_->Start(duration_, flowrateUlperSecond_);
    return DEVICE_OK;
}

/**
* Actual updating in separate function to release the threadlocks asap.
*/
int WPIPump::UpdateVolume(double dt) {
    MMThreadGuard g1(this->currentVolumeLock_);
    volumeUl_ = startVolume_ - flowrateUlperSecond_ * dt;
    return DEVICE_OK;
}

int WPIPump::RunOnThread(double dt) {
    UpdateVolume(dt);
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// WPIPump class
// Action Handlers
///////////////////////////////////////////////////////////////////////////////

int WPIPump::OnMaxVolume(MM::PropertyBase* pProp, MM::ActionType eAct) {
    switch (eAct) {
    case MM::BeforeGet:
        pProp->Set(maxVolumeUl_);
        break;
    case MM::AfterSet:
        if (IsPumping()) {
            return DEVICE_PUMP_IS_RUNNING;
        }
        string temp;
        pProp->Get(temp);
        SetMaxVolumeUl(stod(temp));
        break;
    }
    return DEVICE_OK;
}

int WPIPump::OnCurrentVolume(MM::PropertyBase* pProp, MM::ActionType eAct) {
    MMThreadGuard g(this->currentVolumeLock_);
    switch (eAct) {
    case MM::BeforeGet:
        if (volumeUl_ < minVolumeUl_ || volumeUl_ > maxVolumeUl_)
            volumeUl_ = (volumeUl_ < minVolumeUl_) ? minVolumeUl_ : maxVolumeUl_;
        pProp->Set(volumeUl_);
        break;
    case MM::AfterSet:
        if (IsPumping()) {
            return DEVICE_PUMP_IS_RUNNING;
        }
        string temp;
        pProp->Get(temp);
        SetVolumeUl(stod(temp));
        break;
    }
    return DEVICE_OK;
}

int WPIPump::OnDiameter(MM::PropertyBase* pProp, MM::ActionType eAct) {
    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::BeforeGet:
        pProp->Set(diameter_);
        break;
    case MM::AfterSet:
        if (IsPumping()) {
            return DEVICE_PUMP_IS_RUNNING;
        }
        string temp;
        pProp->Get(temp);
        ret = SetDiameter(stod(temp));
        if (ret == DEVICE_OK) {
            diameter_ = stod(temp);
        }
        break;
    }
    return ret;
}

int WPIPump::OnDirection(MM::PropertyBase* pProp, MM::ActionType eAct) {
    switch (eAct) {
    case MM::BeforeGet:
        pProp->Set((long)direction_);
        break;
    case MM::AfterSet:
        if (IsPumping()) {
            return DEVICE_PUMP_IS_RUNNING;
        }
        string temp;
        pProp->Get(temp);
        InvertDirection(temp == "-1");
        break;
    }
    return DEVICE_OK;
}

int WPIPump::OnFlowrate(MM::PropertyBase* pProp, MM::ActionType eAct) {
    MMThreadGuard g(currentFlowrateLock_);

    int ret = DEVICE_OK;
    switch (eAct) {
    case MM::BeforeGet:
        pProp->Set(flowrateUlperSecond_);
        break;
    case MM::AfterSet:
        string temp;
        pProp->Get(temp);
        ret = SetFlowrateUlPerSecond(stod(temp));
        if (ret == DEVICE_OK) {
            flowrateUlperSecond_ = stod(temp);
        }
        break;
    }
    return ret;
}

int WPIPump::OnRun(MM::PropertyBase* pProp, MM::ActionType eAct) {
    switch (eAct) {
    case MM::BeforeGet:
        pProp->Set(run_);
        break;
    case MM::AfterSet:
        string temp;
        pProp->Get(temp);

        if (stol(temp) == run_) { return DEVICE_OK; }
        if (temp == "1") {
            Start();
        }
        else if (temp == "0") {
            Stop();
        }
        run_ = (temp == "1") ? 1 : 0;
        break;
    }
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// WPIPump class
// Utility methods
///////////////////////////////////////////////////////////////////////////////

int WPIPump::Send(string cmd) {
    LogMessage("Sent command: " + cmd + " to pump");
    CDeviceUtils::SleepMs(10);
    Purge();
    int ret = SendSerialCommand(port_.c_str(), cmd.c_str(), CR);
    return (ret == DEVICE_OK) ? DEVICE_OK : DEVICE_SERIAL_COMMAND_FAILED;
}

int WPIPump::ReceiveOneLine(string& ans) {
    ans = "";
    int ret = GetSerialAnswer(port_.c_str(), ETX, ans);
    return (ret == DEVICE_OK) ? DEVICE_OK : DEVICE_SERIAL_COMMAND_FAILED;
}

int WPIPump::Purge() {
    int ret = PurgeComPort(port_.c_str());
    return (ret == DEVICE_OK) ? DEVICE_OK : DEVICE_SERIAL_COMMAND_FAILED;
}

int WPIPump::AdjustUnits(double flowrate) {
    int temp_unit = flowrate_unit_;
    double pump_value = uLToPumpFlowrate(flowrate, temp_unit);
    while (pump_value < 1 || pump_value >= 10000) {
        if (pump_value < 0.1) {
            if (temp_unit == uL_hr)
                break;
            else
                temp_unit++;
        }
        else
            if (temp_unit == mL_min)
                break;
            else
                temp_unit++;
        pump_value = uLToPumpFlowrate(flowrate, temp_unit);
    }
    flowrate_unit_ = temp_unit;
    return DEVICE_OK;
}

double WPIPump::uLToPumpFlowrate(double flowrate, int flowrate_unit) {
    double value_to_pump = flowrate;
    switch (flowrate_unit) {
    case mL_min:
        value_to_pump *= 0.06;
        break;
    case mL_hr:
        value_to_pump *= 3.6;
        break;
    case uL_min:
        value_to_pump *= 60;
        break;
    case uL_hr:
        value_to_pump *= 3600;
        break;
    }
    return value_to_pump;
}

double WPIPump::PumpFlowrateTouL(string flowrate) {
    double flowrate_uL_s = stod(flowrate.substr(0, flowrate.size()-2));
    string unit = flowrate.substr(flowrate.size() - 2, 2);
    if (unit == "UM") {
        flowrate_uL_s /= 60;
    }
    else if (unit == "MM") {
        flowrate_uL_s *= 1000 / 60;
    }
    else if (unit == "UH") {
        flowrate_uL_s /= 3600;
    }
    else if (unit == "MH") {
        flowrate_uL_s /= 3.6;
    }
    return flowrate_uL_s;
}

string WPIPump::GetUnitString() {
    switch (flowrate_unit_) {
    case mL_min:
        return "MM";
    case mL_hr:
        return "MH";
    case uL_min:
        return "UM";
    case uL_hr:
        return "UH";
    }
    return "";
}

bool WPIPump::IsPumping() {
    return !thd_->IsStopped();
}

///////////////////////////////////////////////////////////////////////////////
// PumpThread class
// Thread to run pump (prevents blocking of main thread).
///////////////////////////////////////////////////////////////////////////////

PumpThread::PumpThread(WPIPump* pPump) {
    pump_ = pPump;
    stop_ = true;
};

PumpThread::~PumpThread() {};

///////////////////////////////////////////////////////////////////////////////
// PumpThread class
// MMPump API
///////////////////////////////////////////////////////////////////////////////

void PumpThread::Start(double duration, double flowrateUlperSecond) {
    MMThreadGuard g(this->stopLock_);
    duration_ = duration;
    flowrateUlperSecond_ = flowrateUlperSecond;
    stop_ = false;
    pump_->LogMessage("Thread is started");
    activate();
};

void PumpThread::Stop() {
    MMThreadGuard g(this->stopLock_);
    stop_ = true;
};

bool PumpThread::IsStopped() {
    MMThreadGuard g(this->stopLock_);
    return stop_;
}

///////////////////////////////////////////////////////////////////////////////
// PumpThread class
// MMDeviceThreadBase API
///////////////////////////////////////////////////////////////////////////////

int PumpThread::svc(void) throw() {
    int ret = DEVICE_OK;
    startTime_ = pump_->GetCurrentMMTime();
    try {
        MM::MMTime currentTime = pump_->GetCurrentMMTime();
        while (DEVICE_OK == ret && !IsStopped() && dt_ <= duration_) {
            CDeviceUtils::SleepMs(1); // Limit computational stress
            dt_ = (pump_->GetCurrentMMTime() - startTime_).getMsec() / 1000; // Convert ms to seconds
            ret = pump_->RunOnThread(dt_);
		    updateDuration();
        }
        if (IsStopped())
            pump_->LogMessage("Pump stopped by the user.\n");
        if (ret == DEVICE_OK) {
            pump_->LogMessage("Dispense/withdrawal finished.\n");
        }
    }
    catch (...) {
        pump_->LogMessage(g_Msg_EXCEPTION_IN_THREAD);
    }
    Stop();
    return ret;
}

int PumpThread::updateDuration() {
    double pumpFlowrate = 0;
    {
        MMThreadGuard g(pump_->currentFlowrateLock_);
        pumpFlowrate = pump_->flowrateUlperSecond_;
    }
    if (pumpFlowrate == flowrateUlperSecond_) { return DEVICE_OK; }

    duration_ = dt_ + (duration_ - dt_) * (flowrateUlperSecond_ / pumpFlowrate);
    flowrateUlperSecond_ = pumpFlowrate;
    return DEVICE_OK;
}