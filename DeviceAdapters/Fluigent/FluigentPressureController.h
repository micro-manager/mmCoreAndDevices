///////////////////////////////////////////////////////////////////////////////
// FILE:          FluigentPressureController.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Fluigent pressure controllers
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
//LAST UPDATE:    08.04.2025 LK

#ifndef _FLUIGENT_PRESSURE_CONTROLLER_H_
#define _FLUIGENT_PRESSURE_CONTROLLER_H_

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include "ModuleInterface.h"
#include <string>
#include <map>
#include <stdint.h>
#include <future>

#include "fgt_SDK.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// FluigentHub class
// Hub for Fluigent Pressure Controller devices
///////////////////////////////////////////////////////////////////////////////

class FluigentHub : public HubBase<FluigentHub>
{
public:
    FluigentHub();
    ~FluigentHub();

    ///////////////////////////////////////////////////////////////////////////
    // FluigentChannel class
    // MMDevice API
    ///////////////////////////////////////////////////////////////////////////

    /**
    * Initializes the Fluigent Hub. This initializes the Fluigent library, and
    * automatically detects all connected Fluigent pressure controllers (it
    * ignores all non-pressure controller devices)
    * Required by the MM::Device API.
    *
    * @returns MMDeviceConstants ErrorCode
    */
    int Initialize();

    /**
    * Hub safely breaks any communication with Fluigent devices.
    * Required by the MM::Device API.
    *
    * @returns MMDeviceConstants ErrorCode
    */
    int Shutdown();

    /**
    * Obtains Device name.
    * Required by the MM::Device API.
    *
    * @param[in] pName - char array with name of device
    */
    void GetName(char* pName) const;

    /**
    * Hub only handles initialization of devices, hence the Hub is never busy
    * Required by the MM::Device API.
    *
    * @return bool - Whether device accepts new commands or not
    */
    bool Busy() { return busy_; };

    ///////////////////////////////////////////////////////////////////////////
    // FluigentHub class
    // MMHub API
    ///////////////////////////////////////////////////////////////////////////

    /**
    * Automatically detects all connected Fluigent devices, and adds them to
    * the list of installed devices
    * Required by the MM::Hub API.
    *
    * @returns MMDeviceConstants ErrorCode
    */
    int DetectInstalledDevices();

    ///////////////////////////////////////////////////////////////////////////
    // FluigentChannel class
    // Action handlers
    ///////////////////////////////////////////////////////////////////////////

    /**
    * This function is called upon a change in the Measured Pressure device
    *  property. It handles both get and set calls.
    *
    * @param[in] pProp - Imposed Pressure property
    * @param[in] eAct - Action Type (get/set)
    * @returns MMDeviceConstants ErrorCode
    */
    int OnCalibrate(MM::PropertyBase* pProp, MM::ActionType eAct);    

    ///////////////////////////////////////////////////////////////////////////
    // FluigentHub class
    // Class Members
    ///////////////////////////////////////////////////////////////////////////

    /**
    *  Getter for the number of detected channels
    *
    * @param[out] nChannels - Number of detected channels
    * @returns MMDeviceConstants ErrorCode
    */
    int GetNChannels(int& nChannels);

public:
    // Hub public values
    int nDevices_; // Number of different devices
    int nChannels_; // Number of pressure channels
    string calibrate_; // Current calibration mode (basically always "None")

private:
    // Class member parameters
    bool initialized_;
    bool busy_;
    unsigned char errorCode_; // return errorcode of Fluigent functions
    unsigned short SNs_[256] = { 0 }; // array of device serial numbers
    int instrumentTypes_[256] = { 0 }; // array of device types
    fgt_CHANNEL_INFO channelInfo_[256]; // array of deviceInfo structs
    string unit_; // unit of pressure (basically always kPa)
};

///////////////////////////////////////////////////////////////////////////////
// FluigentChannel class
// Fluigent Pressure Controller Channel
///////////////////////////////////////////////////////////////////////////////

class FluigentChannel : public CPressurePumpBase<FluigentChannel>
{
public:
    FluigentChannel(int idx);
    ~FluigentChannel() {};

    ///////////////////////////////////////////////////////////////////////////
    // FluigentChannel class
    // MMDevice API
    ///////////////////////////////////////////////////////////////////////////

    /**
    * Initializes the Fluigent Channel. Device properties and action handlers
    * are initialized
    * Required by the MM::Device API.
    * 
    * @returns MMDeviceConstants ErrorCode
    */
    int Initialize();

    /**
    * No need to further implement anything, shutdown is handled by the Hub
    * Required by the MM::Device API.
    * 
    * @returns MMDeviceConstants ErrorCode
    */
    int Shutdown() { return DEVICE_OK; }

    /**
    * Obtains Device name.
    * Required by the MM::Device API.
    * 
    * @param[in] pName - char array with name of device
    */
    void GetName(char* pName) const;

    /**
    * Changing states is nearly instantaneous, hence device is never busy
    * Required by the MM::Device API.
    * 
    * @return bool - Whether device accepts new commands or not
    */
    bool Busy() { return false; }

    ///////////////////////////////////////////////////////////////////////////
    // FluigentChannel class
    // MMPump API
    ///////////////////////////////////////////////////////////////////////////

    /**
    * Needs to stop pressure, i.e. set pressure to 0
    * Required by the MM::PressurePump API
    *
    * @returns MMDeviceConstants ErrorCode
    */
    int Stop();

    /**
    * Needs to stop pressure, i.e. set pressure to 0
    * Required by the MM::PressurePump API
    *
    * @returns boolean
    */
    bool RequiresCalibration() { return false; }

    /**
    * Get-function for the pressure of a specific pressure channel
    * Optional function of the MM::PressurePump API
    * 
    * @param[out] P - Measured pressure of that channel in kPa
    * @returns MMDeviceConstants ErrorCode
    */
    int GetPressureKPa(double& P);

    /**
    * Set-function for the pressure of a specific channel
    * Optional function of the MM::PressurePump API
    *
    * @param[in] P - Pressure to be set in kPa
    * @returns MMDeviceConstants ErrorCode
    */
    int SetPressureKPa(double P);

    /**
    * Get-function for the limits of a specific channel
    * Optional function of the MM::PressurePump API
    * 
    * @param[out] Pmin - Minimum pressure of that channel
    * @param[out] Pmax - Maximum pressure of that channel
    * @returns MMDeviceConstants ErrorCode
    */
    int GetPressureLimits(double& Pmin, double& Pmax);

    /**
    * Execute internal calibration on a specific channel
    * Optional function of the MM::PressurePump API
    * 
    * @returns MMDeviceConstants ErrorCode
    */
    int Calibrate();

    ///////////////////////////////////////////////////////////////////////////
    // FluigentChannel class
    // Action handlers
    ///////////////////////////////////////////////////////////////////////////

    /**
    * This function is called upon a change in the Imposed Pressure device
    *  property. It handles both get and set calls.
    *
    * @param[in] pProp - Imposed Pressure property
    * @param[in] eAct - Action Type (get/set), others are ignored
    * @returns MMDeviceConstants ErrorCode
    */
    int OnImposedPressure(MM::PropertyBase* pProp, MM::ActionType eAct);

    /**
    * This function is called upon a change in the Measured Pressure device
    *  property. This is a get-only function (other calls are ignored)
    *
    * @param[in] pProp - Imposed Pressure property
    * @param[in] eAct - Action Type (get)
    * @returns MMDeviceConstants ErrorCode
    */
    int OnMeasuredPressure(MM::PropertyBase* pProp, MM::ActionType eAct);

///////////////////////////////////////////////////////////////////////////
// FluigentChannel class
// Class Members
///////////////////////////////////////////////////////////////////////////

public:
    // Channel public values
    double Pmin_ = 0; // Minimum pressure
    double Pmax_ = 0; // Maximum pressure
    double Pimp_ = 0; // Imposed pressure
    double Pmeas_= 0; // Measured pressure

private:
    // Class member parameters
    bool initialized_;
    bool busy_;
    int idx_; // index of this channel in the device array of the Hub
    unsigned char errorCode_; // Fluigent error code
    fgt_CHANNEL_INFO channelInfo_; // Channel info of this channel
};
#endif //_FLUIGENT_PRESSURE_CONTROLLER_H_