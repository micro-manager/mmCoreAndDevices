///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_Commands.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Implementation of commands to be used with AMF devices.
//                
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes, Paris, France
//
// YEAR:          2024
//                
// VERSION:       0.1
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
//LAST UPDATE:    26.02.2024 LK

#ifndef _AMF_COMMANDS_H_
#define _AMF_COMMANDS_H_

#include "MMDeviceConstants.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>

enum {
    AMF_ERR_UNKNOWN_COMMAND = 10128,
    AMF_ERR_NOT_HOMED = 10144,
    AMF_ERR_MOVE_OUT_RANGE = 10145,
    AMF_ERR_SPEED_OUT_RANGE = 146,
    AMF_ERR_BLOCKED = 10224,
    AMF_ERR_SENSOR = 10225,
    AMF_ERR_MISSING_MAIN_REFERENCE = 10226,
    AMF_ERR_MISSING_REFERENCE = 10227,
    AMF_ERR_BAD_POLARITY = 10227
};

enum class AMF_Command {
	// Generic
	Initialize,
	Get_pump_status,
    Get_valve_status,
	Get_firmware_version,
	Is_initialized,
	Stop,
	Get_address,

	// Valve
	Move_valve_shortest,
	Move_valve_cw,
	Move_valve_ccw,
	Get_valve_position,
	Get_n_valves,

	// Pump
	Move_plunger_absolute,
	Move_plunger_pickup,
	Move_plunger_dispense,
	Set_acceleration,
	Set_deceleration,
	Set_n_steps,
	Set_flowrate,
	Get_plunger_position,
	Get_acceleration,
	Get_deceleration,
	Get_n_steps,
	Get_flowrate,
	Get_pump_info
};

namespace {

    const char* AMF_Baud = "9600";
    const char* AMF_Parity = "None";
    const char* AMF_StopBits = "1";
    const char* AMF_EOL = "\n";

    const char* AMF_Hub_Name = "AMF Hub";
    const char* AMF_LSP_Hub_Name = "AMF LSP Hub";
    const char* AMF_RVM_Name = "AMF RVM";
    const char* AMF_LSP_Pump_Name = "AMF LSP Pump";

    const char* AMF_START = "/";
    const char* AMF_END = "R";
    const char* AMF_TERM = "\r";
    const char AMF_ACK = 0;
    const char AMF_NACK = 1;

	const char* AMF_Rotation_Shortest = "Shortest";
	const char* AMF_Rotation_Clockwise = "Clockwise";
	const char* AMF_Rotation_CounterClockwise = "Counterclockwise";

    bool AMF_ERROR_INITIALIZED = false;

    std::string AMF_get_command_string(
        int address,
        AMF_Command cmd,
        int value
    )
    {
        std::string cmd_string = "/" + std::to_string(address);
        switch (cmd) {
        case AMF_Command::Initialize:
            cmd_string += "ZR";
            break;
        case AMF_Command::Get_pump_status:
            cmd_string += "?9100";
            break;
        case AMF_Command::Get_valve_status:
            cmd_string += "?9200";
            break;
        case AMF_Command::Get_firmware_version:
            cmd_string += "?23";
            break;
        case AMF_Command::Is_initialized:
            cmd_string += "?9010";
            break;
        case AMF_Command::Stop:
            cmd_string += "T";
            break;
        case AMF_Command::Get_address:
            // Address is a placeholder, so use the broadcast address '_'
            cmd_string = "/_?26";
            break;
        case AMF_Command::Move_valve_shortest:
            cmd_string += "B" + std::to_string(value) + "R";
            break;
        case AMF_Command::Move_valve_cw:
            cmd_string += "I" + std::to_string(value) + "R";
            break;
        case AMF_Command::Move_valve_ccw:
            cmd_string += "O" + std::to_string(value) + "R";
            break;
        case AMF_Command::Get_valve_position:
            cmd_string += "?6";
            break;
        case AMF_Command::Get_n_valves:
            cmd_string += "?801";
            break;
        case AMF_Command::Move_plunger_absolute:
            cmd_string += "A" + std::to_string(value) + "R";
            break;
        case AMF_Command::Move_plunger_pickup:
            cmd_string += "P" + std::to_string(value) + "R";
            break;
        case AMF_Command::Move_plunger_dispense:
            cmd_string += "D" + std::to_string(value) + "R";
            break;
        case AMF_Command::Set_acceleration:
            cmd_string += "L" + std::to_string(value) + "R";
            break;
        case AMF_Command::Set_deceleration:
            cmd_string += "l" + std::to_string(value) + "R";
            break;
        case AMF_Command::Set_n_steps:
            cmd_string += "N" + std::to_string(value) + "R";
            break;
        case AMF_Command::Set_flowrate:
            cmd_string += "u" + std::to_string(value) + "R";
            break;
        case AMF_Command::Get_plunger_position:
            cmd_string += "?";
            break;
        case AMF_Command::Get_acceleration:
            cmd_string += "?25";
            break;
        case AMF_Command::Get_deceleration:
            cmd_string += "?27";
            break;
        case AMF_Command::Get_n_steps:
            cmd_string += "?28";
            break;
        case AMF_Command::Get_flowrate:
            cmd_string += "?2";
            break;
        }
        return cmd_string;
    }

    int AMF_Initialize_Error_Messages() {
        if (AMF_ERROR_INITIALIZED) { return DEVICE_OK; }
        AMF_ERROR_INITIALIZED = true;
        return DEVICE_OK;
    }
}
#endif //_AMF_COMMANDS_H_