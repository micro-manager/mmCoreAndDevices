///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentProtocol.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 microscope protocol definitions
//
// COPYRIGHT:     University of California, San Francisco, 2025
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Nico Stuurman, 2025

#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace EvidentIX85 {

// Protocol constants
const char* const TERMINATOR = "\r\n";
const char TAG_DELIMITER = ' ';
const char DATA_DELIMITER = ',';
const char POSITIVE_ACK = '+';
const char NEGATIVE_ACK = '!';
const char UNKNOWN_RESPONSE = 'X';

const int MAX_COMMAND_LENGTH = 128;
const long ANSWER_TIMEOUT_MS = 4000;

// Serial port settings
const int BAUD_RATE = 115200;
const int DATA_BITS = 8;
const char PARITY = 'E';  // Even
const int STOP_BITS = 2;

// Command tags - System commands
const char* const CMD_LOGIN = "L";
const char* const CMD_UNIT = "U";
const char* const CMD_VERSION = "V";
const char* const CMD_ERROR = "ER";

// V command unit numbers for device detection
const int V_CONTROLLER = 1;
const int V_NOSEPIECE = 2;
const int V_CORRECTION_COLLAR_LINK = 3;
const int V_CORRECTION_COLLAR_ROULETTE = 4;
const int V_FOCUS = 5;
const int V_LIGHTPATH = 6;
const int V_CONDENSER_UNIT = 7;  // IX3-LWUCDA: Polarizer, CondenserTurret, DIAShutter, DIAAperture
const int V_DIC_UNIT = 8;        // IX5-DICTA: DICPrism, DICRetardation
const int V_MIRROR_UNIT1 = 9;
const int V_EPI_SHUTTER1 = 10;
const int V_MIRROR_UNIT2 = 11;
const int V_EPI_SHUTTER2 = 12;
const int V_MANUAL_CONTROL = 13;
const int V_EPIND = 14;
const int V_SD_MAGNIFICATION = 15;  // SDCA magnification (SPIN/SR system)
const int V_AUTOFOCUS = 16;
const int V_OFFSET_LENS = 17;
const int V_FV40_PSU = 18;

// Command tags - Focus
const char* const CMD_FOCUS_GOTO = "FG";
const char* const CMD_FOCUS_MOVE = "FM";
const char* const CMD_FOCUS_STOP = "FSTP";
const char* const CMD_FOCUS_POSITION = "FP";
const char* const CMD_FOCUS_NOTIFY = "NFP";
const char* const CMD_FOCUS_SPEED = "FSPD";
const char* const CMD_FOCUS_NEAR_LIMIT = "NL";
const char* const CMD_FOCUS_FAR_LIMIT = "FL";

// Command tags - Nosepiece (Objective Turret)
const char* const CMD_NOSEPIECE = "OB";
const char* const CMD_NOSEPIECE_NOTIFY = "NOB";

// Command tags - Magnification Changer
const char* const CMD_MAGNIFICATION = "CA";
const char* const CMD_MAGNIFICATION_NOTIFY = "NCA";

// Command tags - Light Path
const char* const CMD_LIGHT_PATH = "BIL";

// Command tags - Condenser
const char* const CMD_CONDENSER_TURRET = "TR";
const char* const CMD_CONDENSER_TURRET_NOTIFY = "NTR";
const char* const CMD_CONDENSER_CONTROL = "CD";
const char* const CMD_CONDENSER_NOTIFY = "NCD";
const char* const CMD_CONDENSER_SWITCH = "S1";

// Command tags - DIA (Transmitted Light)
const char* const CMD_DIA_APERTURE = "DAS";
const char* const CMD_DIA_APERTURE_NOTIFY = "NDAS";
const char* const CMD_DIA_APERTURE_STOP = "DASSTP";
const char* const CMD_DIA_SHUTTER = "DSH";
const char* const CMD_DIA_ILLUMINATION = "DIL";
const char* const CMD_DIA_ILLUMINATION_NOTIFY = "NDIL";

// Command tags - Polarizer
const char* const CMD_POLARIZER = "PO";
const char* const CMD_POLARIZER_NOTIFY = "NPO";

// Command tags - DIC
const char* const CMD_DIC_PRISM = "DIC";
const char* const CMD_DIC_RETARDATION = "DICR";
const char* const CMD_DIC_RETARDATION_NOTIFY = "NDICR";
const char* const CMD_DIC_LOCALIZED_NOTIFY = "NLC";
const char* const CMD_DIC_LOCALIZED = "LC";

// Command tags - EPI (Reflected Light)
const char* const CMD_EPI_SHUTTER1 = "ESH1";
const char* const CMD_EPI_SHUTTER2 = "ESH2";
const char* const CMD_MIRROR_UNIT1 = "MU1";
const char* const CMD_MIRROR_UNIT2 = "MU2";
const char* const CMD_MIRROR_UNIT_NOTIFY1 = "NMUINIT1";
const char* const CMD_MIRROR_UNIT_NOTIFY2 = "NMUINIT2";
const char* const CMD_COVER_SWITCH1 = "C1";
const char* const CMD_COVER_SWITCH2 = "C2";
const char* const CMD_EPI_ND = "END";

// Command tags - Right Port
const char* const CMD_RIGHT_PORT = "BIR";
const char* const CMD_RIGHT_PORT_NOTIFY = "NBIR";

// Command tags - Correction Collar
const char* const CMD_CORRECTION_COLLAR = "CC";
const char* const CMD_CORRECTION_COLLAR_LINK = "CCL";
const char* const CMD_CORRECTION_COLLAR_INIT = "CCINIT";

// Command tags - Autofocus (ZDC)
const char* const CMD_AF_START_STOP = "AF";
const char* const CMD_AF_STATUS = "AFST";
const char* const CMD_AF_TABLE = "AFTBL";
const char* const CMD_AF_PARAMETER = "AFP";
const char* const CMD_AF_GET_PARAMETER = "GAFP";
const char* const CMD_AF_NEAR_LIMIT = "AFNL";
const char* const CMD_AF_FAR_LIMIT = "AFFL";
const char* const CMD_AF_BUZZER = "AFBZ";
const char* const CMD_AF_APERTURE = "AFAS";
const char* const CMD_AF_DICHROIC = "AFDM";
const char* const CMD_AF_DICHROIC_MOVE = "AFDMG";

// Command tags - Offset Lens (part of ZDC)
const char* const CMD_OFFSET_LENS_GOTO = "ABG";
const char* const CMD_OFFSET_LENS_MOVE = "ABM";
const char* const CMD_OFFSET_LENS_STOP = "ABSTP";
const char* const CMD_OFFSET_LENS_POSITION = "ABP";
const char* const CMD_OFFSET_LENS_NOTIFY = "NABP";
const char* const CMD_OFFSET_LENS_RANGE = "ABRANGE";
const char* const CMD_OFFSET_LENS_LIMIT = "ABLMT";
const char* const CMD_OFFSET_LENS_LOST_MOTION = "ABLM";

// Command tags - MCZ (Manual Control Unit)
const char* const CMD_JOG = "JG";
const char* const CMD_JOG_SENSITIVITY_FINE = "JGSF";
const char* const CMD_JOG_SENSITIVITY_COARSE = "JGSC";
const char* const CMD_JOG_DIRECTION = "JGDR";  // Jog direction (0=Reverse, 1=Default)
const char* const CMD_JOG_LIMIT = "JGL";
const char* const CMD_OFFSET_LENS_SENSITIVITY_FINE = "ABJGSF";
const char* const CMD_OFFSET_LENS_SENSITIVITY_COARSE = "ABJGSC";

// Command tags - SD Magnification Changer
const char* const CMD_SD_MAGNIFICATION = "SDCA";

// Command tags - Indicators and Encoders
const char* const CMD_INDICATOR_CONTROL = "I";
const char* const CMD_INDICATOR1 = "I1";
const char* const CMD_INDICATOR2 = "I2";
const char* const CMD_INDICATOR3 = "I3";
const char* const CMD_INDICATOR4 = "I4";
const char* const CMD_INDICATOR5 = "I5";
const char* const CMD_ENCODER1 = "E1";
const char* const CMD_ENCODER2 = "E2";
const char* const CMD_ENCODER3 = "E3";
const char* const CMD_DIL_ENCODER_CONTROL = "DILE";
const char* const CMD_MCZ_SWITCH = "S2";

// Device limits and constants
const long FOCUS_MIN_POS = 0;
const long FOCUS_MAX_POS = 1050000;  // 10.5mm in 10nm units
const double FOCUS_STEP_SIZE_UM = 0.01;  // 10nm = 0.01um
const long FOCUS_POSITION_TOLERANCE = 10;  // 10 steps = 100nm tolerance for "at position" detection

const int NOSEPIECE_MIN_POS = 1;
const int NOSEPIECE_MAX_POS = 6;

const int MAGNIFICATION_MIN_POS = 1;
const int MAGNIFICATION_MAX_POS = 3;

// Most turrets and state devices have up to 6 positions
const int CONDENSER_TURRET_MAX_POS = 6;
const int MIRROR_UNIT_MAX_POS = 6;
const int POLARIZER_MAX_POS = 2;  // Out (0) and In (1)
const int DIC_PRISM_MAX_POS = 6;
const int EPIND_MAX_POS = 6;

const int LIGHT_PATH_LEFT_PORT = 1;
const int LIGHT_PATH_BI_50_50 = 2;
const int LIGHT_PATH_BI_100 = 3;
const int LIGHT_PATH_RIGHT_PORT = 4;

// Correction Collar
const long CORRECTION_COLLAR_MIN_POS = -3200;
const long CORRECTION_COLLAR_MAX_POS = 3200;
const double CORRECTION_COLLAR_STEP_SIZE_UM = 1.0;  // 1 step = 1 µm

// Manual Control Unit (MCU) 7-segment display codes
// These hex codes drive the 7-segment displays on the MCU indicators
const int SEG7_0 = 0xEE;
const int SEG7_1 = 0x28;
const int SEG7_2 = 0xCD;
const int SEG7_3 = 0x6D;
const int SEG7_4 = 0x2B;
const int SEG7_5 = 0x67;
const int SEG7_6 = 0xE7;
const int SEG7_7 = 0x2E;
const int SEG7_8 = 0xEF;
const int SEG7_9 = 0x6F;
const int SEG7_DASH = 0x01;

// Helper function to get 7-segment code for a digit
inline int Get7SegmentCode(int digit)
{
    switch (digit)
    {
        case 0: return SEG7_0;
        case 1: return SEG7_1;
        case 2: return SEG7_2;
        case 3: return SEG7_3;
        case 4: return SEG7_4;
        case 5: return SEG7_5;
        case 6: return SEG7_6;
        case 7: return SEG7_7;
        case 8: return SEG7_8;
        case 9: return SEG7_9;
        default: return SEG7_DASH;  // Return dash for invalid digits
    }
}

// Error codes (Evident specific, starting at 10100)
const int ERR_EVIDENT_OFFSET = 10100;
const int ERR_COMMAND_TIMEOUT = ERR_EVIDENT_OFFSET + 1;
const int ERR_NEGATIVE_ACK = ERR_EVIDENT_OFFSET + 2;
const int ERR_INVALID_RESPONSE = ERR_EVIDENT_OFFSET + 3;
const int ERR_NOT_IN_REMOTE_MODE = ERR_EVIDENT_OFFSET + 4;
const int ERR_DEVICE_NOT_AVAILABLE = ERR_EVIDENT_OFFSET + 5;
const int ERR_POSITION_UNKNOWN = ERR_EVIDENT_OFFSET + 6;
const int ERR_MONITOR_THREAD_FAILED = ERR_EVIDENT_OFFSET + 7;
const int ERR_PORT_NOT_SET = ERR_EVIDENT_OFFSET + 8;
const int ERR_PORT_CHANGE_FORBIDDEN = ERR_EVIDENT_OFFSET + 9;
const int ERR_CORRECTION_COLLAR_NOT_LINKED = ERR_EVIDENT_OFFSET + 10;
const int ERR_CORRECTION_COLLAR_LINK_FAILED = ERR_EVIDENT_OFFSET + 11;

// Helper functions
inline std::string BuildCommand(const char* tag)
{
    std::ostringstream cmd;
    cmd << tag;  // Don't add TERMINATOR - SendSerialCommand adds it
    return cmd.str();
}

inline std::string BuildCommand(const char* tag, int param1)
{
    std::ostringstream cmd;
    cmd << tag << TAG_DELIMITER << param1;
    return cmd.str();
}

inline std::string BuildCommand(const char* tag, int param1, int param2)
{
    std::ostringstream cmd;
    cmd << tag << TAG_DELIMITER << param1 << DATA_DELIMITER << param2;
    return cmd.str();
}

inline std::string BuildCommand(const char* tag, int param1, int param2, int param3)
{
    std::ostringstream cmd;
    cmd << tag << TAG_DELIMITER << param1 << DATA_DELIMITER << param2 << DATA_DELIMITER << param3;
    return cmd.str();
}

inline std::string BuildQuery(const char* tag)
{
    std::ostringstream cmd;
    cmd << tag << "?";
    return cmd.str();
}

// Parse response helpers
inline bool IsPositiveAck(const std::string& response, const char* tag)
{
    std::string expected = std::string(tag) + " +";
    return response.find(expected) == 0;
}

inline bool IsValidAnswer(const std::string& response, const char* tag)
{
   if (response.length() < 3)
      return false;
   if (response.substr(0, strlen(tag)) != tag)
      return false;
   if (response.substr(3, 1) == "!")
      return false;
   return true;
}

inline bool IsNegativeAck(const std::string& response, const char* tag)
{
    std::string expected = std::string(tag) + " !";
    return response.find(expected) == 0;
}

inline bool IsUnknown(const std::string& response)
{
    return response.find(" X") != std::string::npos;
}

inline std::string ExtractTag(const std::string& response)
{
    size_t pos = response.find(TAG_DELIMITER);
    if (pos == std::string::npos)
    {
        // No delimiter, might be tag-only response or tag with '?'
        pos = response.find('?');
        if (pos == std::string::npos)
            return response;
        else
            return response.substr(0, pos);
    }
    return response.substr(0, pos);
}

inline std::vector<std::string> ParseParameters(const std::string& response)
{
    std::vector<std::string> params;
    size_t tagEnd = response.find(TAG_DELIMITER);
    if (tagEnd == std::string::npos)
        return params;

    std::string dataStr = response.substr(tagEnd + 1);
    std::istringstream iss(dataStr);
    std::string param;

    while (std::getline(iss, param, DATA_DELIMITER))
    {
        // Trim whitespace
        size_t start = param.find_first_not_of(" \t\r\n");
        size_t end = param.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos)
            params.push_back(param.substr(start, end - start + 1));
        else if (start != std::string::npos)
            params.push_back(param.substr(start));
    }

    return params;
}

inline int ParseIntParameter(const std::string& param)
{
    // Handle empty string
    if (param.empty())
        return -1;

    // Handle unknown/not-present indicators
    if (param == "X" || param == "x")
        return -1;

    // Handle acknowledgment characters (shouldn't be parsed as numbers)
    if (param == "+" || param == "!")
        return -1;

    // Try to parse as integer with exception handling
    try
    {
        return std::stoi(param);
    }
    catch (const std::invalid_argument&)
    {
        // Not a valid integer
        return -1;
    }
    catch (const std::out_of_range&)
    {
        // Number too large for int
        return -1;
    }
}

inline long ParseLongParameter(const std::string& param)
{
    // Handle empty string
    if (param.empty())
        return -1;

    // Handle unknown/not-present indicators
    if (param == "X" || param == "x")
        return -1;

    // Handle acknowledgment characters (shouldn't be parsed as numbers)
    if (param == "+" || param == "!")
        return -1;

    // Try to parse as long with exception handling
    try
    {
        return std::stol(param);
    }
    catch (const std::invalid_argument&)
    {
        // Not a valid long
        return -1;
    }
    catch (const std::out_of_range&)
    {
        // Number too large for long
        return -1;
    }
}

// Helper function to check if position is within tolerance of target
inline bool IsAtTargetPosition(long currentPos, long targetPos, long tolerance)
{
    if (targetPos < 0)
        return false;  // No target set

    long diff = currentPos - targetPos;
    if (diff < 0)
        diff = -diff;  // Absolute value

    return diff <= tolerance;
}

} // namespace EvidentIX85
