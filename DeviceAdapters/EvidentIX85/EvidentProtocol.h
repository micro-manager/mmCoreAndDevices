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

namespace EvidentIX85 {

// Protocol constants
const char* const TERMINATOR = "\r\n";
const char TAG_DELIMITER = ' ';
const char DATA_DELIMITER = ',';
const char POSITIVE_ACK = '+';
const char NEGATIVE_ACK = '!';
const char UNKNOWN_RESPONSE = 'X';

const int MAX_COMMAND_LENGTH = 128;
const long ANSWER_TIMEOUT_MS = 2000;

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
const char* const CMD_JOG_DIRECTION = "JGDR";
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

const int NOSEPIECE_MIN_POS = 1;
const int NOSEPIECE_MAX_POS = 6;

const int MAGNIFICATION_MIN_POS = 1;
const int MAGNIFICATION_MAX_POS = 3;

const int LIGHT_PATH_LEFT_PORT = 1;
const int LIGHT_PATH_BI_50_50 = 2;
const int LIGHT_PATH_BI_100 = 3;
const int LIGHT_PATH_RIGHT_PORT = 4;

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

// Helper functions
inline std::string BuildCommand(const char* tag)
{
    std::ostringstream cmd;
    cmd << tag << TERMINATOR;
    return cmd.str();
}

inline std::string BuildCommand(const char* tag, int param1)
{
    std::ostringstream cmd;
    cmd << tag << TAG_DELIMITER << param1 << TERMINATOR;
    return cmd.str();
}

inline std::string BuildCommand(const char* tag, int param1, int param2)
{
    std::ostringstream cmd;
    cmd << tag << TAG_DELIMITER << param1 << DATA_DELIMITER << param2 << TERMINATOR;
    return cmd.str();
}

inline std::string BuildCommand(const char* tag, int param1, int param2, int param3)
{
    std::ostringstream cmd;
    cmd << tag << TAG_DELIMITER << param1 << DATA_DELIMITER << param2 << DATA_DELIMITER << param3 << TERMINATOR;
    return cmd.str();
}

inline std::string BuildQuery(const char* tag)
{
    std::ostringstream cmd;
    cmd << tag << "?" << TERMINATOR;
    return cmd.str();
}

// Parse response helpers
inline bool IsPositiveAck(const std::string& response, const char* tag)
{
    std::string expected = std::string(tag) + " +";
    return response.find(expected) == 0;
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
    if (param == "X" || param == "x")
        return -1;  // Unknown
    return std::stoi(param);
}

inline long ParseLongParameter(const std::string& param)
{
    if (param == "X" || param == "x")
        return -1;  // Unknown
    return std::stol(param);
}

} // namespace EvidentIX85
