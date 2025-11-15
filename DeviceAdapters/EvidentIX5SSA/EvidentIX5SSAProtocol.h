///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentIX5SSAProtocol.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Protocol constants and helpers for Evident IX5-SSA XY Stage
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

// Protocol constants
namespace IX5SSA {

const char* const TERMINATOR = "\r\n";
const char TAG_DELIMITER = ' ';
const char DATA_DELIMITER = ',';
const char POSITIVE_ACK = '+';
const char NEGATIVE_ACK = '!';

// Serial port settings
const int BAUD_RATE = 115200;
const int STOP_BITS = 2;
const char* const PARITY = "Even";

// Command tags
const char* const CMD_LOGIN = "L";
const char* const CMD_VERSION = "V";
const char* const CMD_UNIT = "U";
const char* const CMD_XY_INIT = "XYINIT";
const char* const CMD_XY_GOTO = "XYG";
const char* const CMD_XY_MOVE = "XYM";
const char* const CMD_XY_STOP = "XYSTP";
const char* const CMD_XY_POSITION = "XYP";
const char* const CMD_XY_POSITION_NOTIFY = "NXYP";
const char* const CMD_XY_SPEED = "XYSPD";
const char* const CMD_XY_JOG = "XYJG";
const char* const CMD_XY_JOG_SENSITIVITY = "XYJGS";
const char* const CMD_XY_JOG_DIR_X = "XJGDR";
const char* const CMD_XY_JOG_DIR_Y = "YJGDR";
const char* const CMD_ENCODER_1 = "E1";
const char* const CMD_ENCODER_2 = "E2";

// Device limits
const long XY_STAGE_MIN_POS_X = -743680;  // X position range in 78.125nm units
const long XY_STAGE_MAX_POS_X = 743680;   // ~58mm range in X
const long XY_STAGE_MIN_POS_Y = -500480;  // Y position range in 78.125nm units
const long XY_STAGE_MAX_POS_Y = 500480;   // ~39mm range in Y
const double XY_STAGE_STEP_SIZE_UM = 0.078125;  // 78.125nm = 0.078125µm per step
const long XY_STAGE_POSITION_TOLERANCE = 10;  // 10 steps tolerance

// Error codes
const int ERR_OFFSET = 10200;
const int ERR_COMMAND_TIMEOUT = ERR_OFFSET + 1;
const int ERR_NEGATIVE_ACK = ERR_OFFSET + 2;
const int ERR_INVALID_RESPONSE = ERR_OFFSET + 3;
const int ERR_PORT_NOT_SET = ERR_OFFSET + 4;
const int ERR_PORT_CHANGE_FORBIDDEN = ERR_OFFSET + 5;
const int ERR_DEVICE_NOT_AVAILABLE = ERR_OFFSET + 6;
const int ERR_POSITION_UNKNOWN = ERR_OFFSET + 7;

// Helper functions
inline std::string BuildCommand(const char* tag)
{
   return std::string(tag);
}

inline std::string BuildCommand(const char* tag, int param)
{
   std::ostringstream cmd;
   cmd << tag << TAG_DELIMITER << param;
   return cmd.str();
}

inline std::string BuildCommand(const char* tag, long param1, long param2)
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

inline bool IsNotificationTag(const std::string& response)
{
   // Notification tags that can come asynchronously
   // Need to distinguish notifications from command acknowledgments

   // Position notification: "NXYP x,y" vs acknowledgment: "NXYP +" or "NXYP !"
   if (response.find("NXYP ") == 0)
   {
      size_t spacePos = response.find(' ');
      if (spacePos != std::string::npos && spacePos + 1 < response.length())
      {
         char nextChar = response[spacePos + 1];
         // Notification has numeric value or 'X', acknowledgment has '+' or '!'
         return (nextChar != '+' && nextChar != '!');
      }
   }

   // Encoder notifications (only when they contain delta values, not acknowledgments)
   if (response.find("E1 ") == 0)
   {
      // Check if it's a notification (has numeric value) vs acknowledgment (+ or !)
      size_t spacePos = response.find(' ');
      if (spacePos != std::string::npos && spacePos + 1 < response.length())
      {
         char nextChar = response[spacePos + 1];
         return (nextChar != '+' && nextChar != '!');
      }
   }
   if (response.find("E2 ") == 0)
   {
      size_t spacePos = response.find(' ');
      if (spacePos != std::string::npos && spacePos + 1 < response.length())
      {
         char nextChar = response[spacePos + 1];
         return (nextChar != '+' && nextChar != '!');
      }
   }

   return false;
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

inline long ParseLongParameter(const std::string& param)
{
   if (param.empty() || param == "X" || param == "x" || param == "+" || param == "!")
      return -1;

   try
   {
      return std::stol(param);
   }
   catch (...)
   {
      return -1;
   }
}

inline int ParseIntParameter(const std::string& param)
{
   if (param.empty() || param == "X" || param == "x" || param == "+" || param == "!")
      return -1;

   try
   {
      return std::stoi(param);
   }
   catch (...)
   {
      return -1;
   }
}

inline bool IsAtTargetPosition(long currentPos, long targetPos, long tolerance)
{
   if (targetPos < 0)
      return false;

   long diff = currentPos - targetPos;
   if (diff < 0)
      diff = -diff;

   return diff <= tolerance;
}

inline std::string ExtractTag(const std::string& response)
{
   size_t delimPos = response.find(TAG_DELIMITER);
   if (delimPos == std::string::npos)
      return response;
   return response.substr(0, delimPos);
}

} // namespace IX5SSA
