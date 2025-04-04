///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_RVM.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Advanced MicroFluidics (AMF) Rotary
//				  Valve Modules (RVM).
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

extern const char* AMF_START;
extern const char* AMF_END;
extern const char* AMF_TERM;
extern const char AMF_ACK;
extern const char AMF_NACK;

enum {
    ERR_NO_ERROR = 10000,
    ERR_INITIALIZATION,
    ERR_INVALID_COMMAND,
    ERR_INVALID_OPERAND,
    ERR_MISSING_TRAILING_R,
    ERR_DEVICE_NOT_INIT,
    ERR_VALVE_FAILURE,
    ERR_PLUNGER_OVERLOAD,
    ERR_VALVE_OVERLOAD,
    ERR_MOVE_NOT_ALLOWED,
    ERR_PLUNGER_FAILURE,
    ERR_AD_CONVERTER_FAILURE,
    ERR_COMMAND_OVERFLOW,
    ERR_UNEXPECTED_RESPONSE
};

class AMFCommand
{
public:
    AMFCommand(unsigned int address) :
        address_(address),
        responsesParsed_(0),
        status_('@')
    {}

    virtual ~AMFCommand() {}

    std::string Get()
    {
        return std::string(AMF_START) + GetAddressChar() + GetCommandString();
    }

    int ParseResponse(const std::string& response)
    {
        debug_ = response;
        status_ = response[3];
        int ret = ParseStatus(status_);
        if (ret == DEVICE_OK) {
            std::string content = response.substr(4, response.size() - 5);
            ret = ParseContent(content);
        }
        return DEVICE_OK;
    }

    std::string GetDebug() { return debug_; }

protected:
    unsigned int address_;
    int responsesParsed_;
    char status_;
    std::string debug_;
    
    char GetAddressChar() {
        return (address_ < 10) ? address_ + '0' : address_ - 10 + 'A';
    }

    int ParseDecimal(const std::string& s, int maxNDigits, int& result)
    {
        if (s.empty() || s.size() > (unsigned int)maxNDigits)
            return ERR_UNEXPECTED_RESPONSE;
        for (unsigned int i = 0; i < s.size(); ++i)
        {
            char c = s[i];
            if (c < '0' || c > '9')
                return ERR_UNEXPECTED_RESPONSE;
        }
        result = std::atoi(s.c_str());
        return DEVICE_OK;
    }

    std::string FormatDecimal(int /* nDigits */, int value)
    {
        // It turns out that there is no need to zero-pad to a fixed number of
        // digits, despite what the manual may seem to imply.
        char buf[16];
        snprintf(buf, 15, "%d", value);
        return std::string(buf);
    }

    virtual int ParseStatus(char status) {
        switch (status) {
        case '@': case '`':
            return DEVICE_OK;
            break;
        case 'A': case 'a':
            return ERR_INITIALIZATION;
            break;
        case 'B': case 'b':
            return ERR_INVALID_COMMAND;
            break;
        case 'C': case 'c':
            return ERR_INVALID_OPERAND;
            break;
        case 'D': case 'd':
            return ERR_MISSING_TRAILING_R;
            break;
        case 'G': case 'g':
            return ERR_DEVICE_NOT_INIT;
            break;
        case 'H': case 'h':
            return ERR_VALVE_FAILURE;
            break;
        case 'I': case 'i':
            return ERR_PLUNGER_OVERLOAD;
            break;
        case 'J': case 'j':
            return ERR_VALVE_OVERLOAD;
            break;
        case 'K': case 'k':
            return ERR_MOVE_NOT_ALLOWED;
            break;
        case 'L': case 'l':
            return ERR_PLUNGER_FAILURE;
            break;
        case 'N': case 'n':
            return ERR_AD_CONVERTER_FAILURE;
            break;
        case 'O': case 'o':
            return ERR_COMMAND_OVERFLOW;
            break;
        default:
            return ERR_UNEXPECTED_RESPONSE;
            break;
        }
    }

    virtual std::string GetCommandString() = 0;
    virtual int ParseContent(const std::string& content) = 0;
};

class InitializationCommand : public AMFCommand
{
public:
    InitializationCommand(unsigned int address) :
        AMFCommand(address)
    {}

protected:
    std::string GetCommandString() { return "ZR"; }
    int ParseContent(const std::string& content) {
        if (!content.empty())
            return ERR_UNEXPECTED_RESPONSE;
        else
            return DEVICE_OK;
    }
};

class ValvePositionCommand : public AMFCommand
{
    unsigned int position_;
    char direction_;

public:
    ValvePositionCommand(unsigned int address, unsigned int position, char direction) :
        AMFCommand(address)
    {
        position_ = position + 1; // AMF firmware is 1-indexed
        direction_ = direction;
    }

protected:
    std::string GetCommandString()
    {
        std::string commandString = direction_ + std::to_string(position_) + AMF_END;
        return commandString;
    }

    int ParseContent(const std::string& content)
    {
        if (!content.empty())
            return ERR_UNEXPECTED_RESPONSE;
        else
            return DEVICE_OK;
    }
};

class ValvePositionRequest : public AMFCommand
{
    int positionOneBased_;

protected:
    std::string GetCommandString() { return "?6"; }

    int ParseContent(const std::string& content)
    {
        return ParseDecimal(content, 2, positionOneBased_);
    }

public:
    ValvePositionRequest(unsigned int address) :
        AMFCommand(address),
        positionOneBased_(0)
    {}

    int GetPosition() { return positionOneBased_ - 1; }
};

class ValveMaxPositionsRequest : public AMFCommand
{
private:
    int maxPositions_ = 0;

protected:
    std::string GetCommandString() { return "?801"; }
    int ParseContent(const std::string& content)
    {
        return ParseDecimal(content, 2, maxPositions_);;
    }

public:
    ValveMaxPositionsRequest(unsigned int address) :
        AMFCommand(address),
        maxPositions_(0)
    {}

    int GetMaxPositions() { return maxPositions_; }
};

class InstrumentStatusRequest : public AMFCommand
{
public:
    InstrumentStatusRequest(unsigned int address) :
        AMFCommand(address)
    {}

protected:
    std::string GetCommandString() { return "Q"; }
    int ParseContent(const std::string& content)
    {
        if (!content.empty()) {
            return ERR_UNEXPECTED_RESPONSE;
        }
        else {
            return DEVICE_OK;
        }
    }

    char GetStatus() { return status_; }
};

class FirmwareVersionRequest : public AMFCommand
{
    std::string version_;

protected:
    std::string GetCommandString() { return "?23"; }

    int ParseContent(const std::string& content)
    {
        if (content.empty())
            return ERR_UNEXPECTED_RESPONSE;
        version_ = content;
        return DEVICE_OK;
    }

public:
    FirmwareVersionRequest(unsigned int address) :
        AMFCommand(address)
    {}

    std::string GetFirmwareVersion() { return version_; }
};

#endif //_AMF_COMMANDS_H_