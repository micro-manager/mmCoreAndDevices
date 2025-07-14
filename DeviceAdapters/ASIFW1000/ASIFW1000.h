///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIFW1000.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASIFW1000  controller adapter
// COPYRIGHT:     University of California, San Francisco, 2006
//                All rights reserved
//
// LICENSE:       This library is free software; you can redistribute it and/or
//                modify it under the terms of the GNU Lesser General Public
//                License as published by the Free Software Foundation.
//                
//                You should have received a copy of the GNU Lesser General Public
//                License along with the source distribution; if not, write to
//                the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
//                Boston, MA  02111-1307  USA
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// AUTHOR:        Nico Stuurman (nico@cmp.ucsf.edu) based on code by Nenad Amodaj, April 2007
//                additions by Jon Daniels (ASI) June 2019
//

#ifndef _ASIFW1000_H_
#define _ASIFW1000_H_

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//

// from Nico's code
#define ERR_UNKNOWN_COMMAND          10002
#define ERR_UNKNOWN_POSITION         10003
#define ERR_HALT_COMMAND             10004
#define ERR_CANNOT_CHANGE_PROPERTY   10005
// eof from Nico's code

// from Prior
#define ERR_INVALID_STEP_SIZE        10006
#define ERR_INVALID_MODE             10008
#define ERR_UNRECOGNIZED_ANSWER      10009
#define ERR_UNSPECIFIED_ERROR        10010

#define ERR_OFFSET 10100

// From ASIFW1000HUb
#define ERR_NOT_CONNECTED           11002
#define ERR_COMMAND_CANNOT_EXECUTE  11003
#define ERR_NO_ANSWER               11004
#define ERR_SETTING_WHEEL           11005
#define ERR_SETTING_VERBOSE_LEVEL   11006
#define ERR_SHUTTER_NOT_FOUND       11007
#define ERR_UNEXPECTED_ANSWER       11008


// Use the name 'return_value' that is unlikely to appear within 'result'.
#define RETURN_ON_MM_ERROR( result ) do { \
   int return_value = (result); \
   if (return_value != DEVICE_OK) { \
      return return_value; \
   } \
} while (0)

const char* const g_SerialTerminatorFW = "\n\r";

class Hub : public CGenericBase<Hub>
{
public:
   Hub();
   ~Hub();
  
   // Device API
   // ----------
   int Initialize();
   int Shutdown();
  
   void GetName(char* pszName) const;
   bool Busy();
   bool SupportsDeviceDetection(void);
   MM::DeviceDetectionStatus DetectDevice(void);

   // action interface
   // ----------------
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   bool initialized_;
   // MMCore name of serial port
   std::string port_;
};

class Shutter : public CShutterBase<Shutter>
{
public:
   Shutter();
   ~Shutter();

   int Initialize();
   int Shutdown();

   void GetName (char* pszName) const;
   bool Busy();
   unsigned long GetShutterNr() const {return shutterNr_;}

   // Shutter API
   int SetOpen (bool open = true);
   int GetOpen(bool& open);
   int Fire(double deltaT);

   // action interface
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnType(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnShutterNr(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   bool initialized_;
   std::string name_;
   std::string shutterType_;
   MM::MMTime changedTime_;
   unsigned shutterNr_;
};

class FilterWheel : public CStateDeviceBase<FilterWheel>
{
public:
   FilterWheel();
   ~FilterWheel();
 
   // MMDevice API                                                           
   // ------------                                                           
   int Initialize();                                                         
   int Shutdown();                                                           
                                                                             
   void GetName(char* pszName) const;                                        
   bool Busy();                                                              
   unsigned long GetNumberOfPositions()const {return numPos_;}               
   unsigned long GetWheelNr() const {return wheelNr_;}
                                                                             
   // action interface                                                       
   // ----------------                                                       
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);                
   int OnWheelNr(MM::PropertyBase* pProp, MM::ActionType eAct);                
   int OnSpeedSetting(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSerialCommand(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSerialResponse(MM::PropertyBase* pProp, MM::ActionType eAct);
                                                                             
private:                                                                     
   bool open_;
   bool initialized_;                                                        
   std::string name_;  
   long  pos_;
   int wheelNr_;
   int numPos_;
   std::string manualSerialAnswer_; // last answer received when the SerialCommand property was used
};

class FilterWheelSA : public CStateDeviceBase<FilterWheelSA>
{
public:
   FilterWheelSA();
   ~FilterWheelSA();

   // MMDevice API
   // ------------
   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();
   unsigned long GetNumberOfPositions()const {return numPos_;}
   unsigned long GetWheelNr() const {return wheelNr_;}

   // action interface
   // ----------------
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnWheelNr(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSpeedSetting(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSerialCommand(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSerialResponse(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   std::string port_;
   bool initialized_;
   std::string name_;
   int pos_;
   int wheelNr_;
   int numPos_;
   std::string serialAnswer_;
   std::string serialCommand_;     // the last command sent, or can be set for calling commands without args
   std::string manualSerialAnswer_; // last answer received when the SerialCommand property was used
   MMThreadLock threadLock_;  // used to lock thread during serial transaction

   int ClearComPort();
   int QueryCommand(const char *command);
   int QueryCommand(const std::string &command)
         { return QueryCommand(command.c_str()); }
   int QueryCommandVerify(const char *command, const char *expectedReplyPrefix);
   int QueryCommandVerify(const std::string &command, const std::string &expectedReplyPrefix)
         { return QueryCommandVerify(command.c_str(), expectedReplyPrefix.c_str()); }
   int ParseAnswerAfterPosition(unsigned int pos, int &val);
   int SelectWheel();

};


#endif //_ASIFW1000_H_
