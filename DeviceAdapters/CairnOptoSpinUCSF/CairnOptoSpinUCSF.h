///////////////////////////////////////////////////////////////////////////////
// FILE:       CairnOptospin_UCSF.h
// PROJECT:    MicroManager
// SUBSYSTEM:  DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:
// CairnOptospin adapter, UCSF version
//                
// AUTHOR: Nico Stuurman, 09/25/2025
//         

#ifndef _CAIRN_OPTOSPIN_UCSF_H
#define _CAIRN_OPTOSPIN_UCSF_H

#include "MMDevice.h"
#include "DeviceBase.h"
#include "ftd2xx.h"
#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_UNKNOWN_COMMAND         10002
#define ERR_UNKNOWN_POSITION        10003
#define ERR_HALT_COMMAND            10004
#define ERR_UNRECOGNIZED_ANSWER     10005
#define ERR_PORT_CHANGE_FORBIDDEN   10006
#define ERR_OFFSET                  11000
#define ERR_OPTOSPIN_BUSY           11001
#define ERR_INVALID_WHEEL_NUMBER    11002
#define ERR_WHEEL_NOT_CONNECTED     11003
#define ERR_INVALID_POSITION        11004



//////////////////////////////////////////////////////////////////////////////
// CairnOptospin class
//
//////////////////////////////////////////////////////////////////////////////

class CairnHub : public HubBase<CairnHub>
{
public:
   CairnHub();
   ~CairnHub();

   // Device API
   // ----------
   int Initialize();
   int Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   int DetectInstalledDevices();

   int SetWheelPosition(long wheelNumber, long position);
   int GetWheelPosition(long wheelNumber, long& position);

   // action interface
   // ----------------
   int OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSoftwareVersion(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   bool initialized_;
	FT_HANDLE ftHandle_;
   std::string serialNumber_;
   long softwareVersion_; // Software version of the controller
   bool filterWheels_[4]; // Which of the 4 possible filter wheels are connected
};

class CairnOptospin : public CStateDeviceBase<CairnOptospin>
{
public:
   CairnOptospin(long wheelNumber);
   ~CairnOptospin();
  
   // Device API
   // ----------
   int Initialize();
   int Shutdown();
  
   void GetName(char* pszName) const;
   bool Busy();
   unsigned long GetNumberOfPositions()const {return numPos_;}


   // action interface
   // ----------------
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

   // pre-init properties
   int OnWheelNumber(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnWheelMode(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   bool initialized_;
   long numPos_;
   long serialNumber_; // Serial number of the controller, obtained through USB API
   long wheelNumber_; // Wheel number  (1 - 4)
   std::string wheelMode_; // Wheel mode (Independent or Synchronized)
};

#endif // _CairnOptospin_UCSF_H_
