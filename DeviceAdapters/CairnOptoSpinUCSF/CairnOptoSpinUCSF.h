///////////////////////////////////////////////////////////////////////////////
// FILE:       Conix.h
// PROJECT:    MicroManage
// SUBSYSTEM:  DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:
// Conix adapter
//                
// AUTHOR: Nico Stuurman, 02/27/2006
//		   Trevor Osborn (ConixXYStage, ConixZStage), trevor@conixresearch.com, 04/21/2010
//         

#ifndef _CONIX_H_
#define _CONIX_H_

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

enum ConixControllerType {UNKNOWN_CONTROLLER = 0, CONIX_XYZ_CONTROLLER, CONIX_RFA_CONTROLLER};


//////////////////////////////////////////////////////////////////////////////
// CairnOptospin class
//
//////////////////////////////////////////////////////////////////////////////

class CairnOptospin : public CStateDeviceBase<CairnOptospin>
{
public:
   CairnOptospin(DWORD devId);
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

private:
   int GetDevicePosition(int& position);
   int SetDevicePosition(int position);

   DWORD devId_;
   bool initialized_;
   long numPos_;
   // MMCore name of serial port
   // Command exchange with MMCore
   std::string command_;
	FT_HANDLE ftHandle_;
};

#endif //_CONIX_H_
