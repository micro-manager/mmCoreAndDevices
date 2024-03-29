/*
File:		MicroDriveXYStage.h
Copyright:	Mad City Labs Inc., 2023
License:	Distributed under the BSD license.
*/
#ifndef _MICRODRIVEXYSTAGE_H_
#define _MICRODRIVEXYSTAGE_H_

// MCL headers
#include "MicroDrive.h"
#include "MCL_MicroDrive.h"

// MM headers
#include "MMDevice.h"
#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "ModuleInterface.h"

// List headers
#include "handle_list_if.h"
#include "HandleListType.h"

#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_NOT_VALID_INPUT      104


class MicroDriveXYStage : public CXYStageBase<MicroDriveXYStage>
{
public:
   MicroDriveXYStage();
   ~MicroDriveXYStage();

   // Device Interface
   int Initialize();
   int Shutdown();
   bool Busy();
   void GetName(char* pszName) const;
     
   // XYStage API
   virtual double GetStepSize();
   virtual int SetPositionSteps(long x, long y);
   virtual int GetPositionSteps(long& x, long& y);
   virtual int Home();
   virtual int SetOrigin();
   virtual int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax);
   virtual int GetStepLimits(long &xMin, long &xMax, long &yMin, long &yMax);
   virtual double GetStepSizeXUm();
   virtual double GetStepSizeYUm();
   virtual int SetRelativePositionUm(double dx, double dy);
   virtual int SetPositionUm(double x, double y);
   virtual int IsXYStageSequenceable(bool& isSequenceable) const; 
   virtual int GetPositionUm(double& x, double& y);

   // Action interface
   int OnPositionXmm(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPositionYmm(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMoveXmm(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMoveYmm(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSetOriginHere(MM::PropertyBase* pProp, MM::ActionType eAct);  
   int OnCalibrate(MM::PropertyBase* pProp, MM::ActionType eAct);  
   int OnReturnToOrigin(MM::PropertyBase* pProp, MM::ActionType eAct);  
   int OnPositionXYmm(MM::PropertyBase* pProp, MM::ActionType eAct);  
   int OnVelocity(MM::PropertyBase* pProp, MM::ActionType eAct);  
   int OnEncoded(MM::PropertyBase* pProp, MM::ActionType eAct);  
   int OnIterativeMove(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnImRetry(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnImToleranceUm(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnIsTirfModuleAxis1(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnIsTirfModuleAxis2(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFindEpi(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnStop(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   // Initialization
   int CreateMicroDriveXYProperties();
   int InitDeviceAdapter();

   // Set/Get positions
   int SetPositionMmSync(double x, double y);
   int GetPositionMm(double& x, double& y);
   int SetRelativePositionMmSync(double x, double y);

   // Calibration & origin methods
   int CalibrateSync();
   int MoveToForwardLimitsSync();
   int ReturnToOriginSync();
   int FindEpiSync();
   int SetOriginSync();

   int ConvertRelativeToAbsoluteMm(double relX, double relY, double &absX, double &absY);

   // Pause devices
   void PauseDevice();
   int Stop();

   // Check if blocked
   bool XMoveBlocked(double possNewPos);
   bool YMoveBlocked(double possNewPos);

   void GetOrientation(bool& mirrorX, bool& mirrorY);

   // Threading
   int BeginMovementThread(int type, double distanceX, double distanceY);
   static DWORD WINAPI ExecuteMovement(LPVOID lpParam);   

   // Device Information
   int handle_;
   int serialNumber_;
   unsigned short pid_;
   int axis1_;
   int axis2_;
   unsigned char axisBitmap_;
   double stepSize_mm_;
   double encoderResolution_; 
   double maxVelocity_;
   double maxVelocityTwoAxis_;
   double maxVelocityThreeAxis_;
   double minVelocity_;
   double velocity_;
   // Device State
   bool initialized_;
   bool encoded_;
   double lastX_;
   double lastY_;
   // Iterative Move State
   bool iterativeMoves_;
   int imRetry_;
   double imToleranceUm_;
   // Tirf-Module State
   bool deviceHasTirfModuleAxis_;
   bool axis1IsTirfModule_;
   bool axis2IsTirfModule_;
   bool hasUnknownTirfModuleAxis_;
   double tirfModCalibrationMm_;
   // Threading
   bool stopCommanded_;
   int movementType_;
   double movementDistanceX_;
   double movementDistanceY_;
   HANDLE movementThread_;
   HANDLE threadStartMutex_;
};


#endif //_MICRODRIVEXYSTAGE_H_
