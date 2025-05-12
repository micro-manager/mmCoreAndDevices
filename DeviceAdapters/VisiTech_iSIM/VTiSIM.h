// Micro-Manager device adapter for VisiTech iSIM
//
// Copyright (C) 2016 Open Imaging, Inc.
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation; version 2.1.
//
// This library is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
// for more details.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
//
// Author: Mark Tsuchida <mark@open-imaging.com>

// Version 2.2.0.0	-	1. Implemented PIFOC SGS 100µm controls	-	17/08/2020
// Version 2.2.1.0	-	1. FRAP function revisit				-	18/08/2020
// Version 2.2.2.0	-	1. FRAP function 2nd visit 				-	10/09/2020
// Version 2.3.0.0	-	1. FRAP - add FRAP range controls		-	23/09/2020
// Version 2.3.1.0	-	1. more FRAP...modulation switching		-	05/10/2020
// Version 2.3.2.0	-	1. FRAP offset							-	05/10/2020
// Version 2.4.0.0	-	1. Add different Pifoc Devices			-	24/09/2020
//					-	2. Save Picfoc position in dll			-	24/09/2020
//					-	3. Add version number in dll			-	24/09/2020
// Version 2.5.0.0	-	1. Moved to new source code				-	12/03/2020
#pragma once

#include "DeviceBase.h"


class VTiSIMHub : public HubBase<VTiSIMHub>
{
public:
   VTiSIMHub();
   virtual ~VTiSIMHub();

   virtual int Initialize();
   virtual int Shutdown();
   virtual void GetName(char* name) const;
   virtual bool Busy();

   virtual int DetectInstalledDevices();

public:
   HANDLE GetAOTFHandle() { return hAotfControl_; }
   HANDLE GetScanAndMotorHandle() { return hScanAndMotorControl_; }

private:
   HANDLE hAotfControl_;
   HANDLE hScanAndMotorControl_;
};


class VTiSIMLaserShutter : public CShutterBase<VTiSIMLaserShutter>
{
public:
   VTiSIMLaserShutter();
   virtual ~VTiSIMLaserShutter();

   virtual int Initialize();
   virtual int Shutdown();
   virtual void GetName(char* name) const;
   virtual bool Busy();

   virtual int GetOpen(bool& open);
   virtual int SetOpen(bool open);
   virtual int Fire(double) { return DEVICE_UNSUPPORTED_COMMAND; }

private:
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   VTiSIMHub* VTiHub();
   int DoSetOpen(bool open);

private:
   bool isOpen_;
};


class VTiSIMLasers : public CStateDeviceBase<VTiSIMLasers>
{
   static const int nChannels = 8;
   static const int nModType = 3;
  
public:
   VTiSIMLasers();
   virtual ~VTiSIMLasers();

   virtual int Initialize();
   virtual int Shutdown();
   virtual void GetName(char* name) const;
   virtual bool Busy();
   virtual unsigned long GetNumberOfPositions() const;

private:
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct, long chan);
   int OnModulation(MM::PropertyBase* pProp, MM::ActionType eAct); // Laser Modulation
   int OnLaserState(MM::PropertyBase* pProp, MM::ActionType eAct,long channel); // To allow to turn on  mutliple laser at the same time. 
   int OnLaserNameState (MM::PropertyBase* pProp, MM::ActionType eAct,long channel);// To allow to set the laser Name
private:
   VTiSIMHub* VTiHub();
   int DoSetChannel(int chan);
   int DoSetIntensity(int chan, int percentage);
   int DoSetModulation(int mode);
   int DoUploadTTLBitmask(int channel , int ShouldOn); // To allow to turn on/Off  mutliple laser at the same time. 


private:
   int curChan_;
   int intensities_[nChannels];
   int Bitmask;
   int LaserName[8];
   std::string strLaserName[8];
 
};
class VTiSIMScanner : public CGenericBase<VTiSIMScanner>
{
public:
   VTiSIMScanner();
   virtual ~VTiSIMScanner();

   virtual int Initialize();
   virtual int Shutdown();
   virtual void GetName(char* name) const;
   virtual bool Busy();

private:
   int OnScanRate(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanWidth(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnScanOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnStartStop(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnActualScanRate(MM::PropertyBase* pProp, MM::ActionType eAct);
 
private:
   VTiSIMHub* VTiHub();
   int DoSetScanRate(int rateHz);
   int DoSetScanWidth(int width);
   int DoSetScanOffset(int offset);
   int DoStartStopScan(bool shouldScan);
   int DoGetScanning(bool& scanning);
   int DoSetScanOffsetPolarity( bool Polarity);

   int GetMaxOffset() const
   { return (maxWidth_ - scanWidth_) / 2; }

private:
   LONG minRate_, maxRate_;
   LONG minWidth_, maxWidth_;

   int scanRate_;
   int scanWidth_;
   int scanOffset_;
   bool scanOffsetPolarity_;
   float actualRate_;
};


class VTiSIMPinholeArray : public CGenericBase<VTiSIMPinholeArray>
{
   static const int nSizes = 7;

public:
   VTiSIMPinholeArray();
   virtual ~VTiSIMPinholeArray();

   virtual int Initialize();
   virtual int Shutdown();
   virtual void GetName(char* name) const;
   virtual bool Busy();

private:
   int OnFinePosition(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPinholeSize(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBacklashCompensation(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   VTiSIMHub* VTiHub();
   int DoGetPinholePositions(int* positions);
   int DoSetFinePosition(int position, int backlashComp = 0);
   int GetNearestPinholeIndex(int finePosition) const;
   int GetPinholeSizeUmForIndex(int index) const;
   int GetPinholeSizeIndex(int sizeUm) const;
   int ClipFinePositionToMotorRange(int finePosition) const;

private:
   int pinholePositions_[nSizes];
   LONG minFinePosition_, maxFinePosition_;
   int curFinePosition_;
   int backlashCompensation_;
};

class VTiSIMDichroic : public CStateDeviceBase<VTiSIMDichroic>
{
static const int nDSizes = 3;

public:
   VTiSIMDichroic();
   virtual ~VTiSIMDichroic();

   virtual int Initialize();
   virtual int Shutdown();
   virtual void GetName(char* name) const;
   virtual bool Busy();
   
   unsigned long GetNumberOfPositions()const {return nDSizes;}
   
private:
   int OnDichroicPosition(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnDichroicPos(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
 
private:
   VTiSIMHub* VTiHub();
   int DoMotorMove(int Pos);

private:
   LONG minFinePosition_, maxFinePosition_;
   LONG curDichroicPosition_;
   int DichroicPositions[3];
};


class VTiSIMBarrierFilter : public CStateDeviceBase<VTiSIMBarrierFilter>
{
	static const int nSizes = 6;

public:
   VTiSIMBarrierFilter();
   virtual ~VTiSIMBarrierFilter();

   virtual int Initialize();
   virtual int Shutdown();
   virtual void GetName(char* name) const;
   virtual bool Busy();
   
   unsigned long GetNumberOfPositions()const {return nSizes;}

private:
   int OnBarrierFilterPosition(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFilterPos(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   VTiSIMHub* VTiHub();
   int DoMotorMove(int Pos);

private:
   LONG minFinePosition_, maxFinePosition_;
   LONG curFilterPosition_;
   LONG FilterPositions[6];

};

// Ver 2.1.0.0 - FRAP - start
class VTiSIMFRAP : public CGalvoBase<VTiSIMFRAP>
{
public:
   VTiSIMFRAP();
   virtual ~VTiSIMFRAP();
   virtual int Initialize();
   virtual int Shutdown();
   virtual void GetName(char* name) const;
   virtual bool Busy();

   // Ver 2.3.0.0 - Start
   private:
   int OnYRange(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnYMin(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnXRange(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnXMin(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnXOffset(MM::PropertyBase* pProp, MM::ActionType eAct); // Ver 2.3.2.0	
   int OnYOffset(MM::PropertyBase* pProp, MM::ActionType eAct); // Ver 2.3.2.0
   // Ver 2.3.0.0 - End
private:
    // Galvo API
   int PointAndFire(double x, double y, double pulseTime_us);
   int SetSpotInterval(double pulseTime_us);
   int SetPosition(double x, double y);
   int GetPosition(double& x, double& y);
   int SetIlluminationState(bool on);
   int AddPolygonVertex(int polygonIndex, double x, double y);
   int DeletePolygons();
   int LoadPolygons();
   int SetPolygonRepetitions(int repetitions);
   int RunPolygons();
   int RunSequence();
   int StopSequence();
   int GetChannel(char* channelName);
   double GetXRange();
   double GetYRange();
   double GetXMinimum();
   double GetYMinimum();

private:
   //int DriveXDiection(MM::PropertyBase* pProp, MM::ActionType eAct);
  // int DriveYDiection(MM::PropertyBase* pProp, MM::ActionType eAct);
   //int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	VTiSIMHub* VTiHub();
	LONG minFinePosition_, maxFinePosition_;
	double x_;
	double y_;
	 long polygonRepetitions_;
	std::vector<std::vector<std::pair<double,double> > > polygons_; //18062020
	 // std::vector<std::pair<double,double> >  polygons_;
	  // vector< pair<double,double> > polygons_;
	long long_x;
	long long_y;
	unsigned short Position_x;
	unsigned short Position_y;
	long long_pulseTime_us;
	long long_repetitions;

	// Ver 2.2.1.0 - start
	long BleachingRegionLeft;
	long BleachingRegionRight;
	long BleachingRegionTop;
	long BleachingRegionBottom;

	bool FirstVertex ;

	bool FRAPEnable;

	int FRAPYRange_;
	int FRAPYMin_;
	int FRAPXRange_;
	int FRAPXMin_;

	int DoSetFRAPYRange(int yRange);
	int DoSetFRAPYMin(int yMin);
	int DoSetFRAPXRange(int xRange);
	int DoSetFRAPXMin(int yMax);
	// Ver 2.2.1.0 - End 

	// Ver 2.3.2.0 - Start
	int FRAPXOffset_;
	int FRAPYOffset_;

	int GetMaxXOffset() const
		//{ return (4095 - Position_x) / 2; }
		{ return 1000 / 2; }
	int GetMaxYOffset() const
		//{ return (4095 - Position_y) / 2; }
		{ return 1000 / 2; }
	int DoSetFRAPXOffset(int XOffset);
	int DoSetFRAPYOffset(int YOffset);
	// Ver 2.3.2.0 - End
};

// Ver 2.1.0.0 - FRAP - End

// Ver 2.2.0.0 - Pifoc - Start
class VTiSIMPifoc : public CStageBase<VTiSIMPifoc>
{
	public:
		VTiSIMPifoc();
		virtual ~VTiSIMPifoc();

		// Device API
		// ----------
		virtual int Initialize();
		virtual int Shutdown();
		virtual void GetName(char* name) const;
		virtual bool Busy();

		// Stage API
		// ---------
		int SetPositionUm(double pos);
		int GetPositionUm(double& pos);
		int SetPositionSteps(long steps);
		int GetPositionSteps(long& steps);
		int SetOrigin();
		int GetLimits(double& min, double& max);

		int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }
		bool IsContinuousFocusDrive() const { return false; }

		// action interface
		// ----------------
		int OnControllerName(MM::PropertyBase* pProp, MM::ActionType eAct);

		int OnStepSizeUm(MM::PropertyBase* pProp, MM::ActionType eAct);
		int OnAxisName(MM::PropertyBase* pProp, MM::ActionType eAct);
		int OnAxisLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
		int OnAxisTravelRange(MM::PropertyBase* pProp, MM::ActionType eAct);

		int OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct);
		int OnStageType(MM::PropertyBase* pProp, MM::ActionType eAct);
		int OnHoming(MM::PropertyBase* pProp, MM::ActionType eAct);
		int OnVelocity(MM::PropertyBase* pProp, MM::ActionType eAct);

		// Ver 2.4.0.0 - Start
		int OnPiezoTravelRangeUm(MM::PropertyBase* pProp, MM::ActionType eAct);
		// Ver 2.4.0.0 - End 
	private:
	VTiSIMHub* VTiHub();
		std::string axisName_;
		double stepSizeUm_;
		bool initialized_;
		double axisLimitUm_;
		bool invertTravelRange_;
		std::string stageType_;
		std::string controllerName_;
		unsigned short PositionSteps;
		double PosNm;
		//PIController* ctrl_;
	// Ver 2.4.0.0 - Start
	double PiezoTravelRangeUm;
	LONG minTravelRangeUm_;
	LONG maxTravelRangeUm_;
	long Steps_;
	int NmPosition_;
 

	// Ver 2.4.0.0 - End
};

//// Ver 2.2.0.0 - Pifoc - End