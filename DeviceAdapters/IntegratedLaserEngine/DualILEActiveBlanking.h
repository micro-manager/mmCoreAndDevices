///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILEActiveBlanking.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _DUALILEACTIVEBLANKING_H_
#define _DUALILEACTIVEBLANKING_H_

#include "Property.h"
#include <map>

class IALC_REV_ILE4;
class CIntegratedLaserEngine;
class CPortsConfiguration;

class CDualILEActiveBlanking
{
public:
  CDualILEActiveBlanking( IALC_REV_ILE4* DualActiveBlankingInterface, const CPortsConfiguration* PortsConfiguration, CIntegratedLaserEngine* MMILE );
  ~CDualILEActiveBlanking();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act, long PortIndex );
  typedef MM::ActionEx<CDualILEActiveBlanking> CPropertyActionEx;

  void UpdateILEInterface( IALC_REV_ILE4* DualActiveBlankingInterface );

private:
  IALC_REV_ILE4* DualActiveBlankingInterface_;
  const CPortsConfiguration* PortsConfiguration_;
  CIntegratedLaserEngine* MMILE_;
  std::vector<std::string> PortNames_;
  int Unit1EnabledPattern_;
  int Unit2EnabledPattern_;
  bool Unit1ActiveBlankingPresent_;
  bool Unit2ActiveBlankingPresent_;
  int Unit1NbLines_;
  int Unit2NbLines_;

  bool IsLineEnabledForSinglePort( int Unit, int Port ) const;
  bool IsLineEnabledForDualPort( const std::string& PortName ) const;
  void SetLineStateForSinglePort( int Unit, int Line, bool Enable );
  void SetLineStateForDualPort( const std::string& PortName, bool Enable );
  void LogSetActiveBlankingError( int Unit, const std::string& PortName, bool Enabling );
};

#endif