///////////////////////////////////////////////////////////////////////////////
// FILE:          QSICameraAdapter.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   QSI camera adapter for use with Micro-Manager
//            
// COPYRIGHT:     Quantum Scientific Imaging, Inc. 2024
//

#ifndef _QSICAMERAADAPTER_H_
#define _QSICAMERAADAPTER_H_

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "ModuleInterface.h"

#include "QSICameraCLib.h"
#include "QSIError.h"

class QSICameraAdapter : public CLegacyCameraBase<QSICameraAdapter>  
{
private:

  typedef CLegacyCameraBase<QSICameraAdapter> base;

public:

  /////////////////////////////////////////////////////////////////////////////
  // Constructor / Destructor
  //
  QSICameraAdapter();
  ~QSICameraAdapter();
  
  /////////////////////////////////////////////////////////////////////////////
  // MMDevice API
  //
  int Initialize();
  int Shutdown();
  void GetName( char * name ) const;      
   
  /////////////////////////////////////////////////////////////////////////////
  // Camera API
  //
  int ClearROI();
  unsigned int GetBitDepth() const;
  const unsigned char * GetImageBuffer();
  long GetImageBufferSize() const;
  int GetBinning() const;
  double GetExposure() const;
  unsigned int GetImageBytesPerPixel() const;
  unsigned int GetImageHeight() const;
  unsigned int GetImageWidth() const;
  double GetPixelSizeUm() const;
  int GetROI( unsigned int & x, unsigned int & y, unsigned int & xSize, unsigned int & ySize ); 
  int InsertImage();
  int IsExposureSequenceable( bool & seq ) const;
  int SetBinning( int binSize );
  void SetExposure( double exp );
  int SetROI( unsigned int x, unsigned int y, unsigned int xSize, unsigned int ySize ); 
  int SnapImage();
  int StartSequenceAcquisition( double interval );
  int ThreadRun();

  /////////////////////////////////////////////////////////////////////////////
  // Property setup and handling
  //
  int AntiBloomingPropertySetup();
  int AntiBloomingPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int BinningPropertiesSetup();
  int BinningPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int BodyTemperaturePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int BodyTemperaturePropertySetup();
  int CCDTemperaturePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int CCDTemperaturePropertySetup();
  int CCDTemperatureSetpointPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int CCDTemperatureSetpointPropertySetup();
  int DescriptionPropertySetup();
  int DriverInfoPropertySetup();
  int CoolerStatePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int CoolerStatePropertySetup();
  int CoolerPowerPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int CoolerPowerPropertySetup();
  int ExposurePropertiesSetup();
  int ExposurePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int FanModePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int FanModePropertySetup();
  int FilterWheelPositionPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int FilterWheelPropertiesSetup();
  int FullWellCapacityPropertySetup();
  int GainPropertiesSetup();
  int GainPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int GainModePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int LEDEnabledPropertySetup();
  int LEDEnabledPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int MaxADUPropertySetup();
  int ModelNamePropertySetup();
  int ModelNumberPropertySetup();
  int OpenShutterPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int OpenShutterPropertySetup();
  int PCBTemperaturePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int PCBTemperaturePropertySetup();
  int PixelSizePropertiesSetup();
  int PreExposureFlushPropertySetup();
  int PreExposureFlushPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int ReadoutModePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int ReadoutModePropertySetup();
  int SerialNumberPropertySetup();
  int ShutterPropertiesSetup();
  int ShutterPriorityPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );
  int SoundEnabledPropertySetup();
  int SoundEnabledPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct );

private:

#pragma region [ Private Members ]

  static const int QSI_IMAGE_BYTES_PER_PIXEL = 2;
  static const int QSI_IMAGE_BIT_DEPTH = QSI_IMAGE_BYTES_PER_PIXEL * 8;

  bool m_initialized;
  
  qsi_handle m_handle;
  qsi_status m_status;

  bool m_exposureOpenShutter;
  double m_exposureDuration;
  double m_exposureDurationMax;
  double m_exposureDurationMin;

  int m_imageBinning;
  int m_imageMaxX;
  int m_imageMaxY;
  int m_imageNumX;
  int m_imageNumY;
  int m_imageStartX;
  int m_imageStartY;
  unsigned short * m_pImageBuffer;

  double m_pixelSizeX;
  double m_pixelSizeY;

#pragma endregion

};

#endif //_QSICAMERAADAPTER_H_
