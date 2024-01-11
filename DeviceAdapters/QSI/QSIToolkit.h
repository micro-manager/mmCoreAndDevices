#pragma once
#ifndef _QSITOOLKIT_H_
#define _QSITOOLKIT_H_

#include <string>

#include "QSICameraAdapter.h"
#include "QSICameraCLib.h"

#pragma region [ Macros ]

///////////////////////////////////////////////////////////////////////////////
//
#define HANDLE_QSI_ERROR(cameraAdapterReference,errorCode) \
  do \
  { \
    if( (errorCode) ) \
    { \
      (cameraAdapterReference)->LogMessage( GetLastQSIError( m_handle ), false );  \
      return (errorCode); \
    } \
  } \
  while( false )

///////////////////////////////////////////////////////////////////////////////
//
#define HANDLE_MM_ERROR(cameraAdapterReference,errorCode,errorMessage) \
  do \
  { \
    if( (errorCode) ) \
    { \
      (cameraAdapterReference)->LogMessage( (errorMessage), false ); \
      return (errorCode); \
    } \
  } \
  while( false )

#pragma endregion

#pragma region [ Global Methods ]

///////////////////////////////////////////////////////////////////////////////
//
std::string GetLastQSIError( qsi_handle handle )
{
  char errorStringArray[QSI_LENGTH_ERROR_STRING];
  qsi_status response;

  memset( errorStringArray, 0, QSI_LENGTH_ERROR_STRING ); // Not really necessary but it never hurts

  response = QSI_GetLastError( handle, errorStringArray, QSI_LENGTH_ERROR_STRING );

  if( response == QSI_ERROR_INVALID_HANDLE )
    return std::string( "Unable to get last error due to bad handle" );

  return std::string( errorStringArray, QSI_LENGTH_ERROR_STRING );
}

#pragma endregion

#endif //_QSITOOLKIT_H_