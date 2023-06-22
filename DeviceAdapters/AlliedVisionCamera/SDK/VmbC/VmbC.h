/*=============================================================================
  Copyright (C) 2012 - 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        VmbC.h

-------------------------------------------------------------------------------

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, 
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED  
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

/**
 * \file
 * \brief Main header file for the VmbC API.
 * 
 * This file describes all necessary definitions for using Allied Vision's
 * VmbC API. These type definitions are designed to be portable from other
 * languages and other operating systems.
 *
 * General conventions:
 * - Method names are composed in the following manner:
 *    - Vmb"Action"                                        example: ::VmbStartup()
 *    - Vmb"Entity""Action" or Vmb"ActionTarget""Action"   example: ::VmbCameraOpen()
 *    - Vmb"Entity""SubEntity/ActionTarget""Action"        example: ::VmbFeatureCommandRun()
 * 
 * - Strings (generally declared as "const char *") are assumed to have a trailing 0 character
 * - All pointer parameters should of course be valid, except if stated otherwise.
 * - To ensure compatibility with older programs linked against a former version of the API,
 *   all struct* parameters have an accompanying sizeofstruct parameter.
 * - Functions returning lists are usually called twice: once with a zero buffer
 *   to get the length of the list, and then again with a buffer of the correct length.
 */

#ifndef VMBC_H_INCLUDE_
#define VMBC_H_INCLUDE_

#include <stddef.h>
#include <stdint.h>

#include <VmbC/VmbCommonTypes.h>
#include <VmbC/VmbCTypeDefinitions.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Timeout parameter signaling a blocking call.
 */
#define VMBINFINITE        0xFFFFFFFF

/**
 * \brief Define for the handle to use for accessing the Vmb system features cast to a given type
 *
 * Note: The primary purpose of this macro is the use in the VmbC sources.
 *       API users should use ::gVmbHandle instead.
 */
#define VMB_API_HANDLE(typeName) ((typeName)((((VmbUint64_t)1) << (VmbUint64_t)(sizeof(VmbHandle_t) * 8 - 4)) | ((VmbUint64_t) 1)))

/**
 * \brief Constant for the Vmb handle to be able to access Vmb system features.
 */
static const VmbHandle_t gVmbHandle = VMB_API_HANDLE(VmbHandle_t);

//===== FUNCTION PROTOTYPES ===================================================

/**
 * \defgroup Functions Vmb C API Functions
 * \{
 */

/**
 * \name API Version
 * \defgroup Version API Version
 * \{
 */

/**
 * \brief Retrieve the version number of VmbC.
 * 
 * This function can be called at anytime, even before the API is
 * initialized. All other version numbers may be queried via feature access.
 * 
 * \param[out]  versionInfo             Pointer to the struct where version information resides
 * \param[in]   sizeofVersionInfo       Size of structure in bytes
 * 
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback.
 * 
 * \retval ::VmbErrorStructSize         The given struct size is not valid for this version of the API
 *
 * \retval ::VmbErrorBadParameter       \p versionInfo is null.
 * 
 */
IMEXPORTC VmbError_t VMB_CALL VmbVersionQuery ( VmbVersionInfo_t*   versionInfo,
                                                VmbUint32_t         sizeofVersionInfo );
/**
 * \}
 */

/**
 * \name API Initialization
 * \{
 * \defgroup Init API Initialization
 * \{
 */

/**
 * \brief Initializes the VmbC API.
 * 
 * Note: This function must be called before any VmbC function other than ::VmbVersionQuery() is run.
 * 
 * \param[in]   pathConfiguration       A string containing a semicolon (Windows) or colon (other os) separated list of paths. The paths contain directories to search for .cti files,
 *                                      paths to .cti files and optionally the path to a configuration xml file. If null is passed the parameter is the cti files found in the paths
 *                                      the GENICAM_GENTL{32|64}_PATH environment variable are considered
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 * 
 * \retval ::VmbErrorAlready            This function was called before and call to ::VmbShutdown has been executed on a non-callback thread
 *
 * \retval ::VmbErrorInvalidCall        If called from a callback or ::VmbShutdown is currently running
 * 
 * \retval ::VmbErrorXml                If parsing the settings xml is unsuccessful; a missing default xml file does not result in this error.
 * 
 * \retval ::VmbErrorTLNotFound         A transport layer that was marked as required was not found.
 * 
 * \retval ::VmbErrorNoTL               No transport layer was found on the system; note that some of the transport layers may have been filtered out via the settings file.
 * 
 * \retval ::VmbErrorIO                 A log file should be written according to the settings xml file, but this log file could not be opened.
 * 
 * \retval ::VmbErrorBadParameter       \p pathConfiguration contains only separator and whitespace chars.
 * 
 */
IMEXPORTC VmbError_t VMB_CALL VmbStartup (const VmbFilePathChar_t* pathConfiguration);

/**
 * \brief Perform a shutdown of the API.
 * 
 * This frees some resources and deallocates all physical resources if applicable.
 * 
 * The call is silently ignored, if executed from a callback.
 * 
 */
IMEXPORTC void VMB_CALL VmbShutdown ( void );

/**
 * \} \}
 */

/**
 * \name Camera Enumeration & Information
 * \{
 * \defgroup CameraInfo Camera Enumeration & Information
 * \{
 */

/**
 * List all the cameras that are currently visible to the API.
 *
 * Note: This function is usually called twice: once with an empty array to query the length
 *       of the list, and then again with an array of the correct length.
 *       If camera lists change between the calls, numFound may deviate from the query return.
 *
 * \param[in,out]    cameraInfo             Array of VmbCameraInfo_t, allocated by the caller.
 *                                          The camera list is copied here. May be null.
 *
 * \param[in]        listLength             Number of entries in the callers cameraInfo array.
 *
 * \param[in,out]    numFound               Number of cameras found. Can be more than listLength.
 *
 * \param[in]        sizeofCameraInfo       Size of one VmbCameraInfo_t entry (if \p cameraInfo is null, this parameter is ignored).
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p numFound is null
 *
 * \retval ::VmbErrorStructSize         The given struct size is not valid for this API version and \p cameraInfo is not null
 *
 * \retval ::VmbErrorMoreData           The given list length was insufficient to hold all available entries
 */
IMEXPORTC VmbError_t VMB_CALL VmbCamerasList ( VmbCameraInfo_t*   cameraInfo,
                                               VmbUint32_t        listLength,
                                               VmbUint32_t*       numFound,
                                               VmbUint32_t        sizeofCameraInfo );

/**
 * \brief Retrieve information about a single camera given its handle.
 *
 * Note: Some information is only filled for opened cameras.
 *
 * \param[in]       cameraHandle            The handle of the camera; both remote and local device handles are permitted
 *
 * \param[in,out]   info                    Structure where information will be copied
 *
 * \param[in]       sizeofCameraInfo        Size of the structure
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorStructSize         The given struct size is not valid for this API version
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorBadParameter       \p info is null
 * 
 * \retval ::VmbErrorBadHandle          The handle does not correspond to a camera
 */
IMEXPORTC VmbError_t VMB_CALL VmbCameraInfoQueryByHandle(   VmbHandle_t         cameraHandle,
                                                            VmbCameraInfo_t*    info,
                                                            VmbUint32_t         sizeofCameraInfo);

/**
 * \brief Retrieve information about a single camera given the ID of the camera.
 *
 * Note: Some information is only filled for opened cameras.
 *
 * \param[in]       idString                ID of the camera
 *
 * \param[in,out]   info                    Structure where information will be copied
 *
 * \param[in]       sizeofCameraInfo        Size of the structure
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 * 
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p idString or \p info are null or \p idString is the empty string
 * 
 * \retval ::VmbErrorNotFound           No camera with the given id is found
 *
 * \retval ::VmbErrorStructSize         The given struct size is not valid for this API version
 */
IMEXPORTC VmbError_t VMB_CALL VmbCameraInfoQuery ( const char*         idString,
                                                   VmbCameraInfo_t*    info,
                                                   VmbUint32_t         sizeofCameraInfo );

/**
 * \brief Open the specified camera.
 * 
 * \param[in]   idString            ID of the camera.
 * \param[in]   accessMode          The desired access mode.
 * \param[out]  cameraHandle        The remote device handle of the camera, if opened successfully.
 * 
 * A camera may be opened in a specific access mode, which determines
 * the level of control you have on a camera.
 * Examples for idString:
 * 
 * "DEV_81237473991" for an ID given by a transport layer,
 * "169.254.12.13" for an IP address,
 * "000F314C4BE5" for a MAC address or 
 * "DEV_1234567890" for an ID as reported by Vmb
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInUse              The camera with the given ID is already opened
 *
 * \retval ::VmbErrorInvalidCall        If called from frame callback or chunk access callback
 *
 * \retval ::VmbErrorBadParameter       If \p idString or \p cameraHandle are null
 *
 * \retval ::VmbErrorInvalidAccess      A camera with the given id was found, but could not be opened
 * 
 * \retval ::VmbErrorNotFound           The designated camera cannot be found
 */
IMEXPORTC VmbError_t VMB_CALL VmbCameraOpen ( const char*      idString,
                                              VmbAccessMode_t  accessMode,
                                              VmbHandle_t*     cameraHandle );

/**
 * \brief Close the specified camera.
 * 
 * Depending on the access mode this camera was opened with, events are killed,
 * callbacks are unregistered, and camera control is released.
 * 
 * \param[in]   cameraHandle        A valid camera handle
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 * 
 * \retval ::VmbErrorInUse              The camera is currently in use with ::VmbChunkDataAccess
 * 
 * \retval ::VmbErrorBadHandle          The handle does not correspond to an open camera
 *
 * \retval ::VmbErrorInvalidCall        If called from frame callback or chunk access callback
 */
IMEXPORTC VmbError_t VMB_CALL VmbCameraClose ( const VmbHandle_t  cameraHandle );

/**
 * \} \}
 */

//----- Features ----------------------------------------------------------

/**
 * \name General Feature Functions
 * \{
 * \defgroup GeneralFeatures General Feature Functions
 * \{
 */

/**
 * \brief List all the features for this entity.
 * 
 * This function lists all implemented features, whether they are currently available or not.
 * The list of features does not change as long as the entity is connected.
 *
 * This function is usually called twice: once with an empty list to query the length
 * of the list, and then again with a list of the correct length.
 * 
 * If ::VmbErrorMoreData is returned and \p numFound is non-null, the total number of features has been written to \p numFound.
 * 
 * If there are more elements in \p featureInfoList than features available, the remaining elements
 * are filled with zero-initialized ::VmbFeatureInfo_t structs.
 * 
 * \param[in]   handle                  Handle for an entity that exposes features
 * \param[out]  featureInfoList         An array of ::VmbFeatureInfo_t to be filled by the API. May be null if \p numFund is used for size query.
 * \param[in]   listLength              Number of ::VmbFeatureInfo_t elements provided
 * \param[out]  numFound                Number of ::VmbFeatureInfo_t elements found. May be null if \p featureInfoList is not null.
 * \param[in]   sizeofFeatureInfo       Size of a ::VmbFeatureInfo_t entry
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 * 
 * \retval ::VmbErrorStructSize         The given struct size of ::VmbFeatureInfo_t is not valid for this version of the API
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       Both \p featureInfoList and \p numFound are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorMoreData           The given list length was insufficient to hold all available entries
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeaturesList ( VmbHandle_t         handle,
                                                VmbFeatureInfo_t*   featureInfoList,
                                                VmbUint32_t         listLength,
                                                VmbUint32_t*        numFound,
                                                VmbUint32_t         sizeofFeatureInfo );

/**
 * \brief Query information about the constant properties of a feature.
 * 
 * Users provide a pointer to ::VmbFeatureInfo_t, which is then set to the internal representation.
 * 
 * \param[in]   handle                  Handle for an entity that exposes features
 * \param[in]   name                    Name of the feature
 * \param[out]  featureInfo             The feature info to query
 * \param[in]   sizeofFeatureInfo       Size of the structure
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorStructSize         The given struct size of ::VmbFeatureInfo_t is not valid for this version of the API
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name or \p featureInfo are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 * 
 * \retval ::VmbErrorNotFound           A feature with the given name does not exist.
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureInfoQuery ( const VmbHandle_t   handle,
                                                    const char*         name,
                                                    VmbFeatureInfo_t*   featureInfo,
                                                    VmbUint32_t         sizeofFeatureInfo );

/**
 * \brief List all the features selected by a given feature for this module.
 *
 * This function lists all selected features, whether they are currently available or not.
 * Features with selected features ("selectors") have no direct impact on the camera,
 * but only influence the register address that selected features point to.
 * The list of features does not change while the camera/interface is connected.
 * This function is usually called twice: once with an empty array to query the length
 * of the list, and then again with an array of the correct length.
 * 
 * \param[in]   handle                  Handle for an entity that exposes features
 * \param[in]   name                    Name of the feature
 * \param[out]  featureInfoList         An array of ::VmbFeatureInfo_t to be filled by the API. May be null if \p numFound is used for size query.
 * \param[in]   listLength              Number of ::VmbFeatureInfo_t elements provided
 * \param[out]  numFound                Number of ::VmbFeatureInfo_t elements found. May be null if \p featureInfoList is not null.
 * \param[in]   sizeofFeatureInfo       Size of a ::VmbFeatureInfo_t entry
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 * 
 * \retval ::VmbErrorBadParameter       \p name is null or both \p featureInfoList and \p numFound are null
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorStructSize         The given struct size of ::VmbFeatureInfo_t is not valid for this version of the API
 *
 * \retval ::VmbErrorMoreData           The given list length was insufficient to hold all available entries
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureListSelected ( const VmbHandle_t  handle,
                                                       const char*        name,
                                                       VmbFeatureInfo_t*  featureInfoList,
                                                       VmbUint32_t        listLength,
                                                       VmbUint32_t*       numFound,
                                                       VmbUint32_t        sizeofFeatureInfo );

/**
 * \brief Return the dynamic read and write capabilities of this feature.
 * 
 * The access mode of a feature may change. For example, if "PacketSize"
 * is locked while image data is streamed, it is only readable.
 * 
 * \param[in]   handle              Handle for an entity that exposes features.
 * \param[in]   name                Name of the feature.
 * \param[out]  isReadable          Indicates if this feature is readable. May be null.
 * \param[out]  isWriteable         Indicates if this feature is writable. May be null.
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name is null or both \p isReadable and \p isWriteable are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureAccessQuery ( const VmbHandle_t   handle,
                                                      const char*         name,
                                                      VmbBool_t *         isReadable,
                                                      VmbBool_t *         isWriteable );

/**
 * \} \}
 */

/**
 * \name Integer Feature Access
 * \{
 * \defgroup IntAccess Integer Feature Access
 * \{
 */

/**
 * \brief Get the value of an integer feature.
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the feature
 * \param[out]  value       Value to get
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name or \p value are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Integer
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureIntGet ( const VmbHandle_t   handle,
                                                 const char*         name,
                                                 VmbInt64_t*         value );

/**
 * \brief Set the value of an integer feature.
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the feature
 * \param[in]   value       Value to set
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorInvalidCall        If called from feature callback
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       If \p name is null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Integer
 *
 * \retval ::VmbErrorInvalidAccess      The feature is unavailable or not writable
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 *
 * \retval ::VmbErrorInvalidValue       If value is either out of bounds or not an increment of the minimum
 * 
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureIntSet ( const VmbHandle_t   handle,
                                                 const char*         name,
                                                 VmbInt64_t          value );

/**
 * \brief Query the range of an integer feature.
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the feature
 * \param[out]  min         Minimum value to be returned. May be null.
 * \param[out]  max         Maximum value to be returned. May be null.
 * 
 * 
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       If \p name is null or both \p min and \p max are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature name is not Integer
 *
 * \retval ::VmbErrorInvalidAccess      The range information is unavailable or not writable
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureIntRangeQuery ( const VmbHandle_t   handle,
                                                        const char*         name,
                                                        VmbInt64_t*         min,
                                                        VmbInt64_t*         max );

/**
 * \brief Query the increment of an integer feature.
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the feature
 * \param[out]  value       Value of the increment to get.
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       If \p name or \p value are null
 * 
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 * 
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Integer
 *
 * \retval ::VmbErrorInvalidAccess      The information is unavailable or cannot be read
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureIntIncrementQuery ( const VmbHandle_t   handle,
                                                            const char*         name,
                                                            VmbInt64_t*         value );

/**
 * \brief Retrieves info about the valid value set of an integer feature.
 * 
 * Retrieves information about the set of valid values of an integer feature. If null is passed as buffer,
 * only the size of the set is determined and written to bufferFilledCount; Otherwise the largest possible
 * number of elements of the valid value set is copied to buffer.
 * 
 * \param[in]   handle                  The handle for the entity the feature information is retrieved from
 * \param[in]   name                    The name of the feature to retrieve the info for; if null is passed ::VmbErrorBadParameter is returned
 * \param[in]   buffer                  The array to copy the valid values to or null if only the size of the set is requested
 * \param[in]   bufferSize              The size of buffer; if buffer is null, the value is ignored
 * \param[out]  setSize                 The total number of elements in the set; the value is set, if ::VmbErrorMoreData is returned
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess                        The call was successful 
 *
 * \retval ::VmbErrorApiNotStarted                  ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter                   \p name is null or both \p buffer and \p bufferFilledCount are null
 *
 * \retval ::VmbErrorBadHandle                      The given handle is not valid
 *
 * \retval ::VmbErrorNotFound                       The feature was not found
 * 
 * \retval ::VmbErrorWrongType                      The type of the feature is not Integer
 *
 * \retval ::VmbErrorValidValueSetNotPresent        The feature does not provide a valid value set
 *
 * \retval ::VmbErrorMoreData                       Some of data was retrieved successfully, but the size of buffer is insufficient to store all elements
 *
 * \retval ::VmbErrorIncomplete                     The module the handle refers to is in a state where it cannot complete the request
 *
 * \retval ::VmbErrorOther                          Some other issue occurred
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureIntValidValueSetQuery(const VmbHandle_t   handle,
                                                              const char*         name,
                                                              VmbInt64_t*         buffer,
                                                              VmbUint32_t         bufferSize,
                                                              VmbUint32_t*        setSize);

/**
 * \} \}
 */

/**
 * \name Float Feature Access
 * \{
 * \defgroup FloatAccess Float Feature Access
 * \{
 */

/**
 * \brief Get the value of a float feature.
 * 
 * \param[in]   handle  Handle for an entity that exposes features
 * \param[in]   name    Name of the feature
 * \param[out]  value   Value to get
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful 
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name or \p value are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Float
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 * 
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureFloatGet ( const VmbHandle_t   handle,
                                                   const char*         name,
                                                   double*             value );

/**
 * \brief Set the value of a float feature.
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the feature
 * \param[in]   value       Value to set
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful 
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInvalidCall        If called from feature callback
 *
 * \retval ::VmbErrorBadParameter       \p name is null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Float
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 *
 * \retval ::VmbErrorInvalidValue       If value is not within valid bounds
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureFloatSet ( const VmbHandle_t   handle,
                                                   const char*         name,
                                                   double              value );

/**
 * \brief Query the range of a float feature.
 * 
 * Only one of the values may be queried if the other parameter is set to null,
 * but if both parameters are null, an error is returned.
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the feature
 * \param[out]  min         Minimum value to be returned. May be null.
 * \param[out]  max         Maximum value to be returned. May be null.
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful 
 *
 * \retval ::VmbErrorBadParameter       \p name is null or both \p min and \p max are null
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Float
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureFloatRangeQuery ( const VmbHandle_t   handle,
                                                          const char*         name,
                                                          double*             min,
                                                          double*             max );

/**
 * \brief Query the increment of a float feature.
 * 
 * \param[in]   handle              Handle for an entity that exposes features
 * \param[in]   name                Name of the feature
 * \param[out]  hasIncrement        `true` if this float feature has an increment.
 * \param[out]  value               Value of the increment to get.
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful 
 *
 * \retval ::VmbErrorBadParameter       \p name is null or both \p value and \p hasIncrement are null
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Float
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureFloatIncrementQuery ( const VmbHandle_t   handle,
                                                              const char*         name,
                                                              VmbBool_t*          hasIncrement,
                                                              double*             value );

/**
 * \} \}
*/

/**
 * \name Enum Feature Access
 * \{
 * \defgroup EnumAccess Enum Feature Access
 * \{
 */

/**
 * \brief Get the value of an enumeration feature.
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the feature
 * \param[out]  value       The current enumeration value. The returned value is a
 *                          reference to the API value
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful 
 *
 * \retval ::VmbErrorBadParameter       \p name or \p value are null
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature featureName is not Enumeration
 * 
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature is not available
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureEnumGet ( const VmbHandle_t   handle,
                                                  const char*         name,
                                                  const char**        value );

/**
 * \brief Set the value of an enumeration feature.
 *
 * \param[in] handle    Handle for an entity that exposes features
 * \param[in] name      Name of the feature
 * \param[in] value     Value to set
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInvalidCall        If called from feature callback
 *
 * \retval ::VmbErrorBadParameter       If \p name or \p value are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Enumeration
 *
 * \retval ::VmbErrorNotAvailable       The feature is not available
 * 
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 *
 * \retval ::VmbErrorInvalidValue       \p value is not a enum entry for the feature or the existing enum entry is currently not available
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureEnumSet ( const VmbHandle_t   handle,
                                                  const char*         name,
                                                  const char*         value );

/**
 * \brief Query the value range of an enumeration feature.
 * 
 * All elements not filled with the names of enum entries by the function are set to null.
 * 
 * \param[in]   handle          Handle for an entity that exposes features
 * \param[in]   name            Name of the feature
 * \param[out]  nameArray       An array of enumeration value names; may be null if \p numFound is used for size query
 * \param[in]   arrayLength     Number of elements in the array
 * \param[out]  numFound        Number of elements found; may be null if \p nameArray is not null
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name is null or both \p nameArray and \p numFound are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 * 
 * \retval ::VmbErrorNotImplemented     The feature \p name is not implemented
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Enumeration
 *
 * \retval ::VmbErrorMoreData           The given array length was insufficient to hold all available entries
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureEnumRangeQuery ( const VmbHandle_t   handle,
                                                         const char*         name,
                                                         const char**        nameArray,
                                                         VmbUint32_t         arrayLength,
                                                         VmbUint32_t*        numFound );

/**
 * \brief Check if a certain value of an enumeration is available.
 * 
 * \param[in]   handle              Handle for an entity that exposes features
 * \param[in]   name                Name of the feature
 * \param[in]   value               Value to check
 * \param[out]  isAvailable         Indicates if the given enumeration value is available
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name, \p value or \p isAvailable are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Enumeration
 * 
 * \retval ::VmbErrorNotImplemented     The feature \p name is not implemented
 *
 * \retval ::VmbErrorInvalidValue       There is no enum entry with string representation of \p value for the given enum feature
 * 
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureEnumIsAvailable ( const VmbHandle_t   handle,
                                                          const char*         name,
                                                          const char*         value,
                                                          VmbBool_t *         isAvailable );

/**
 * \brief Get the integer value for a given enumeration string value.
 * 
 * Converts a name of an enum member into an int value ("Mono12Packed" to 0x10C0006)
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the feature
 * \param[in]   value       The enumeration value to get the integer value for
 * \param[out]  intVal      The integer value for this enumeration entry
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       If \p name, \p value or \p intVal are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           No feature with the given name was found
 * 
 * \retval ::VmbErrorNotImplemented     The feature \p name is not implemented
 * 
 * \retval ::VmbErrorInvalidValue       \p value is not the name of a enum entry for the feature
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Enumeration
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureEnumAsInt ( const VmbHandle_t   handle,
                                                    const char*         name,
                                                    const char*         value,
                                                    VmbInt64_t*         intVal );

/**
 * \brief Get the enumeration string value for a given integer value.
 * 
 * Converts an int value to a name of an enum member (e.g. 0x10C0006 to "Mono12Packed")
 * 
 * \param[in]   handle              Handle for an entity that exposes features
 * \param[in]   name                Name of the feature
 * \param[in]   intValue            The numeric value
 * \param[out]  stringValue         The string value for the numeric value
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful 
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name or \p stringValue are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           No feature with the given name was found
 * 
 * \retval ::VmbErrorNotImplemented     No feature \p name is not implemented
 * 
 * \retval ::VmbErrorInvalidValue       \p intValue is not the int value of an enum entry
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Enumeration
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureEnumAsString ( VmbHandle_t   handle,
                                                       const char*   name,
                                                       VmbInt64_t    intValue,
                                                       const char**  stringValue );

/**
 * \brief Get infos about an entry of an enumeration feature.
 * 
 * \param[in]   handle                      Handle for an entity that exposes features
 * \param[in]   featureName                 Name of the feature
 * \param[in]   entryName                   Name of the enum entry of that feature
 * \param[out]  featureEnumEntry            Infos about that entry returned by the API
 * \param[in]   sizeofFeatureEnumEntry      Size of the structure
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorStructSize         Size of ::VmbFeatureEnumEntry_t is not compatible with the API version
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p featureName, \p entryName or \p featureEnumEntry are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 * 
 * \retval ::VmbErrorNotImplemented     The feature \p name is not implemented
 * 
 * \retval ::VmbErrorInvalidValue       There is no enum entry with a string representation of \p entryName
 *
 * \retval ::VmbErrorWrongType          The type of feature featureName is not Enumeration
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureEnumEntryGet ( const VmbHandle_t        handle,
                                                       const char*              featureName,
                                                       const char*              entryName,
                                                       VmbFeatureEnumEntry_t*   featureEnumEntry,
                                                       VmbUint32_t              sizeofFeatureEnumEntry );

/**
 * \} \}
 */

/**
 * \name String Feature Access
 * \{
 * \defgroup StringAccess String Feature Access
 * \{
 */

/**
 * \brief Get the value of a string feature.
 * 
 * This function is usually called twice: once with an empty buffer to query the length
 * of the string, and then again with a buffer of the correct length.
 *
 * The value written to \p sizeFilled includes the terminating 0 character of the string.
 * 
 * If a \p buffer is provided and there its  insufficient to hold all the data, the longest
 * possible prefix fitting the buffer is copied to \p buffer; the last element of \p buffer is
 * set to 0 case.
 * 
 * \param[in]   handle          Handle for an entity that exposes features
 * \param[in]   name            Name of the string feature
 * \param[out]  buffer          String buffer to fill. May be null if \p sizeFilled is used for size query.
 * \param[in]   bufferSize      Size of the input buffer
 * \param[out]  sizeFilled      Size actually filled. May be null if \p buffer is not null.
 *
 *
 * \return An error code indicating the type of error, if any.
 * 
 * \retval ::VmbErrorSuccess            The call was successful
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 * 
 * \retval ::VmbErrorBadParameter       \p name is null, both \p buffer and \p sizeFilled are null or \p buffer is non-null and bufferSize is 0
 * 
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 * 
 * \retval ::VmbErrorNotFound           The feature was not found
 * 
 * \retval ::VmbErrorWrongType          The type of feature \p name is not String
 * 
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 * 
 * \retval ::VmbErrorMoreData           The given buffer size was too small
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureStringGet ( const VmbHandle_t   handle,
                                                    const char*         name,
                                                    char*               buffer,
                                                    VmbUint32_t         bufferSize,
                                                    VmbUint32_t*        sizeFilled );

/**
 * \brief Set the value of a string feature.
 *
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the string feature
 * \param[in]   value       Value to set
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInvalidCall        If called from feature callback
 *
 * \retval ::VmbErrorBadParameter       \p name or \p value are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           The feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not String
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorInvalidValue       If length of value exceeded the maximum length
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureStringSet ( const VmbHandle_t   handle,
                                                    const char*         name,
                                                    const char*         value );

/**
 * \brief Get the maximum length of a string feature.
 * 
 * The length reported does not include the terminating 0 char.
 * 
 * Note: For some features the maximum size is not fixed and may change.
 *
 * \param[in]   handle          Handle for an entity that exposes features
 * \param[in]   name            Name of the string feature
 * \param[out]  maxLength       Maximum length of this string feature
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name or \p maxLength are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not String
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureStringMaxlengthQuery ( const VmbHandle_t   handle,
                                                               const char*         name,
                                                               VmbUint32_t*        maxLength );

/**
 * \} \}
 */

/**
 * \name Boolean Feature Access
 * \{
 * \defgroup BoolAccess Boolean Feature Access
 * \{
 */

/**
 * \brief Get the value of a boolean feature.
 *
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the boolean feature
 * \param[out]  value       Value to be read
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name or \p value are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           If feature is not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Boolean
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 * 
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureBoolGet ( const VmbHandle_t   handle,
                                                  const char*         name,
                                                  VmbBool_t *         value );

/**
 * \brief Set the value of a boolean feature.
 *
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the boolean feature
 * \param[in]   value       Value to write
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name is null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           If the feature is not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Boolean
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 *
 * \retval ::VmbErrorInvalidValue       If value is not within valid bounds
 *
 * \retval ::VmbErrorInvalidCall        If called from feature callback
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureBoolSet ( const VmbHandle_t   handle,
                                                  const char*         name,
                                                  VmbBool_t           value );

/**
 * \} \}
 */

/**
 * \name Command Feature Access
 * \{
 * \defgroup CmdAccess Command Feature Access
 * \{
 */

/**
 * \brief Run a feature command.
 *
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the command feature
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 * 
 * \retval ::VmbErrorInvalidCall        If called from a feature callback or chunk access callback
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name is null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           Feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Command
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureCommandRun ( const VmbHandle_t   handle,
                                                     const char*         name );

/**
 * \brief Check if a feature command is done.
 * 
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the command feature
 * \param[out]  isDone      State of the command.
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 * 
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorBadParameter       If \p name or \p isDone are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           Feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Command
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 * 
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureCommandIsDone ( const VmbHandle_t   handle,
                                                        const char*         name,
                                                        VmbBool_t *         isDone );

/**
 * \} \}
 */

/**
 * \name Raw Feature Access
 * \{
 * \defgroup RawAccess Raw Feature Access
 * \{
 */

/**
 * \brief Read the memory contents of an area given by a feature name.
 * 
 * This feature type corresponds to a top-level "Register" feature in GenICam.
 * Data transfer is split up by the transport layer if the feature length is too large.
 * You can get the size of the memory area addressed by the feature name by ::VmbFeatureRawLengthQuery().
 *
 * \param[in]   handle          Handle for an entity that exposes features
 * \param[in]   name            Name of the raw feature
 * \param[out]  buffer          Buffer to fill
 * \param[in]   bufferSize      Size of the buffer to be filled
 * \param[out]  sizeFilled      Number of bytes actually filled
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p name, \p buffer or \p sizeFilled are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           Feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Register
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureRawGet ( const VmbHandle_t   handle,
                                                 const char*         name,
                                                 char*               buffer,
                                                 VmbUint32_t         bufferSize,
                                                 VmbUint32_t*        sizeFilled );

/**
 * \brief Write to a memory area given by a feature name.
 *
 * This feature type corresponds to a first-level "Register" node in the XML file.
 * Data transfer is split up by the transport layer if the feature length is too large.
 * You can get the size of the memory area addressed by the feature name by ::VmbFeatureRawLengthQuery().
 *
 * \param[in]   handle          Handle for an entity that exposes features
 * \param[in]   name            Name of the raw feature
 * \param[in]   buffer          Data buffer to use
 * \param[in]   bufferSize      Size of the buffer
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInvalidCall        If called from feature callback or a chunk access callback
 *
 * \retval ::VmbErrorBadParameter       \p name or \p buffer are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           Feature was not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Register
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureRawSet ( const VmbHandle_t   handle,
                                                 const char*         name,
                                                 const char*         buffer,
                                                 VmbUint32_t         bufferSize );

/**
 * \brief Get the length of a raw feature for memory transfers.
 *
 * This feature type corresponds to a first-level "Register" node in the XML file.
 *
 * \param[in]   handle      Handle for an entity that exposes features
 * \param[in]   name        Name of the raw feature
 * \param[out]  length      Length of the raw feature area (in bytes)
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       If \p name or \p length are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           Feature not found
 *
 * \retval ::VmbErrorWrongType          The type of feature \p name is not Register
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorNotImplemented     The feature isn't implemented
 * 
 * \retval ::VmbErrorNotAvailable       The feature isn't available currently
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureRawLengthQuery ( const VmbHandle_t   handle,
                                                         const char*         name,
                                                         VmbUint32_t*        length );

/**
 * \} \}
 */

/**
 * \name Feature Invalidation
 * \{
 * \defgroup FeatureInvalidation Feature Invalidation
 * \{
 */

/**
 * \brief Register a VmbInvalidationCallback callback for feature invalidation signaling.
 *
 * Any feature change, either of its value or of its access state, may be tracked
 * by registering an invalidation callback.
 * Registering multiple callbacks for one feature invalidation event is possible because
 * only the combination of handle, name, and callback is used as key. If the same
 * combination of handle, name, and callback is registered a second time, the callback remains
 * registered and the context is overwritten with \p userContext.
 *
 * \param[in]   handle              Handle for an entity that emits events
 * \param[in]   name                Name of the event
 * \param[in]   callback            Callback to be run when invalidation occurs
 * \param[in]   userContext         User context passed to function
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 * 
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorBadParameter       If \p name or \p callback are null
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           No feature with \p name was found for the module associated with \p handle
 * 
 * \retval ::VmbErrorNotImplemented     The feature \p name is not implemented
 * 
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureInvalidationRegister ( VmbHandle_t              handle,
                                                               const char*              name,
                                                               VmbInvalidationCallback  callback,
                                                               void*                    userContext );

/**
 * \brief Unregister a previously registered feature invalidation callback.
 *
 * Since multiple callbacks may be registered for a feature invalidation event,
 * a combination of handle, name, and callback is needed for unregistering, too.
 *
 * \param[in] handle          Handle for an entity that emits events
 * \param[in] name            Name of the event
 * \param[in] callback        Callback to be removed
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 * 
 * \retval ::VmbErrorBadParameter       If \p name or \p callback are null
 * 
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorNotFound           No feature with \p name was found for the module associated with \p handle or there was no listener to unregister
 * 
 * \retval ::VmbErrorNotImplemented     The feature \p name is not implemented
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 */
IMEXPORTC VmbError_t VMB_CALL VmbFeatureInvalidationUnregister ( VmbHandle_t              handle,
                                                                 const char*              name,
                                                                 VmbInvalidationCallback  callback );

/**
 * \} \}
 */

/**
 * \name Image preparation and acquisition
 * \{
 * \defgroup Capture Image preparation and acquisition
 * \{
 */

/**
* \brief Get the necessary payload size for buffer allocation.
*
* Returns the payload size necessary for buffer allocation as queried from the Camera.
* If the stream module provides a PayloadSize feature, this value will be returned instead.
* If a camera handle is passed, the payload size refers to the stream with index 0.
*
* \param[in]    handle          Camera or stream handle
* \param[out]   payloadSize     Payload Size
*
*
* \return An error code indicating success or the type of error that occurred.
*
* \retval ::VmbErrorSuccess             If no error
*
* \retval ::VmbErrorApiNotStarted       ::VmbStartup() was not called before the current command
*
* \retval ::VmbErrorBadHandle           The given handle is not valid
*
* \retval ::VmbErrorBadParameter        \p payloadSize is null
*/
IMEXPORTC VmbError_t VMB_CALL VmbPayloadSizeGet(VmbHandle_t     handle,
                                                VmbUint32_t*    payloadSize);

/**
 * \brief Announce frames to the API that may be queued for frame capturing later.
 *
 * Allows some preparation for frames like DMA preparation depending on the transport layer.
 * The order in which the frames are announced is not taken into consideration by the API.
 * If frame.buffer is null, the allocation is done by the transport layer.
 *
 * \param[in]   handle          Camera or stream handle
 * \param[in]   frame           Frame buffer to announce
 * \param[in]   sizeofFrame     Size of the frame structure
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorStructSize         The given struct size is not valid for this version of the API
 * 
 * \retval ::VmbErrorInvalidCall        If called from a frame callback or a chunk access callback
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given camera handle is not valid
 *
 * \retval ::VmbErrorBadParameter       \p frame is null
 * 
 * \retval ::VmbErrorAlready            The frame has already been announced
 *
 * \retval ::VmbErrorBusy               The underlying transport layer does not support announcing frames during acquisition
 *
 * \retval ::VmbErrorMoreData           The given buffer size is invalid (usually 0)
 */
IMEXPORTC VmbError_t VMB_CALL VmbFrameAnnounce ( VmbHandle_t        handle,
                                                 const VmbFrame_t*  frame,
                                                 VmbUint32_t        sizeofFrame );


/**
 * \brief Revoke a frame from the API.
 *
 * The referenced frame is removed from the pool of frames for capturing images.
 *
 * \param[in]   handle      Handle for a camera or stream
 * \param[in]   frame       Frame buffer to be removed from the list of announced frames
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorInvalidCall        If called from a frame callback or a chunk access callback
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorBadParameter       The given frame pointer is not valid
 *
 * \retval ::VmbErrorBusy               The underlying transport layer does not support revoking frames during acquisition
 *
 * \retval ::VmbErrorNotFound           The given frame could not be found for the stream
 * 
 * \retval ::VmbErrorInUse              The frame is currently still in use (e.g. in a running frame callback)
 */
IMEXPORTC VmbError_t VMB_CALL VmbFrameRevoke ( VmbHandle_t          handle,
                                               const VmbFrame_t*    frame );


/**
 * \brief Revoke all frames assigned to a certain stream or camera.
 * 
 * In case of an failure some of the frames may have been revoked. To prevent this it is recommended to call
 * ::VmbCaptureQueueFlush for the same handle before invoking this function.
 *
 * \param[in]   handle      Handle for a stream or camera
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorInvalidCall        If called from a frame callback or a chunk access callback
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          \p handle is not valid
 * 
 * \retval ::VmbErrorInUse              One of the frames of the stream is still in use
 */
IMEXPORTC VmbError_t VMB_CALL VmbFrameRevokeAll ( VmbHandle_t  handle );


/**
 * \brief Prepare the API for incoming frames.
 *
 * \param[in]   handle      Handle for a camera or a stream
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess                    If no error
 *
 * \retval ::VmbErrorInvalidCall                If called from a frame callback or a chunk access callback
 * 
 * \retval ::VmbErrorApiNotStarted              ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle                  The given handle is not valid; this includes the camera no longer being open
 *
 * \retval ::VmbErrorInvalidAccess              Operation is invalid with the current access mode
 * 
 * \retval ::VmbErrorMoreData                   The buffer size of the announced frames is insufficient
 *
 * \retval ::VmbErrorInsufficientBufferCount    The operation requires more buffers to be announced; see the StreamAnnounceBufferMinimum stream feature
 *
 * \retval ::VmbErrorAlready                    Capturing was already started
 */
IMEXPORTC VmbError_t VMB_CALL VmbCaptureStart ( VmbHandle_t  handle );


/**
 * \brief Stop the API from being able to receive frames.
 *
 * Consequences of VmbCaptureEnd():
 * The frame callback will not be called anymore
 *
 * \note This function waits for the completion of the last callback for the current capture.
 *       If the callback does not return in finite time, this function may not return in finite time either.
 * 
 * \param[in]   handle      Handle for a stream or camera
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorInvalidCall        If called from a frame callback or a chunk access callback
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          \p handle is not valid
 */
IMEXPORTC VmbError_t VMB_CALL VmbCaptureEnd ( VmbHandle_t handle );


/**
 * \brief Queue frames that may be filled during frame capturing.
 *
 * The given frame is put into a queue that will be filled sequentially.
 * The order in which the frames are filled is determined by the order in which they are queued.
 * If the frame was announced with ::VmbFrameAnnounce() before, the application
 * has to ensure that the frame is also revoked by calling ::VmbFrameRevoke() or
 * ::VmbFrameRevokeAll() when cleaning up.
 * 
 * \warning \p callback should to return in finite time. Otherwise ::VmbCaptureEnd and
 *          operations resulting in the stream being closed may not return.
 *
 * \param[in]   handle              Handle of a camera or stream
 * \param[in]   frame               Pointer to an already announced frame
 * \param[in]   callback            Callback to be run when the frame is complete. Null is OK.
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 * 
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 * 
 * \retval ::VmbErrorBadParameter       If \p frame is null
 * 
 * \retval ::VmbErrorBadHandle          No stream related to \p handle could be found
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 * 
 * \retval ::VmbErrorInternalFault      The buffer or bufferSize members of \p frame have been set to null or zero respectively
 * 
 * \retval ::VmbErrorNotFound           The frame is not a frame announced for the given stream
 * 
 * \retval ::VmbErrorAlready            The frame is currently queued
 */
IMEXPORTC VmbError_t VMB_CALL VmbCaptureFrameQueue ( VmbHandle_t        handle,
                                                     const VmbFrame_t*  frame,
                                                     VmbFrameCallback   callback );

/**
 * \brief Wait for a queued frame to be filled (or dequeued).
 * 
 * The frame needs to be queued and not filled for the function to complete successfully.
 * 
 * If a camera handle is passed, the first stream of the camera is used.
 *
 * \param[in]   handle          Handle of a camera or stream
 * \param[in]   frame           Pointer to an already announced and queued frame
 * \param[in]   timeout         Timeout (in milliseconds)
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 * 
 * \retval ::VmbErrorBadParameter       If \p frame or the buffer of \p frame are null or the the buffer size of \p frame is 0
 *
 * \retval ::VmbErrorBadHandle          No stream related to \p handle could be found
 *
 * \retval ::VmbErrorNotFound           The frame is not one currently queued for the stream 
 * 
 * \retval ::VmbErrorAlready            The frame has already been dequeued or VmbCaptureFrameWait has been called already for this frame
 * 
 * \retval ::VmbErrorInUse              If the frame was queued with a frame callback
 * 
 * \retval ::VmbErrorTimeout            Call timed out
 * 
 * \retval ::VmbErrorIncomplete         Capture is not active when the function is called
 */
IMEXPORTC VmbError_t VMB_CALL VmbCaptureFrameWait ( const VmbHandle_t   handle,
                                                    const VmbFrame_t*   frame,
                                                    VmbUint32_t         timeout);


/**
 * \brief Flush the capture queue.
 *
 * Control of all the currently queued frames will be returned to the user,
 * leaving no frames in the capture queue.
 * After this call, no frame notification will occur until frames are queued again
 * 
 * Frames need to be revoked separately, if desired.
 * 
 * This function can only succeeds, if no capture is currently active.
 * If ::VmbCaptureStart has been called for the stream, but no successful call to ::VmbCaptureEnd
 * happened, the function fails with error code ::VmbErrorInUse.
 *
 * \param[in]   handle  The handle of the camera or stream to flush.
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorBadHandle          No stream related to \p handle could be found.
 * 
 * \retval ::VmbErrorInUse              There is currently an active capture
 */
IMEXPORTC VmbError_t VMB_CALL VmbCaptureQueueFlush(VmbHandle_t handle);

/**
 * \} \}
 */

/**
 * \name Transport Layer Enumeration & Information
 * \{
 * \defgroup TransportLayer Transport Layer Enumeration & Information
 * \{
 */
 
/**
 * \brief List all the transport layers that are used by the API.
 *
 * Note: This function is usually called twice: once with an empty array to query the length
 *       of the list, and then again with an array of the correct length.
 *
 * \param[in,out]   transportLayerInfo              Array of VmbTransportLayerInfo_t, allocated by the caller.
 *                                                  The transport layer list is copied here. May be null.
 * \param[in]       listLength                      Number of entries in the caller's transportLayerInfo array.
 * \param[in,out]   numFound                        Number of transport layers found. May be more than listLength.
 * \param[in]       sizeofTransportLayerInfo        Size of one ::VmbTransportLayerInfo_t entry (ignored if \p transportLayerInfo is null).
 * 
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 * 
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 * 
 * \retval ::VmbErrorInternalFault      An internal fault occurred
 *
 * \retval ::VmbErrorNotImplemented     One of the transport layers does not provide the required information
 *
 * \retval ::VmbErrorBadParameter       \p numFound is null
 *
 * \retval ::VmbErrorStructSize         The given struct size is not valid for this API version
 *
 * \retval ::VmbErrorMoreData           The given list length was insufficient to hold all available entries
 */
IMEXPORTC VmbError_t VMB_CALL VmbTransportLayersList ( VmbTransportLayerInfo_t*   transportLayerInfo,
                                                       VmbUint32_t                listLength,
                                                       VmbUint32_t*               numFound,
                                                       VmbUint32_t                sizeofTransportLayerInfo);

/**
 * \} \}
*/

/**
 * \name Interface Enumeration & Information
 * \{
 * \defgroup Interface Interface Enumeration & Information
 * \{
 */

/**
 * \brief List all the interfaces that are currently visible to the API.
 *
 * Note: All the interfaces known via GenICam transport layers are listed by this 
 *       command and filled into the provided array. Interfaces may correspond to 
 *       adapter cards or frame grabber cards.
 *       This function is usually called twice: once with an empty array to query the length
 *       of the list, and then again with an array of the correct length.
 *
 * \param[in,out]   interfaceInfo           Array of ::VmbInterfaceInfo_t, allocated by the caller.
 *                                          The interface list is copied here. May be null.
 *
 * \param[in]       listLength              Number of entries in the callers interfaceInfo array
 *
 * \param[in,out]   numFound                Number of interfaces found. Can be more than listLength
 *
 * \param[in]       sizeofInterfaceInfo     Size of one ::VmbInterfaceInfo_t entry
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            The call was successful
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter       \p numFound is null
 *
 * \retval ::VmbErrorStructSize         The given struct size is not valid for this API version
 *
 * \retval ::VmbErrorMoreData           The given list length was insufficient to hold all available entries
 */
IMEXPORTC VmbError_t VMB_CALL VmbInterfacesList ( VmbInterfaceInfo_t*   interfaceInfo,
                                                  VmbUint32_t           listLength,
                                                  VmbUint32_t*          numFound,
                                                  VmbUint32_t           sizeofInterfaceInfo );

/**
 * \} \}
 */

/**
 * \name Direct Access
 * \{
 * \defgroup DirectAccess Direct Access
 * \{
 */

//----- Memory/Register access --------------------------------------------

/**
 * \brief Read an array of bytes.
 *
 * \param[in]   handle              Handle for an entity that allows memory access
 * \param[in]   address             Address to be used for this read operation
 * \param[in]   bufferSize          Size of the data buffer to read
 * \param[out]  dataBuffer          Buffer to be filled
 * \param[out]  sizeComplete        Size of the data actually read
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 */
IMEXPORTC VmbError_t VMB_CALL VmbMemoryRead ( const VmbHandle_t   handle,
                                              VmbUint64_t         address,
                                              VmbUint32_t         bufferSize,
                                              char*               dataBuffer,
                                              VmbUint32_t*        sizeComplete );

/**
 * \brief Write an array of bytes.
 *
 * \param[in]   handle              Handle for an entity that allows memory access
 * \param[in]   address             Address to be used for this read operation
 * \param[in]   bufferSize          Size of the data buffer to write
 * \param[in]   dataBuffer          Data to write
 * \param[out]  sizeComplete        Number of bytes successfully written; if an
 *                                  error occurs this is less than bufferSize
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 *
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorMoreData           Not all data were written; see sizeComplete value for the number of bytes written
 */
IMEXPORTC VmbError_t VMB_CALL VmbMemoryWrite ( const VmbHandle_t   handle,
                                               VmbUint64_t         address,
                                               VmbUint32_t         bufferSize,
                                               const char*         dataBuffer,
                                               VmbUint32_t*        sizeComplete );

/**
 * \} \}
 */

/**
 * \name Load & Save Settings
 * \{
 * \defgroup LoadSaveSettings Load & Save Settings
 * \{
 */

/**
 * \brief Write the current features related to a module to a xml file
 *
 * Camera must be opened beforehand and function needs corresponding handle.
 * With given filename parameter path and name of XML file can be determined.
 * Additionally behaviour of function can be set with providing 'persistent struct'.
 *
 * \param[in]   handle              Handle for an entity that allows register access
 * \param[in]   filePath            The path to the file to save the settings to; relative paths are relative to the current working directory
 * \param[in]   settings            Settings struct; if null the default settings are used
 *                                  (persist features except LUT for the remote device, maximum 5 iterations, logging only errors)
 * \param[in]   sizeofSettings      Size of settings struct
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 * 
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 *
 * \retval ::VmbErrorBadParameter       If \p filePath is or the settings struct is invalid
 * 
 * \retval ::VmbErrorStructSize         If sizeofSettings the struct size does not match the size of the struct expected by the API
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 * 
 * \retval ::VmbErrorNotFound           The provided handle is insufficient to identify all the modules that should be saved
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 * 
 * \retval ::VmbErrorIO                 There was an issue writing the file.
 */
IMEXPORTC VmbError_t VMB_CALL VmbSettingsSave(VmbHandle_t                           handle,
                                              const VmbFilePathChar_t*              filePath,
                                              const VmbFeaturePersistSettings_t*    settings,
                                              VmbUint32_t                           sizeofSettings);

/**
 * \brief Load all feature values from xml file to device-related modules.
 *
 * The modules must be opened beforehand. If the handle is non-null it must be a valid handle other than the Vmb API handle.
 * Additionally behaviour of function can be set with providing \p settings . Note that even in case of an failure some or all of the features
 * may have been set for some of the modules.
 *
 * The error code ::VmbErrorRetriesExceeded only indicates that the number of retries was insufficient
 * to restore the features. Even if the features could not be restored for one of the modules, restoring the features is not aborted but the process
 * continues for other modules, if present.
 *
 * \param[in]   handle              Handle related to the modules to write the values to;
 *                                  may be null to indicate that modules should be identified based on the information provided in the input file
 *
 * \param[in]   filePath            The path to the file to load the settings from; relative paths are relative to the current working directory
 * \param[in]   settings            Settings struct; pass null to use the default settings. If the \p maxIterations field is 0, the number of
 *                                  iterations is determined by the value loaded from the xml file
 * \param[in]   sizeofSettings      Size of the settings struct
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess            If no error
 * 
 * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
 * 
 * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
 * 
 * \retval ::VmbErrorStructSize         If sizeofSettings the struct size does not match the size of the struct expected by the API
 *
 * \retval ::VmbErrorWrongType          \p handle is neither null nor a transport layer, interface, local device, remote device or stream handle
 *
 * \retval ::VmbErrorBadHandle          The given handle is not valid
 * 
 * \retval ::VmbErrorAmbiguous          The modules to restore the settings for cannot be uniquely identified based on the information available
 * 
 * \retval ::VmbErrorNotFound           The provided handle is insufficient to identify all the modules that should be restored
 * 
 * \retval ::VmbErrorRetriesExceeded    Some or all of the features could not be restored with the max iterations specified
 *
 * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
 *
 * \retval ::VmbErrorBadParameter       If \p filePath is null or the settings struct is invalid
 * 
 * \retval ::VmbErrorIO                 There was an issue with reading the file.
 */
IMEXPORTC VmbError_t VMB_CALL VmbSettingsLoad(VmbHandle_t                           handle,
                                              const VmbFilePathChar_t*              filePath,
                                              const VmbFeaturePersistSettings_t*    settings,
                                              VmbUint32_t                           sizeofSettings);

/**
 * \} \}
 */

/**
 * \name Chunk Data
 * \{
 * \defgroup ChunkData Chunk Data
 * \{
 */

/**
 * \brief Access chunk data for a frame.
 *
 * This function can only succeed if the given frame has been filled by the API.
 *
 * \param[in] frame                  A pointer to a filled frame that is announced
 * \param[in] chunkAccessCallback    A callback to access the chunk data from
 * \param[in] userContext            A pointer to pass to the callback
 *
 *
 * \return An error code indicating success or the type of error that occurred.
 *
 * \retval ::VmbErrorSuccess                The call was successful
 *
 * \retval ::VmbErrorInvalidCall            If called from a chunk access callback or a feature callback
 *
 * \retval ::VmbErrorApiNotStarted          ::VmbStartup() was not called before the current command
 *
 * \retval ::VmbErrorBadParameter           \p frame or \p chunkAccessCallback are null
 *
 * \retval ::VmbErrorInUse                  The frame state does not allow for retrieval of chunk data
 *                                          (e.g. the frame could have been reenqueued before the chunk access could happen).
 *
 * \retval ::VmbErrorNotFound               The frame is currently not announced for a stream
 * 
 * \retval ::VmbErrorDeviceNotOpen          If the device the frame was received from is no longer open
 *
 * \retval ::VmbErrorNoChunkData            \p frame does not contain chunk data
 *
 * \retval ::VmbErrorParsingChunkData       The chunk data does not adhere to the expected format
 *
 * \retval ::VmbErrorUserCallbackException  The callback threw an exception
 *
 * \retval ::VmbErrorFeaturesUnavailable    The feature description for the remote device is unavailable
 *
 * \retval ::VmbErrorCustom                 The minimum a user defined error code returned by the callback
 */
IMEXPORTC VmbError_t VMB_CALL VmbChunkDataAccess(const VmbFrame_t*         frame,
                                                 VmbChunkAccessCallback    chunkAccessCallback,
                                                 void*                     userContext);

/**
 * \} \} \}
 */
#ifdef __cplusplus
}
#endif

#endif // VMBC_H_INCLUDE_
