/*=============================================================================
  Copyright (C) 2012 - 2021 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------
 
  File:        VmbCTypeDefinitions.h

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
 * \brief Struct definitions for the VmbC API.
 */

#ifndef VMBC_TYPE_DEFINITIONS_H_INCLUDE_
#define VMBC_TYPE_DEFINITIONS_H_INCLUDE_

#include <stddef.h>
#include <stdint.h>

#include <VmbC/VmbCommonTypes.h>

#if defined (_WIN32)
#if defined AVT_VMBAPI_C_EXPORTS                // DLL exports
#define IMEXPORTC                               // We export via the .def file
#elif defined AVT_VMBAPI_C_LIB                  // static LIB
#define IMEXPORTC
#else                                           // import
#define IMEXPORTC __declspec(dllimport)
#endif

#ifndef _WIN64
 // Calling convention
#define VMB_CALL __stdcall
#else
 // Calling convention
#define VMB_CALL
#endif
#elif defined (__GNUC__) && (__GNUC__ >= 4) && defined (__ELF__)
 // SO exports (requires compiler option -fvisibility=hidden)
#ifdef AVT_VMBAPI_C_EXPORTS
#define IMEXPORTC __attribute__((visibility("default")))
#else
#define IMEXPORTC
#endif

#ifdef __i386__
    // Calling convention
#define VMB_CALL __attribute__((stdcall))
#else
    // Calling convention
#define VMB_CALL
#endif
#elif defined (__APPLE__)
#define IMEXPORTC __attribute__((visibility("default")))
 // Calling convention
#define VMB_CALL
#else
#error Unknown platform, file needs adaption
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \name Transport layer
 * \{
 */

 /**
* \brief Camera or transport layer type (for instance U3V or GEV).
*/
typedef enum VmbTransportLayerType
{
    VmbTransportLayerTypeUnknown        = 0,        //!< Interface is not known to this version of the API
    VmbTransportLayerTypeGEV            = 1,        //!< GigE Vision
    VmbTransportLayerTypeCL             = 2,        //!< Camera Link
    VmbTransportLayerTypeIIDC           = 3,        //!< IIDC 1394
    VmbTransportLayerTypeUVC            = 4,        //!< USB video class
    VmbTransportLayerTypeCXP            = 5,        //!< CoaXPress
    VmbTransportLayerTypeCLHS           = 6,        //!< Camera Link HS
    VmbTransportLayerTypeU3V            = 7,        //!< USB3 Vision Standard
    VmbTransportLayerTypeEthernet       = 8,        //!< Generic Ethernet
    VmbTransportLayerTypePCI            = 9,        //!< PCI / PCIe
    VmbTransportLayerTypeCustom         = 10,       //!< Non standard
    VmbTransportLayerTypeMixed          = 11,       //!< Mixed (transport layer only)
} VmbTransportLayerType;

/**
    * \brief Type for an Interface; for values see ::VmbTransportLayerType.
    */
typedef VmbUint32_t VmbTransportLayerType_t;

/**
 * \brief Transport layer information.
 * 
 * Holds read-only information about a transport layer.
 */
typedef struct VmbTransportLayerInfo
{
    /**
     * \name Out
     * \{
     */

    const char*                 transportLayerIdString;     //!< Unique id of the transport layer
    const char*                 transportLayerName;         //!< Name of the transport layer
    const char*                 transportLayerModelName;    //!< Model name of the transport layer
    const char*                 transportLayerVendor;       //!< Vendor of the transport layer
    const char*                 transportLayerVersion;      //!< Version of the transport layer
    const char*                 transportLayerPath;         //!< Full path of the transport layer
    VmbHandle_t                 transportLayerHandle;       //!< Handle of the transport layer for feature access
    VmbTransportLayerType_t     transportLayerType;         //!< The type of the transport layer

    /**
     * \}
     */
} VmbTransportLayerInfo_t;

/**
 * \}
 */

/**
 * \name Interface
 * \{
 */

/**
 * \brief Interface information.
 * 
 * Holds read-only information about an interface.
 */
typedef struct VmbInterfaceInfo
{
    /**
     * \name Out
     * \{
     */

    const char*                 interfaceIdString;          //!< Identifier of the interface
    const char*                 interfaceName;              //!< Interface name, given by the transport layer
    VmbHandle_t                 interfaceHandle;            //!< Handle of the interface for feature access
    VmbHandle_t                 transportLayerHandle;       //!< Handle of the related transport layer for feature access
    VmbTransportLayerType_t     interfaceType;              //!< The technology of the interface

    /**
     * \}
     */
} VmbInterfaceInfo_t;

/**
 * \}
 */

/**
 * \name Camera
 * \{
 */

 /**
  * \brief Access mode for cameras.
  *
  * Used in ::VmbCameraInfo_t as flags, so multiple modes can be
  * announced, while in ::VmbCameraOpen(), no combination must be used.
  */
typedef enum VmbAccessModeType
{
    VmbAccessModeNone       = 0,    //!< No access
    VmbAccessModeFull       = 1,    //!< Read and write access
    VmbAccessModeRead       = 2,    //!< Read-only access
    VmbAccessModeUnknown    = 4,    //!< Access type unknown
    VmbAccessModeExclusive  = 8,    //!< Read and write access without permitting access for other consumers
} VmbAccessModeType;

/**
 * \brief Type for an AccessMode; for values see ::VmbAccessModeType.
 */
typedef VmbUint32_t VmbAccessMode_t;

/**
 * \brief Camera information.
 * 
 * Holds read-only information about a camera.
 */
typedef struct VmbCameraInfo
{
    /**
     * \name Out
     * \{
     */

    const char*         cameraIdString;             //!< Identifier of the camera
    const char*         cameraIdExtended;           //!< globally unique identifier for the camera
    const char*         cameraName;                 //!< The display name of the camera
    const char*         modelName;                  //!< Model name
    const char*         serialString;               //!< Serial number
    VmbHandle_t         transportLayerHandle;       //!< Handle of the related transport layer for feature access
    VmbHandle_t         interfaceHandle;            //!< Handle of the related interface for feature access
    VmbHandle_t         localDeviceHandle;          //!< Handle of the related GenTL local device. NULL if the camera is not opened
    VmbHandle_t const*  streamHandles;              //!< Handles of the streams provided by the camera.  NULL if the camera is not opened
    VmbUint32_t         streamCount;                //!< Number of stream handles in the streamHandles array
    VmbAccessMode_t     permittedAccess;            //!< Permitted access modes, see ::VmbAccessModeType

    /**
     * \}
     */
} VmbCameraInfo_t;

/**
 * \}
 */

/**
 * \name Feature
 * \{
 */

/**
 * \brief Supported feature data types.
 */
typedef enum VmbFeatureDataType
{
    VmbFeatureDataUnknown       = 0,        //!< Unknown feature type
    VmbFeatureDataInt           = 1,        //!< 64-bit integer feature
    VmbFeatureDataFloat         = 2,        //!< 64-bit floating point feature
    VmbFeatureDataEnum          = 3,        //!< Enumeration feature
    VmbFeatureDataString        = 4,        //!< String feature
    VmbFeatureDataBool          = 5,        //!< Boolean feature
    VmbFeatureDataCommand       = 6,        //!< Command feature
    VmbFeatureDataRaw           = 7,        //!< Raw (direct register access) feature
    VmbFeatureDataNone          = 8,        //!< Feature with no data
} VmbFeatureDataType;

/**
 * \brief Data type for a Feature; for values see ::VmbFeatureDataType.
 */
typedef VmbUint32_t VmbFeatureData_t;

/**
 * \brief Feature visibility.
 */
typedef enum VmbFeatureVisibilityType
{
    VmbFeatureVisibilityUnknown         = 0,        //!< Feature visibility is not known
    VmbFeatureVisibilityBeginner        = 1,        //!< Feature is visible in feature list (beginner level)
    VmbFeatureVisibilityExpert          = 2,        //!< Feature is visible in feature list (expert level)
    VmbFeatureVisibilityGuru            = 3,        //!< Feature is visible in feature list (guru level)
    VmbFeatureVisibilityInvisible       = 4,        //!< Feature is visible in the feature list, but should be hidden in GUI applications
} VmbFeatureVisibilityType;

/**
 * \brief Type for Feature visibility; for values see ::VmbFeatureVisibilityType.
 */
typedef VmbUint32_t VmbFeatureVisibility_t;

/**
 * \brief Feature flags.
 */
typedef enum VmbFeatureFlagsType
{
    VmbFeatureFlagsNone             = 0,        //!< No additional information is provided
    VmbFeatureFlagsRead             = 1,        //!< Static info about read access. Current status depends on access mode, check with ::VmbFeatureAccessQuery()
    VmbFeatureFlagsWrite            = 2,        //!< Static info about write access. Current status depends on access mode, check with ::VmbFeatureAccessQuery()
    VmbFeatureFlagsVolatile         = 8,        //!< Value may change at any time
    VmbFeatureFlagsModifyWrite      = 16,       //!< Value may change after a write
} VmbFeatureFlagsType;

/**
 * \brief Type for Feature flags; for values see ::VmbFeatureFlagsType.
 */
typedef VmbUint32_t VmbFeatureFlags_t;

/**
 * \brief Feature information.
 *
 * Holds read-only information about a feature.
 */
typedef struct VmbFeatureInfo
{
    /**
     * \name Out
     * \{
     */

    const char*                 name;                       //!< Name used in the API
    const char*                 category;                   //!< Category this feature can be found in
    const char*                 displayName;                //!< Feature name to be used in GUIs
    const char*                 tooltip;                    //!< Short description, e.g. for a tooltip
    const char*                 description;                //!< Longer description
    const char*                 sfncNamespace;              //!< Namespace this feature resides in
    const char*                 unit;                       //!< Measuring unit as given in the XML file
    const char*                 representation;             //!< Representation of a numeric feature
    VmbFeatureData_t            featureDataType;            //!< Data type of this feature
    VmbFeatureFlags_t           featureFlags;               //!< Access flags for this feature
    VmbUint32_t                 pollingTime;                //!< Predefined polling time for volatile features
    VmbFeatureVisibility_t      visibility;                 //!< GUI visibility
    VmbBool_t                   isStreamable;               //!< Indicates if a feature can be stored to / loaded from a file
    VmbBool_t                   hasSelectedFeatures;        //!< Indicates if the feature selects other features

    /**
     * \}
     */
} VmbFeatureInfo_t;

/**
 * \brief Info about possible entries of an enumeration feature.
 */
typedef struct VmbFeatureEnumEntry
{
    /**
     * \name Out
     * \{
     */

    const char*                 name;               //!< Name used in the API
    const char*                 displayName;        //!< Enumeration entry name to be used in GUIs
    const char*                 tooltip;            //!< Short description, e.g. for a tooltip
    const char*                 description;        //!< Longer description
    VmbInt64_t                  intValue;           //!< Integer value of this enumeration entry
    const char*                 sfncNamespace;      //!< Namespace this feature resides in
    VmbFeatureVisibility_t      visibility;         //!< GUI visibility

    /**
     * \}
     */
} VmbFeatureEnumEntry_t;

/**
 * \}
 */

/**
 * \name Frame
 * \{
 */

/**
 * \brief Status of a frame transfer.
 */
typedef enum VmbFrameStatusType
{
    VmbFrameStatusComplete          =  0,       //!< Frame has been completed without errors
    VmbFrameStatusIncomplete        = -1,       //!< Frame could not be filled to the end
    VmbFrameStatusTooSmall          = -2,       //!< Frame buffer was too small
    VmbFrameStatusInvalid           = -3,       //!< Frame buffer was invalid
} VmbFrameStatusType;

/**
 * \brief Type for the frame status; for values see ::VmbFrameStatusType.
 */
typedef VmbInt32_t VmbFrameStatus_t;

/**
 * \brief Frame flags.
 */
typedef enum VmbFrameFlagsType
{
    VmbFrameFlagsNone                   = 0,        //!< No additional information is provided
    VmbFrameFlagsDimension              = 1,        //!< VmbFrame_t::width and VmbFrame_t::height are provided
    VmbFrameFlagsOffset                 = 2,        //!< VmbFrame_t::offsetX and VmbFrame_t::offsetY are provided (ROI)
    VmbFrameFlagsFrameID                = 4,        //!< VmbFrame_t::frameID is provided
    VmbFrameFlagsTimestamp              = 8,        //!< VmbFrame_t::timestamp is provided
    VmbFrameFlagsImageData              = 16,       //!< VmbFrame_t::imageData is provided
    VmbFrameFlagsPayloadType            = 32,       //!< VmbFrame_t::payloadType is provided
    VmbFrameFlagsChunkDataPresent       = 64,       //!< VmbFrame_t::chunkDataPresent is set based on info provided by the transport layer
} VmbFrameFlagsType;

/**
 * \brief Type for Frame flags; for values see ::VmbFrameFlagsType.
 */
typedef VmbUint32_t VmbFrameFlags_t;

/**
 * \brief Frame payload type.
 */
typedef enum VmbPayloadType
{
    VmbPayloadTypeUnknown               = 0,        //!< Unknown payload type
    VmbPayloadTypeImage                 = 1,        //!< image data
    VmbPayloadTypeRaw                   = 2,        //!< raw data
    VmbPayloadTypeFile                  = 3,        //!< file data
    VmbPayloadTypeJPEG                  = 5,        //!< JPEG data as described in the GigEVision 2.0 specification
    VmbPayloadTypJPEG2000               = 6,        //!< JPEG 2000 data as described in the GigEVision 2.0 specification
    VmbPayloadTypeH264                  = 7,        //!< H.264 data as described in the GigEVision 2.0 specification
    VmbPayloadTypeChunkOnly             = 8,        //!< Chunk data exclusively
    VmbPayloadTypeDeviceSpecific        = 9,        //!< Device specific data format
    VmbPayloadTypeGenDC                 = 11,       //!< GenDC data
} VmbPayloadType;

/**
 * \brief Type representing the payload type of a frame. For values see ::VmbPayloadType.
 */
typedef VmbUint32_t VmbPayloadType_t;

/**
 * \brief Type used to represent a dimension value, e.g. the image height.
 */
typedef VmbUint32_t VmbImageDimension_t;

/**
 * \brief Frame delivered by the camera.
 */
typedef struct VmbFrame
{
    /** 
     * \name In
     * \{
    */

    void*                   buffer;                 //!< Comprises image and potentially chunk data
    VmbUint32_t             bufferSize;             //!< The size of the data buffer
    void*                   context[4];             //!< 4 void pointers that can be employed by the user (e.g. for storing handles)

    /**
     * \}
     */

    /**
     * \name Out
     * \{
     */

    VmbFrameStatus_t        receiveStatus;          //!< The resulting status of the receive operation
    VmbUint64_t             frameID;                //!< Unique ID of this frame in this stream
    VmbUint64_t             timestamp;              //!< The timestamp set by the camera
    VmbUint8_t*             imageData;              //!< The start of the image data, if present, or null
    VmbFrameFlags_t         receiveFlags;           //!< Flags indicating which additional frame information is available
    VmbPixelFormat_t        pixelFormat;            //!< Pixel format of the image
    VmbImageDimension_t     width;                  //!< Width of an image
    VmbImageDimension_t     height;                 //!< Height of an image
    VmbImageDimension_t     offsetX;                //!< Horizontal offset of an image
    VmbImageDimension_t     offsetY;                //!< Vertical offset of an image
    VmbPayloadType_t        payloadType;            //!< The type of payload
    VmbBool_t               chunkDataPresent;       //!< True if the transport layer reported chunk data to be present in the buffer

    /** 
     * \}
     */
} VmbFrame_t;

/**
 * \}
 */

/**
 * \name Save/LoadSettings
 * \{
 */

/**
 * \brief Type of features that are to be saved (persisted) to the XML file when using ::VmbSettingsSave
 */
typedef enum VmbFeaturePersistType
{
    VmbFeaturePersistAll            = 0,        //!< Save all features to XML, including look-up tables (if possible)
    VmbFeaturePersistStreamable     = 1,        //!< Save only features marked as streamable, excluding look-up tables
    VmbFeaturePersistNoLUT          = 2         //!< Save all features except look-up tables (default)
} VmbFeaturePersistType;

/**
 * \brief Type for feature persistence; for values see ::VmbFeaturePersistType.
 */
typedef VmbUint32_t VmbFeaturePersist_t;

/**
 * \brief Parameters determining the operation mode of ::VmbSettingsSave and ::VmbSettingsLoad.
 */
typedef enum VmbModulePersistFlagsType
{
    VmbModulePersistFlagsNone           = 0x00, //!< Persist/Load features for no module. 
    VmbModulePersistFlagsTransportLayer = 0x01, //!< Persist/Load the transport layer features.
    VmbModulePersistFlagsInterface      = 0x02, //!< Persist/Load the interface features.
    VmbModulePersistFlagsRemoteDevice   = 0x04, //!< Persist/Load the remote device features.
    VmbModulePersistFlagsLocalDevice    = 0x08, //!< Persist/Load the local device features.
    VmbModulePersistFlagsStreams        = 0x10, //!< Persist/Load the features of stream modules.
    VmbModulePersistFlagsAll            = 0xff  //!< Persist/Load features for all modules.
} VmbModulePersistFlagsType;

/**
 * \brief Type for module persist flags; for values see VmbModulePersistFlagsType
 * 
 * Use a combination of ::VmbModulePersistFlagsType constants
 */
typedef VmbUint32_t VmbModulePersistFlags_t;

/**
 * \brief A level to use for logging 
 */
typedef enum VmbLogLevel
{
    VmbLogLevelNone = 0,                //!< Nothing is logged regardless of the severity of the issue
    VmbLogLevelError,                   //!< Only errors are logged
    VmbLogLevelDebug,                   //!< Only error and debug messages are logged 
    VmbLogLevelWarn,                    //!< Only error, debug and warn messages are logged 
    VmbLogLevelTrace,                   //!< all messages are logged 
    VmbLogLevelAll = VmbLogLevelTrace,  //!< all messages are logged 
} VmbLogLevel;

/**
 * \brief The type used for storing the log level
 * 
 * Use a constant from ::VmbLogLevel
 */
typedef VmbUint32_t VmbLogLevel_t;

/**
 * \brief Parameters determining the operation mode of ::VmbSettingsSave and ::VmbSettingsLoad
 */
typedef struct VmbFeaturePersistSettings
{
    /**
     * \name In
     * \{
     */

    VmbFeaturePersist_t     persistType;        //!< Type of features that are to be saved
    VmbModulePersistFlags_t modulePersistFlags; //!< Flags specifying the modules to persist/load
    VmbUint32_t             maxIterations;      //!< Number of iterations when loading settings
    VmbLogLevel_t           loggingLevel;       //!< Determines level of detail for load/save settings logging

    /**
     * \}
     */
} VmbFeaturePersistSettings_t;

/**
 * \}
 */

/**
 * \name Callbacks
 * \{
 */

/**
 * \brief Invalidation callback type for a function that gets called in a separate thread
 *        and has been registered with ::VmbFeatureInvalidationRegister().
 *
 * While the callback is run, all feature data is atomic. After the callback finishes,
 * the feature data may be updated with new values.
 *
 * Do not spend too much time in this thread; it prevents the feature values
 * from being updated from any other thread or the lower-level drivers.
 *
 * \param[in]   handle              Handle for an entity that exposes features
 * \param[in]   name                Name of the feature
 * \param[in]   userContext         Pointer to the user context, see ::VmbFeatureInvalidationRegister
 */
typedef void (VMB_CALL* VmbInvalidationCallback)(const VmbHandle_t handle, const char* name, void* userContext);

/**
 * \brief Frame Callback type for a function that gets called in a separate thread
 *        if a frame has been queued with ::VmbCaptureFrameQueue.
 * 
 * \warning Any operations closing the stream including ::VmbShutdown and ::VmbCameraClose in addition to
 *          ::VmbCaptureEnd block until any currently active callbacks return. If the callback does not
 *          return in finite time, the program may not return.
 *
 * \param[in]   cameraHandle      Handle of the camera the frame belongs to
 * \param[in]   streamHandle      Handle of the stream the frame belongs to
 * \param[in]   frame             The received frame
 */
typedef void (VMB_CALL* VmbFrameCallback)(const VmbHandle_t cameraHandle, const VmbHandle_t streamHandle, VmbFrame_t* frame);

/**
 * \brief Function pointer type to access chunk data
 *
 * This function should complete as quickly as possible, since it blocks other updates on the
 * remote device.
 *
 * This function should not throw exceptions, even if VmbC is used from C++. Any exception
 * thrown will only result in an error code indicating that an exception was thrown.
 *
 * \param[in] featureAccessHandle A special handle that can be used for accessing features;
 *                                the handle is only valid during the call of the function.
 * \param[in] userContext         The value the user passed to ::VmbChunkDataAccess.
 *
 * \return An error to be returned from ::VmbChunkDataAccess in the absence of other errors;
 *         A custom exit code >= ::VmbErrorCustom can be returned to indicate a failure via
 *         ::VmbChunkDataAccess return code
 */
typedef VmbError_t(VMB_CALL* VmbChunkAccessCallback)(VmbHandle_t featureAccessHandle, void* userContext);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif // VMBC_TYPE_DEFINITIONS_H_INCLUDE_
