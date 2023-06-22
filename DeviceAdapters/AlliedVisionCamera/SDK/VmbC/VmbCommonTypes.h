/*=============================================================================
  Copyright (C) 2012 - 2021 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------
 
  File:        VmbCommonTypes.h

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
 * \brief Main header file for the common types of the APIs.
 * 
 * This file describes all necessary definitions for types used within
 * the Vmb APIs. These type definitions are designed to be
 * portable from other languages and other operating systems.
 */

#ifndef VMBCOMMONTYPES_H_INCLUDE_
#define VMBCOMMONTYPES_H_INCLUDE_

#ifdef _WIN32
#   include <wchar.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \name Basic Types
 * \{
 */

#if defined (_MSC_VER)

    /**
     * \brief 8-bit signed integer.
     */
    typedef __int8              VmbInt8_t;

    /**
     * \brief 8-bit unsigned integer.
     */
    typedef unsigned __int8     VmbUint8_t;

    /**
     * \brief 16-bit signed integer.
     */
    typedef __int16             VmbInt16_t;

    /**
     * \brief 16-bit unsigned integer.
     */
    typedef unsigned __int16    VmbUint16_t;

    /**
     * \brief 32-bit signed integer.
     */
    typedef __int32             VmbInt32_t;

    /**
     * \brief 32-bit unsigned integer.
     */
    typedef unsigned __int32    VmbUint32_t;

    /**
     * \brief 64-bit signed integer.
     */
    typedef __int64             VmbInt64_t;

    /**
     * \brief 64-bit unsigned integer.
     */
    typedef unsigned __int64    VmbUint64_t;

#else

    /**
     * \brief 8-bit signed integer.
     */
    typedef signed char         VmbInt8_t;

    /**
     * \brief 8-bit unsigned integer.
     */
    typedef unsigned char       VmbUint8_t;

    /**
     * \brief 16-bit signed integer.
     */
    typedef short               VmbInt16_t;

    /**
     * \brief 16-bit unsigned integer.
     */
    typedef unsigned short      VmbUint16_t;

    /**
     * \brief 32-bit signed integer.
     */
    typedef int                 VmbInt32_t;

    /**
     * \brief 32-bit unsigned integer.
     */
    typedef unsigned int        VmbUint32_t;

    /**
     * \brief 64-bit signed integer.
     */
    typedef long long           VmbInt64_t;

    /**
     * \brief 64-bit unsigned integer.
     */
    typedef unsigned long long  VmbUint64_t;

#endif

    /**
     * \brief Handle, e.g. for a camera.
     */
    typedef void*               VmbHandle_t;

#if defined(__cplusplus) || defined(__bool_true_false_are_defined)

    /**
     * \brief Standard type for boolean values.
     */
    typedef bool                VmbBool_t;

#else

    /**
     * \brief Boolean type (equivalent to char).
     *
     * For values see ::VmbBoolVal
     */
    typedef char                VmbBool_t;

#endif

    /**
     * \brief enum for bool values.
     */
    typedef enum VmbBoolVal
    {
        VmbBoolTrue     = 1,
        VmbBoolFalse    = 0,
    } VmbBoolVal;

    /**
     * \brief char type.
     */
    typedef unsigned char       VmbUchar_t;

#ifdef _WIN32

    /**
     * \brief Character type used for file paths (Windows uses wchar_t not char).
     */
    typedef wchar_t VmbFilePathChar_t;

     /**
      * \brief macro for converting a c string literal into a system dependent string literal
      * 
      * Adds L as prefix on Windows and is replaced by the unmodified value on other operating systems.
      * 
      * \code{.c}
      * const VmbFilePathChar_t* path = VMB_FILE_PATH_LITERAL("./some/path/tl.cti");
      * \endcode
      */
#    define VMB_FILE_PATH_LITERAL(value) L##value

#else

    /**
     * Character type used for file paths
     */
    typedef char VmbFilePathChar_t;

    /**
     * \brief macro for converting a c string literal into a system dependent string literal
     *
     * Adds L as prefix on Windows and is replaced by the unmodified value on other operating systems.
     * 
     * \code{.c}
     * const VmbFilePathChar_t* path = VMB_FILE_PATH_LITERAL("./some/path/tl.cti");
     * \endcode
     */
#    define VMB_FILE_PATH_LITERAL(value) value
#endif

/**
 * \}
 */

/**
 * \name Error Codes
 * \{
 */

    /**
     * \brief Error codes, returned by most functions.
     */
    typedef enum VmbErrorType
    {
        VmbErrorSuccess                 =  0,           //!< No error
        VmbErrorInternalFault           = -1,           //!< Unexpected fault in VmbC or driver
        VmbErrorApiNotStarted           = -2,           //!< ::VmbStartup() was not called before the current command
        VmbErrorNotFound                = -3,           //!< The designated instance (camera, feature etc.) cannot be found
        VmbErrorBadHandle               = -4,           //!< The given handle is not valid
        VmbErrorDeviceNotOpen           = -5,           //!< Device was not opened for usage
        VmbErrorInvalidAccess           = -6,           //!< Operation is invalid with the current access mode
        VmbErrorBadParameter            = -7,           //!< One of the parameters is invalid (usually an illegal pointer)
        VmbErrorStructSize              = -8,           //!< The given struct size is not valid for this version of the API
        VmbErrorMoreData                = -9,           //!< More data available in a string/list than space is provided
        VmbErrorWrongType               = -10,          //!< Wrong feature type for this access function
        VmbErrorInvalidValue            = -11,          //!< The value is not valid; either out of bounds or not an increment of the minimum
        VmbErrorTimeout                 = -12,          //!< Timeout during wait
        VmbErrorOther                   = -13,          //!< Other error
        VmbErrorResources               = -14,          //!< Resources not available (e.g. memory)
        VmbErrorInvalidCall             = -15,          //!< Call is invalid in the current context (e.g. callback)
        VmbErrorNoTL                    = -16,          //!< No transport layers are found
        VmbErrorNotImplemented          = -17,          //!< API feature is not implemented
        VmbErrorNotSupported            = -18,          //!< API feature is not supported
        VmbErrorIncomplete              = -19,          //!< The current operation was not completed (e.g. a multiple registers read or write)
        VmbErrorIO                      = -20,          //!< Low level IO error in transport layer
        VmbErrorValidValueSetNotPresent = -21,          //!< The valid value set could not be retrieved, since the feature does not provide this property
        VmbErrorGenTLUnspecified        = -22,          //!< Unspecified GenTL runtime error
        VmbErrorUnspecified             = -23,          //!< Unspecified runtime error
        VmbErrorBusy                    = -24,          //!< The responsible module/entity is busy executing actions
        VmbErrorNoData                  = -25,          //!< The function has no data to work on
        VmbErrorParsingChunkData        = -26,          //!< An error occurred parsing a buffer containing chunk data
        VmbErrorInUse                   = -27,          //!< Something is already in use
        VmbErrorUnknown                 = -28,          //!< Error condition unknown
        VmbErrorXml                     = -29,          //!< Error parsing XML
        VmbErrorNotAvailable            = -30,          //!< Something is not available
        VmbErrorNotInitialized          = -31,          //!< Something is not initialized
        VmbErrorInvalidAddress          = -32,          //!< The given address is out of range or invalid for internal reasons
        VmbErrorAlready                 = -33,          //!< Something has already been done
        VmbErrorNoChunkData             = -34,          //!< A frame expected to contain chunk data does not contain chunk data
        VmbErrorUserCallbackException   = -35,          //!< A callback provided by the user threw an exception
        VmbErrorFeaturesUnavailable     = -36,          //!< The XML for the module is currently not loaded; the module could be in the wrong state or the XML could not be retrieved or could not be parsed properly
        VmbErrorTLNotFound              = -37,          //!< A required transport layer could not be found or loaded
        VmbErrorAmbiguous               = -39,          //!< An entity cannot be uniquely identified based on the information provided
        VmbErrorRetriesExceeded         = -40,          //!< Something could not be accomplished with a given number of retries
        VmbErrorInsufficientBufferCount = -41,          //!< The operation requires more buffers
        VmbErrorCustom                  = 1,            //!< The minimum error code to use for user defined error codes to avoid conflict with existing error codes
    } VmbErrorType;

    /**
     * \brief Type for an error returned by API methods; for values see ::VmbErrorType.
     */
    typedef VmbInt32_t VmbError_t;

/**
 * \}
 */

/**
 * \name Version
 * \{
 */

    /**
     * \brief Version information.
     */
    typedef struct VmbVersionInfo
    {
        /**
         * \name Out
         * \{
         */

        VmbUint32_t             major;          //!< Major version number
        VmbUint32_t             minor;          //!< Minor version number
        VmbUint32_t             patch;          //!< Patch version number

        /**
        * \}
        */
    } VmbVersionInfo_t;

/**
 * \}
 */

 /**
 * \name Pixel information
 * \{
 */

    /**
     * \brief Indicates if pixel is monochrome or RGB.
     */
    typedef enum VmbPixelType
    {
        VmbPixelMono  =         0x01000000,     //!< Monochrome pixel
        VmbPixelColor =         0x02000000      //!< Pixel bearing color information
    } VmbPixelType;

    /**
     * \brief Indicates number of bits for a pixel. Needed for building values of ::VmbPixelFormatType.
     */
    typedef enum VmbPixelOccupyType
    {
        VmbPixelOccupy8Bit  =   0x00080000,     //!< Pixel effectively occupies 8 bits
        VmbPixelOccupy10Bit =   0x000A0000,     //!< Pixel effectively occupies 10 bits
        VmbPixelOccupy12Bit =   0x000C0000,     //!< Pixel effectively occupies 12 bits
        VmbPixelOccupy14Bit =   0x000E0000,     //!< Pixel effectively occupies 14 bits
        VmbPixelOccupy16Bit =   0x00100000,     //!< Pixel effectively occupies 16 bits
        VmbPixelOccupy24Bit =   0x00180000,     //!< Pixel effectively occupies 24 bits
        VmbPixelOccupy32Bit =   0x00200000,     //!< Pixel effectively occupies 32 bits
        VmbPixelOccupy48Bit =   0x00300000,     //!< Pixel effectively occupies 48 bits
        VmbPixelOccupy64Bit =   0x00400000,     //!< Pixel effectively occupies 64 bits
    } VmbPixelOccupyType;

    /**
     * \brief Pixel format types.
     * As far as possible, the Pixel Format Naming Convention (PFNC) has been followed, allowing a few deviations.
     * If data spans more than one byte, it is always LSB aligned, except if stated differently.
     */
    typedef enum VmbPixelFormatType
    {
         // mono formats
        VmbPixelFormatMono8                   = VmbPixelMono  | VmbPixelOccupy8Bit  | 0x0001,  //!< Monochrome, 8 bits (PFNC:  Mono8)
        VmbPixelFormatMono10                  = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0003,  //!< Monochrome, 10 bits in 16 bits (PFNC:  Mono10)
        VmbPixelFormatMono10p                 = VmbPixelMono  | VmbPixelOccupy10Bit | 0x0046,  //!< Monochrome, 10 bits in 16 bits (PFNC:  Mono10p)
        VmbPixelFormatMono12                  = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0005,  //!< Monochrome, 12 bits in 16 bits (PFNC:  Mono12)
        VmbPixelFormatMono12Packed            = VmbPixelMono  | VmbPixelOccupy12Bit | 0x0006,  //!< Monochrome, 2x12 bits in 24 bits (GEV:Mono12Packed)
        VmbPixelFormatMono12p                 = VmbPixelMono  | VmbPixelOccupy12Bit | 0x0047,  //!< Monochrome, 2x12 bits in 24 bits (PFNC:  MonoPacked)
        VmbPixelFormatMono14                  = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0025,  //!< Monochrome, 14 bits in 16 bits (PFNC:  Mono14)
        VmbPixelFormatMono16                  = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0007,  //!< Monochrome, 16 bits (PFNC:  Mono16)

        // bayer formats
        VmbPixelFormatBayerGR8                = VmbPixelMono  | VmbPixelOccupy8Bit  | 0x0008,  //!< Bayer-color, 8 bits, starting with GR line (PFNC:  BayerGR8)
        VmbPixelFormatBayerRG8                = VmbPixelMono  | VmbPixelOccupy8Bit  | 0x0009,  //!< Bayer-color, 8 bits, starting with RG line (PFNC:  BayerRG8)
        VmbPixelFormatBayerGB8                = VmbPixelMono  | VmbPixelOccupy8Bit  | 0x000A,  //!< Bayer-color, 8 bits, starting with GB line (PFNC:  BayerGB8)
        VmbPixelFormatBayerBG8                = VmbPixelMono  | VmbPixelOccupy8Bit  | 0x000B,  //!< Bayer-color, 8 bits, starting with BG line (PFNC:  BayerBG8)
        VmbPixelFormatBayerGR10               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x000C,  //!< Bayer-color, 10 bits in 16 bits, starting with GR line (PFNC:  BayerGR10)
        VmbPixelFormatBayerRG10               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x000D,  //!< Bayer-color, 10 bits in 16 bits, starting with RG line (PFNC:  BayerRG10)
        VmbPixelFormatBayerGB10               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x000E,  //!< Bayer-color, 10 bits in 16 bits, starting with GB line (PFNC:  BayerGB10)
        VmbPixelFormatBayerBG10               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x000F,  //!< Bayer-color, 10 bits in 16 bits, starting with BG line (PFNC:  BayerBG10)
        VmbPixelFormatBayerGR12               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0010,  //!< Bayer-color, 12 bits in 16 bits, starting with GR line (PFNC:  BayerGR12)
        VmbPixelFormatBayerRG12               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0011,  //!< Bayer-color, 12 bits in 16 bits, starting with RG line (PFNC:  BayerRG12)
        VmbPixelFormatBayerGB12               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0012,  //!< Bayer-color, 12 bits in 16 bits, starting with GB line (PFNC:  BayerGB12)
        VmbPixelFormatBayerBG12               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0013,  //!< Bayer-color, 12 bits in 16 bits, starting with BG line (PFNC:  BayerBG12)
        VmbPixelFormatBayerGR12Packed         = VmbPixelMono  | VmbPixelOccupy12Bit | 0x002A,  //!< Bayer-color, 2x12 bits in 24 bits, starting with GR line (GEV:BayerGR12Packed)
        VmbPixelFormatBayerRG12Packed         = VmbPixelMono  | VmbPixelOccupy12Bit | 0x002B,  //!< Bayer-color, 2x12 bits in 24 bits, starting with RG line (GEV:BayerRG12Packed)
        VmbPixelFormatBayerGB12Packed         = VmbPixelMono  | VmbPixelOccupy12Bit | 0x002C,  //!< Bayer-color, 2x12 bits in 24 bits, starting with GB line (GEV:BayerGB12Packed)
        VmbPixelFormatBayerBG12Packed         = VmbPixelMono  | VmbPixelOccupy12Bit | 0x002D,  //!< Bayer-color, 2x12 bits in 24 bits, starting with BG line (GEV:BayerBG12Packed)
        VmbPixelFormatBayerGR10p              = VmbPixelMono  | VmbPixelOccupy10Bit | 0x0056,  //!< Bayer-color, 10 bits continuous packed, starting with GR line (PFNC:  BayerGR10p)
        VmbPixelFormatBayerRG10p              = VmbPixelMono  | VmbPixelOccupy10Bit | 0x0058,  //!< Bayer-color, 10 bits continuous packed, starting with RG line (PFNC:  BayerRG10p)
        VmbPixelFormatBayerGB10p              = VmbPixelMono  | VmbPixelOccupy10Bit | 0x0054,  //!< Bayer-color, 10 bits continuous packed, starting with GB line (PFNC:  BayerGB10p)
        VmbPixelFormatBayerBG10p              = VmbPixelMono  | VmbPixelOccupy10Bit | 0x0052,  //!< Bayer-color, 10 bits continuous packed, starting with BG line (PFNC:  BayerBG10p)
        VmbPixelFormatBayerGR12p              = VmbPixelMono  | VmbPixelOccupy12Bit | 0x0057,  //!< Bayer-color, 12 bits continuous packed, starting with GR line (PFNC:  BayerGR12p)
        VmbPixelFormatBayerRG12p              = VmbPixelMono  | VmbPixelOccupy12Bit | 0x0059,  //!< Bayer-color, 12 bits continuous packed, starting with RG line (PFNC:  BayerRG12p)
        VmbPixelFormatBayerGB12p              = VmbPixelMono  | VmbPixelOccupy12Bit | 0x0055,  //!< Bayer-color, 12 bits continuous packed, starting with GB line (PFNC:  BayerGB12p)
        VmbPixelFormatBayerBG12p              = VmbPixelMono  | VmbPixelOccupy12Bit | 0x0053,  //!< Bayer-color, 12 bits continuous packed, starting with BG line (PFNC: BayerBG12p)
        VmbPixelFormatBayerGR16               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x002E,  //!< Bayer-color, 16 bits, starting with GR line (PFNC: BayerGR16)
        VmbPixelFormatBayerRG16               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x002F,  //!< Bayer-color, 16 bits, starting with RG line (PFNC: BayerRG16)
        VmbPixelFormatBayerGB16               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0030,  //!< Bayer-color, 16 bits, starting with GB line (PFNC: BayerGB16)
        VmbPixelFormatBayerBG16               = VmbPixelMono  | VmbPixelOccupy16Bit | 0x0031,  //!< Bayer-color, 16 bits, starting with BG line (PFNC: BayerBG16)

         // rgb formats
        VmbPixelFormatRgb8                    = VmbPixelColor | VmbPixelOccupy24Bit | 0x0014,  //!< RGB, 8 bits x 3 (PFNC: RGB8)
        VmbPixelFormatBgr8                    = VmbPixelColor | VmbPixelOccupy24Bit | 0x0015,  //!< BGR, 8 bits x 3 (PFNC: BGR8)
        VmbPixelFormatRgb10                   = VmbPixelColor | VmbPixelOccupy48Bit | 0x0018,  //!< RGB, 12 bits in 16 bits x 3 (PFNC: RGB12)
        VmbPixelFormatBgr10                   = VmbPixelColor | VmbPixelOccupy48Bit | 0x0019,  //!< RGB, 12 bits in 16 bits x 3 (PFNC: RGB12)
        VmbPixelFormatRgb12                   = VmbPixelColor | VmbPixelOccupy48Bit | 0x001A,  //!< RGB, 12 bits in 16 bits x 3 (PFNC: RGB12)
        VmbPixelFormatBgr12                   = VmbPixelColor | VmbPixelOccupy48Bit | 0x001B,  //!< RGB, 12 bits in 16 bits x 3 (PFNC: RGB12)
        VmbPixelFormatRgb14                   = VmbPixelColor | VmbPixelOccupy48Bit | 0x005E,  //!< RGB, 14 bits in 16 bits x 3 (PFNC: RGB12)
        VmbPixelFormatBgr14                   = VmbPixelColor | VmbPixelOccupy48Bit | 0x004A,  //!< RGB, 14 bits in 16 bits x 3 (PFNC: RGB12)
        VmbPixelFormatRgb16                   = VmbPixelColor | VmbPixelOccupy48Bit | 0x0033,  //!< RGB, 16 bits x 3 (PFNC: RGB16)
        VmbPixelFormatBgr16                   = VmbPixelColor | VmbPixelOccupy48Bit | 0x004B,  //!< RGB, 16 bits x 3 (PFNC: RGB16)

         // rgba formats
        VmbPixelFormatArgb8                   = VmbPixelColor | VmbPixelOccupy32Bit | 0x0016,  //!< ARGB, 8 bits x 4 (PFNC: RGBa8)
        VmbPixelFormatRgba8                   = VmbPixelFormatArgb8,                           //!< RGBA, 8 bits x 4, legacy name
        VmbPixelFormatBgra8                   = VmbPixelColor | VmbPixelOccupy32Bit | 0x0017,  //!< BGRA, 8 bits x 4 (PFNC: BGRa8)
        VmbPixelFormatRgba10                  = VmbPixelColor | VmbPixelOccupy64Bit | 0x005F,  //!< RGBA, 8 bits x 4, legacy name
        VmbPixelFormatBgra10                  = VmbPixelColor | VmbPixelOccupy64Bit | 0x004C,  //!< RGBA, 8 bits x 4, legacy name
        VmbPixelFormatRgba12                  = VmbPixelColor | VmbPixelOccupy64Bit | 0x0061,  //!< RGBA, 8 bits x 4, legacy name
        VmbPixelFormatBgra12                  = VmbPixelColor | VmbPixelOccupy64Bit | 0x004E,  //!< RGBA, 8 bits x 4, legacy name
        VmbPixelFormatRgba14                  = VmbPixelColor | VmbPixelOccupy64Bit | 0x0063,  //!< RGBA, 8 bits x 4, legacy name
        VmbPixelFormatBgra14                  = VmbPixelColor | VmbPixelOccupy64Bit | 0x0050,  //!< RGBA, 8 bits x 4, legacy name
        VmbPixelFormatRgba16                  = VmbPixelColor | VmbPixelOccupy64Bit | 0x0064,  //!< RGBA, 8 bits x 4, legacy name
        VmbPixelFormatBgra16                  = VmbPixelColor | VmbPixelOccupy64Bit | 0x0051,  //!< RGBA, 8 bits x 4, legacy name

         // yuv/ycbcr formats
        VmbPixelFormatYuv411                  = VmbPixelColor | VmbPixelOccupy12Bit | 0x001E,  //!< YUV 4:1:1 with 8 bits (PFNC: YUV411_8_UYYVYY, GEV:YUV411Packed)
        VmbPixelFormatYuv422                  = VmbPixelColor | VmbPixelOccupy16Bit | 0x001F,  //!< YUV 4:2:2 with 8 bits (PFNC: YUV422_8_UYVY, GEV:YUV422Packed)
        VmbPixelFormatYuv444                  = VmbPixelColor | VmbPixelOccupy24Bit | 0x0020,  //!< YUV 4:4:4 with 8 bits (PFNC: YUV8_UYV, GEV:YUV444Packed)
        VmbPixelFormatYuv422_8                = VmbPixelColor | VmbPixelOccupy16Bit | 0x0032,  //!< YUV 4:2:2 with 8 bits Channel order YUYV (PFNC: YUV422_8)
        VmbPixelFormatYCbCr8_CbYCr            = VmbPixelColor | VmbPixelOccupy24Bit | 0x003A,  //!< YCbCr 4:4:4 with 8 bits (PFNC: YCbCr8_CbYCr) - identical to VmbPixelFormatYuv444
        VmbPixelFormatYCbCr422_8              = VmbPixelColor | VmbPixelOccupy16Bit | 0x003B,  //!< YCbCr 4:2:2 8-bit YCbYCr (PFNC: YCbCr422_8)
        VmbPixelFormatYCbCr411_8_CbYYCrYY     = VmbPixelColor | VmbPixelOccupy12Bit | 0x003C,  //!< YCbCr 4:1:1 with 8 bits (PFNC: YCbCr411_8_CbYYCrYY) - identical to VmbPixelFormatYuv411
        VmbPixelFormatYCbCr601_8_CbYCr        = VmbPixelColor | VmbPixelOccupy24Bit | 0x003D,  //!< YCbCr601 4:4:4 8-bit CbYCrt (PFNC: YCbCr601_8_CbYCr)
        VmbPixelFormatYCbCr601_422_8          = VmbPixelColor | VmbPixelOccupy16Bit | 0x003E,  //!< YCbCr601 4:2:2 8-bit YCbYCr (PFNC: YCbCr601_422_8)
        VmbPixelFormatYCbCr601_411_8_CbYYCrYY = VmbPixelColor | VmbPixelOccupy12Bit | 0x003F,  //!< YCbCr601 4:1:1 8-bit CbYYCrYY (PFNC: YCbCr601_411_8_CbYYCrYY)
        VmbPixelFormatYCbCr709_8_CbYCr        = VmbPixelColor | VmbPixelOccupy24Bit | 0x0040,  //!< YCbCr709 4:4:4 8-bit CbYCr (PFNC: YCbCr709_8_CbYCr)
        VmbPixelFormatYCbCr709_422_8          = VmbPixelColor | VmbPixelOccupy16Bit | 0x0041,  //!< YCbCr709 4:2:2 8-bit YCbYCr (PFNC: YCbCr709_422_8)
        VmbPixelFormatYCbCr709_411_8_CbYYCrYY = VmbPixelColor | VmbPixelOccupy12Bit | 0x0042,  //!< YCbCr709 4:1:1 8-bit CbYYCrYY (PFNC: YCbCr709_411_8_CbYYCrYY)
        VmbPixelFormatYCbCr422_8_CbYCrY       = VmbPixelColor | VmbPixelOccupy16Bit | 0x0043,  //!< YCbCr 4:2:2 with 8 bits (PFNC: YCbCr422_8_CbYCrY) - identical to VmbPixelFormatYuv422
        VmbPixelFormatYCbCr601_422_8_CbYCrY   = VmbPixelColor | VmbPixelOccupy16Bit | 0x0044,  //!< YCbCr601 4:2:2 8-bit CbYCrY (PFNC: YCbCr601_422_8_CbYCrY)
        VmbPixelFormatYCbCr709_422_8_CbYCrY   = VmbPixelColor | VmbPixelOccupy16Bit | 0x0045,  //!< YCbCr709 4:2:2 8-bit CbYCrY (PFNC: YCbCr709_422_8_CbYCrY)
        VmbPixelFormatYCbCr411_8              = VmbPixelColor | VmbPixelOccupy12Bit | 0x005A,  //!< YCbCr 4:1:1 8-bit YYCbYYCr (PFNC: YCbCr411_8)
        VmbPixelFormatYCbCr8                  = VmbPixelColor | VmbPixelOccupy24Bit | 0x005B,  //!< YCbCr 4:4:4 8-bit YCbCr (PFNC: YCbCr8)

        VmbPixelFormatLast,
    } VmbPixelFormatType;

    /**
     * \brief Type for the pixel format; for values see ::VmbPixelFormatType.
     */
    typedef VmbUint32_t VmbPixelFormat_t;

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif // VMBCOMMONTYPES_H_INCLUDE_
