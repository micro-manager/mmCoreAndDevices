/*=============================================================================
  Copyright (C) 2012 - 2021 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        VmbTransformTypes.h

  Description: Definition of types used in the Vmb Image Transform library.

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
 */
#ifndef VMB_TRANSFORM_TYPES_H_
#define VMB_TRANSFORM_TYPES_H_

#include <VmbC/VmbCommonTypes.h>

/**
 * \brief the type of character to use for strings. 
 */
typedef char                    VmbANSIChar_t;

/**
 * \brief The floating point type to use for matrices.
 */
typedef float                   VmbFloat_t;

/**
 * \brief Enumeration for the Bayer pattern.
 */
typedef enum VmbBayerPattern
{
    VmbBayerPatternRGGB=0,     //!< RGGB pattern, red pixel comes first
    VmbBayerPatternGBRG,       //!< RGGB pattern, green pixel of blue row comes first
    VmbBayerPatternGRBG,       //!< RGGB pattern, green pixel of red row comes first
    VmbBayerPatternBGGR,       //!< RGGB pattern, blue pixel comes first
    VmbBayerPatternCYGM=128,   //!< CYGM pattern, cyan pixel comes first in the first row, green in the second row (of the sensor)
    VmbBayerPatternGMCY,       //!< CYGM pattern, green pixel comes first in the first row, cyan in the second row (of the sensor)
    VmbBayerPatternCYMG,       //!< CYGM pattern, cyan pixel comes first in the first row, magenta in the second row (of the sensor)
    VmbBayerPatternMGCY,       //!< CYGM pattern, magenta pixel comes first in the first row, cyan in the second row (of the sensor)
    VmbBayerPatternLAST=255
} VmbBayerPattern;

/**
 * \brief Type for an error returned by API methods; for values see ::VmbBayerPattern.
 */
typedef VmbUint32_t  VmbBayerPattern_t;

/**
 * \brief Enumeration for the endianness.
 */
typedef enum VmbEndianness
{
    VmbEndiannessLittle=0, //!< Little endian data format
    VmbEndiannessBig,      //!< Big endian data format
    VmbEndiannessLast=255
} VmbEndianness;

/**
 * \brief Type for the endianness; for values see ::VmbEndianness.
 */
typedef VmbUint32_t VmbEndianness_t;

/**
 * \brief Enumeration for the image alignment.
 */
typedef enum VmbAlignment
{
    VmbAlignmentMSB=0,    //!< Data is MSB aligned (pppp pppp pppp ....)
    VmbAlignmentLSB,      //!< Data is LSB aligned (.... pppp pppp pppp)
    VmbAlignmentLAST=255
} VmbAlignment;

/**
 * \brief Enumeration for the image alignment; for values see ::VmbAlignment
 */
typedef VmbUint32_t VmbAlignment_t;

/**
 * \name Library Info
 * \defgroup Library Info
 * \{
 */

/**
 * \brief States of the multi media technology support for operating system and processor.
 */
typedef struct VmbSupportState_t
{
    VmbBool_t Processor;         //!< technology supported by the processor
    VmbBool_t OperatingSystem;   //!< technology supported by the OS
} VmbSupportState_t;

/**
 * \brief States of the support for different multimedia technologies
 */
typedef struct VmbTechInfo_t
{
    VmbSupportState_t IntelMMX;       //!< INTEL first gen MultiMedia eXtension
    VmbSupportState_t IntelSSE;       //!< INTEL Streaming SIMD Extension
    VmbSupportState_t IntelSSE2;      //!< INTEL Streaming SIMD Extension 2
    VmbSupportState_t IntelSSE3;      //!< INTEL Streaming SIMD Extension 3
    VmbSupportState_t IntelSSSE3;     //!< INTEL Supplemental Streaming SIMD Extension 3
    VmbSupportState_t AMD3DNow;       //!< AMD 3DNow
} VmbTechInfo_t;

/**
 * \brief API info types
 */
typedef enum VmbAPIInfo
{
    VmbAPIInfoAll,          //!< All the info (platform, build type and technologies)
    VmbAPIInfoPlatform,     //!< Platform the api was build for
    VmbAPIInfoBuild,        //!< build type (debug or release)
    VmbAPIInfoTechnology,   //!< info about special technologies uses in building the API
    VmbAPIInfoLast
} VmbAPIInfo;

/**
 * \brief API info type; for values see ::VmbAPIInfo
 */
typedef VmbUint32_t VmbAPIInfo_t;

/**
 * \}
 */

/**
 * \name Pixel Access Structs
 * \defgroup Pixel Access Structs
 * \{
 */

/**
 * \brief Structure for accessing data in 12-bit transfer mode.
 *
 * Two pixel are coded into 3 bytes.
 */
typedef struct Vmb12BitPackedPair_t
{
    VmbUint8_t   m_nVal8_1       ;   //!< High byte of the first Pixel
    VmbUint8_t   m_nVal8_1Low : 4;   //!< Low nibble of the first pixel
    VmbUint8_t   m_nVal8_2Low : 4;   //!< Low nibble of the second pixel
    VmbUint8_t   m_nVal8_2       ;   //!< High byte of the second pixel
} Vmb12BitPackedPair_t;

/**
 * \brief Struct for accessing data of a 8 bit grayscale image stored as unsigned integer.
 * 
 * This corresponds to ::VmbPixelFormatMono8.
 */
typedef struct VmbMono8_t
{
#ifdef __cplusplus
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t Y;   //!< gray part
} VmbMono8_t;

/**
 * \brief Struct for accessing data of a 8 bit grayscale image stored as signed integer.
 */
typedef struct VmbMono8s_t
{
#ifdef __cplusplus
    typedef VmbInt8_t value_type;
#endif
    VmbInt8_t Y;   //!< gray part
} VmbMono8s_t;

/**
 * \brief Struct for accessing pixel data of a 10 bit mono padded buffer.
 * 
 * The pixel data is LSB aligned and little endianness encoded on little endian systems.
 * 
 * The pixel data is MSB aligned and big endianness encoded on big endian systems.
 * 
 * On little endian systems this corresponds to ::VmbPixelFormatMono10.
 */
typedef struct VmbMono10_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t Y;   //!< gray part
} VmbMono10_t;

/**
 * \brief Struct for accessing pixel data of a 12 bit mono padded buffer.
 *
 * The pixel data is LSB aligned and little endianness encoded on little endian systems.
 * 
 * The pixel data is MSB aligned and big endianness encoded on big endian systems.
 * 
 * On little endian systems this corresponds to ::VmbPixelFormatMono12.
 */
typedef struct VmbMono12_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t Y;   //!< gray part
} VmbMono12_t;

/**
 * \brief Struct for accessing pixel data of a 14 bit mono padded buffer.
 * 
 * The pixel data is LSB aligned and little endianness encoded on little endian systems.
 * 
 * The pixel data is MSB aligned and big endianness encoded on big endian systems.
 * 
 * On little endian systems this corresponds to ::VmbPixelFormatMono14.
 */
typedef struct VmbMono14_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t Y;   //!< gray part
} VmbMono14_t;

/**
 * \brief Struct for accessing 16 bit grayscale image stored as unsigned integer.
 * 
 * The pixel data is LSB aligned and little endianness encoded on little endian systems.
 * 
 * The pixel data is MSB aligned and big endianness encoded on big endian systems.
 * 
 * On little endian systems this corresponds to ::VmbPixelFormatMono16.
 */
typedef struct VmbMono16_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t Y;   //!< gray part
} VmbMono16_t;

/**
 * \brief Struct for accessing 16 bit grayscale image stored as signed integer.
 */
typedef struct VmbMono16s_t
{
#ifdef __cplusplus
    typedef VmbInt16_t value_type;
#endif
    VmbInt16_t Y;   //!< gray part
} VmbMono16s_t;

/**
 * \brief Structure for accessing RGB data using 8 bit per channel.
 * 
 * This corresponds to ::VmbPixelFormatRgb8
 */
typedef struct VmbRGB8_t
{
#ifdef __cplusplus
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t R;   //!< red part
    VmbUint8_t G;   //!< green part
    VmbUint8_t B;   //!< blue part
} VmbRGB8_t;

/**
 * \brief Structure for accessing RGB data using 10 bit per channel padded to 16 bit; 48 bits per pixel are used in total.
 * 
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 * 
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 * 
 * Corresponds to ::VmbPixelFormatRgb10 on little endian systems.
 */
typedef struct VmbRGB10_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t R;   //!< red part
    VmbUint16_t G;   //!< green part
    VmbUint16_t B;   //!< blue part
} VmbRGB10_t;

/**
 * \brief Structure for accessing RGB data using 12 bit per channel padded to 16 bit; 48 bits per pixel are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatRgb12 on little endian systems.
 */
typedef struct VmbRGB12_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t R;   //!< red part
    VmbUint16_t G;   //!< green part
    VmbUint16_t B;   //!< blue part
} VmbRGB12_t;

/**
 * \brief Structure for accessing RGB data using 14 bit per channel padded to 16 bit; 48 bits per pixel are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatRgb14 on little endian systems.
 */
typedef struct VmbRGB14_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t R;   //!< red part
    VmbUint16_t G;   //!< green part
    VmbUint16_t B;   //!< blue part
} VmbRGB14_t;

/**
 * \brief Struct for accessing RGB pixels stored as 16 bit unsigend integer per channel.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatRgb16 on little endian systems.
 */
typedef struct VmbRGB16_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t R;   //!< red part
    VmbUint16_t G;   //!< green part
    VmbUint16_t B;   //!< blue part
} VmbRGB16_t;

/**
 * \brief Structure for accessing BGR data using 8 bit per channel.
 * 
 * Corresponds to ::VmbPixelFormatBgr8
 */
typedef struct VmbBGR8_t
{
#ifdef __cplusplus
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t B;   //!< blue part
    VmbUint8_t G;   //!< green part
    VmbUint8_t R;   //!< red part
} VmbBGR8_t;

/**
 * \brief Structure for accessing BGR data using 10 bit per channel padded to 16 bit; 48 bits per pixel are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatBgr10 on little endian systems.
 */
typedef struct VmbBGR10_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t B;   //!< blue part
    VmbUint16_t G;   //!< green part
    VmbUint16_t R;   //!< red part
} VmbBGR10_t;

/**
 * \brief Structure for accessing BGR data using 12 bit per channel padded to 16 bit; 48 bits per pixel are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatBgr12 on little endian systems.
 */
typedef struct VmbBGR12_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t B;   //!< blue part
    VmbUint16_t G;   //!< green part
    VmbUint16_t R;   //!< red part
} VmbBGR12_t;

/**
 * \brief Structure for accessing BGR data using 14 bit per channel padded to 16 bit; 48 bits per pixel are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatBgr14 on little endian systems.
 */
typedef struct VmbBGR14_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t B;   //!< blue part
    VmbUint16_t G;   //!< green part
    VmbUint16_t R;   //!< red part
} VmbBGR14_t;

/**
 * \brief Structure for accessing BGR data using 16 bit per channel; 48 bits per pixel are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatBgr16 on little endian systems.
 */
typedef struct VmbBGR16_t
{
#ifdef __cplusplus
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t B;  //!< blue part
    VmbUint16_t G;  //!< green part
    VmbUint16_t R;  //!< red part
} VmbBGR16_t;

/**
 * \brief Structure for accessing RGBA data using 8 bit per channel.
 */
typedef struct VmbRGBA8_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t R;   //!< red part
    VmbUint8_t G;   //!< green part
    VmbUint8_t B;   //!< blue part
    VmbUint8_t A;   //!< unused
} VmbRGBA8_t;

/**
 * \brief Alias for ::VmbRGBA8_t
 */
typedef VmbRGBA8_t VmbRGBA32_t;

/**
 * \brief Structure for accessing BGRA data using 8 bit per channel.
 *
 * This corresponds to ::VmbPixelFormatBgra8
 */
typedef struct VmbBGRA8_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t B;   //!< blue part
    VmbUint8_t G;   //!< green part
    VmbUint8_t R;   //!< red part
    VmbUint8_t A;   //!< unused
} VmbBGRA8_t;

/**
 * \brief Alias for ::VmbBGRA8_t
 */
typedef VmbBGRA8_t VmbBGRA32_t;

/**
 * \brief Struct for accessing ARGB values stored using a 8 bit unsigned integer per channel.
 * 
 * Corresponds to ::VmbPixelFormatArgb8
 */
typedef struct VmbARGB8_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t A;   //!< unused
    VmbUint8_t R;   //!< red part
    VmbUint8_t G;   //!< green part
    VmbUint8_t B;   //!< blue part
} VmbARGB8_t;

/**
 * \brief Alias for ::VmbARGB8_t 
 */
typedef VmbARGB8_t VmbARGB32_t;

/**
 * \brief Structure for accessing BGRA data using a 8 bit unsigned integer per channel.
 */
typedef struct VmbABGR8_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t A;   //!< unused
    VmbUint8_t B;   //!< blue part
    VmbUint8_t G;   //!< green part
    VmbUint8_t R;   //!< red part
} VmbABGR8_t;

/**
 * \brief Alias for ::VmbABGR8_t
 */
typedef VmbABGR8_t VmbABGR32_t;

/**
 * \brief Structure for accessing RGBA data using 10 bit per channel padded to 16 bit; 64 bit are used in total.
 * 
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatRgba10 on little endian systems.
 */
typedef struct VmbRGBA10_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t R;   //!< red part
    VmbUint16_t G;   //!< green part
    VmbUint16_t B;   //!< blue part
    VmbUint16_t A;   //!< unused
} VmbRGBA10_t;

/**
 * \brief Structure for accessing BGRA data using 10 bit per channel padded to 16 bit; 64 bit are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatBgra10 on little endian systems.
 */
typedef struct VmbBGRA10_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t B;   //!< blue part
    VmbUint16_t G;   //!< green part
    VmbUint16_t R;   //!< red part
    VmbUint16_t A;   //!< unused
} VmbBGRA10_t;

/**
 * \brief Structure for accessing RGBA data using 12 bit per channel padded to 16 bit; 64 bit are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatRgba12 on little endian systems.
 */
typedef struct VmbRGBA12_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t R;   //!< red part
    VmbUint16_t G;   //!< green part
    VmbUint16_t B;   //!< blue part
    VmbUint16_t A;   //!< unused
} VmbRGBA12_t;

/**
 * \brief Structure for accessing RGBA data using 14 bit per channel padded to 16 bit; 64 bit are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatRgba14 on little endian systems.
 */
typedef struct VmbRGBA14_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t R;   //!< red part
    VmbUint16_t G;   //!< green part
    VmbUint16_t B;   //!< blue part
    VmbUint16_t A;   //!< unused
} VmbRGBA14_t;

/**
 * \brief Structure for accessing BGRA data using 12 bit per channel padded to 16 bit; 64 bit are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatBgra12 on little endian systems.
 */
typedef struct VmbBGRA12_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t B;   //!< blue part
    VmbUint16_t G;   //!< green part
    VmbUint16_t R;   //!< red part
    VmbUint16_t A;   //!< unused
} VmbBGRA12_t;

/**
 * \brief Structure for accessing BGRA data using 14 bit per channel padded to 16 bit; 64 bit are used in total.
 *
 * Each channel is LSB aligned and little endianness encoded on little endian systems.
 *
 * Each channel is MSB aligned and big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatBgra14 on little endian systems.
 */
typedef struct VmbBGRA14_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t B;   //!< blue part
    VmbUint16_t G;   //!< green part
    VmbUint16_t R;   //!< red part
    VmbUint16_t A;   //!< unused
} VmbBGRA14_t;

/**
 * \brief Structure for accessing RGBA data using 16 bit per channel.
 *
 * Each channel is little endianness encoded on little endian systems.
 *
 * Each channel is big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatRgba16 on little endian systems.
 */
typedef struct VmbRGBA16_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t R;   //!< red part
    VmbUint16_t G;   //!< green part
    VmbUint16_t B;   //!< blue part
    VmbUint16_t A;   //!< unused
} VmbRGBA16_t;

/**
 * \brief Alias for ::VmbRGBA16_t
 */
typedef VmbRGBA16_t VmbRGBA64_t;

/**
 * \brief Structure for accessing BGRA data using 16 bit per channel.
 *
 * Each channel is little endianness encoded on little endian systems.
 *
 * Each channel is big endianness encoded on big endian systems.
 *
 * Corresponds to ::VmbPixelFormatBgra16 on little endian systems.
 */
typedef struct VmbBGRA16_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint16_t value_type;
#endif
    VmbUint16_t B;   //!< blue part
    VmbUint16_t G;   //!< green part
    VmbUint16_t R;   //!< red part
    VmbUint16_t A;   //!< unused
} VmbBGRA16_t;

/**
 * \brief Alias for ::VmbBGRA64_t
 */
typedef VmbBGRA16_t VmbBGRA64_t;

/**
 * \brief Structure for accessing data in the YUV 4:4:4 format (YUV) prosilica component order.
 * 
 * Corresponds to ::VmbPixelFormatYuv444
 */
typedef struct VmbYUV444_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t U;   //!< U
    VmbUint8_t Y;   //!< Luma
    VmbUint8_t V;   //!< V
} VmbYUV444_t;

/**
 * \brief Structure for accessing data in the YUV 4:2:2 format (UYVY)
 * 
 * This struct provides data for 2 pixels (Y0, U, V) and (Y1, U, V)
 * 
 * Corresponds to ::VmbPixelFormatYuv422
 */
typedef struct VmbYUV422_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t U;   //!< the U part for both pixels
    VmbUint8_t Y0;  //!< the intensity of the first pixel
    VmbUint8_t V;   //!< the V part for both pixels
    VmbUint8_t Y1;  //!< the intensity of the second pixel
} VmbYUV422_t;

/**
 * \brief Structure for accessing data in the YUV 4:1:1 format (UYYVYY)
 * 
 * This struct provides data for 2 pixels (Y0, U, V), (Y1, U, V), (Y2, U, V) and (Y3, U, V)
 * 
 * Corresponds to ::VmbPixelFormatYuv411
 */
typedef struct VmbYUV411_t
{
#ifdef __cplusplus
    /**
     * \brief The data type used to store one channel.
     */
    typedef VmbUint8_t value_type;
#endif
    VmbUint8_t U;   //!< the U part for all four pixels
    VmbUint8_t Y0;  //!< the intensity of the first pixel
    VmbUint8_t Y1;  //!< the intensity of the second pixel
    VmbUint8_t V;   //!< the V part for all four pixels
    VmbUint8_t Y2;  //!< the intensity of the third pixel
    VmbUint8_t Y3;  //!< the intensity of the fourth pixel
} VmbYUV411_t;

/**
 * \}
 */

/**
 * \brief Image pixel layout information.
 */
typedef enum VmbPixelLayout
{
    VmbPixelLayoutMono,                                         //!< Monochrome pixel data; pixels are padded, if necessary. 
    VmbPixelLayoutMonoPacked,                                   //!< Monochrome pixel data; pixels some bytes contain data for more than one pixel.
    VmbPixelLayoutRaw,                                          //!< Some Bayer pixel format where pixels each byte contains only data for a single pixel.
    VmbPixelLayoutRawPacked,                                    //!< Some Bayer pixel format where some bytes contain data for more than one pixel.
    VmbPixelLayoutRGB,                                          //!< Non-packed RGB data in channel order R, G, B
    VmbPixelLayoutBGR,                                          //!< Non-packed RGB data in channel order B, G, R
    VmbPixelLayoutRGBA,                                         //!< Non-packed RGBA data in channel order R, G, B, A
    VmbPixelLayoutBGRA,                                         //!< Non-packed RGBA data in channel order B, G, R, A
    VmbPixelLayoutYUV411_UYYVYY,                                //!< YUV data; pixel order for 4 pixels is U, Y0, Y1, V, Y2, Y3
    VmbPixelLayoutYUV411_YYUYYV,                                //!< YUV data; pixel order for 4 pixels is Y0, Y1, U, Y2, Y3, V
    VmbPixelLayoutYUV422_UYVY,                                  //!< YUV data; pixel order for 2 pixels is U, Y0, V, Y1
    VmbPixelLayoutYUV422_YUYV,                                  //!< YUV data; pixel order for 2 pixels is Y0, U, Y1, V
    VmbPixelLayoutYUV444_UYV,                                   //!< YUV data; pixel order is U, Y, V
    VmbPixelLayoutYUV444_YUV,                                   //!< YUV data; pixel order is Y, U, V
    VmbPixelLayoutMonoP,                                        //!< Monochrome pixel data; pixels are padded, if necessary. \todo What is the difference to VmbPixelLayoutMono?
    VmbPixelLayoutMonoPl,                                       //!< \todo unused, remove?
    VmbPixelLayoutRawP,                                         //!< Some Bayer pixel format where pixels each byte contains only data for a single pixel. \todo What's the difference to VmbPixelLayoutRawPacked?
    VmbPixelLayoutRawPl,                                        //!< \todo unused, remove?
    VmbPixelLayoutYYCbYYCr411 = VmbPixelLayoutYUV411_YYUYYV,    //!< Alias for ::VmbPixelLayoutYUV411_YYUYYV
    VmbPixelLayoutCbYYCrYY411 = VmbPixelLayoutYUV411_UYYVYY,    //!< Alias for ::VmbPixelLayoutYUV411_UYYVYY
    VmbPixelLayoutYCbYCr422 = VmbPixelLayoutYUV422_YUYV,        //!< Alias for ::VmbPixelLayoutYUV422_YUYV
    VmbPixelLayoutCbYCrY422 = VmbPixelLayoutYUV422_UYVY,        //!< Alias for ::VmbPixelLayoutYUV422_UYVY
    VmbPixelLayoutYCbCr444 = VmbPixelLayoutYUV444_YUV,          //!< Alias for ::VmbPixelLayoutYUV444_YUV
    VmbPixelLayoutCbYCr444 = VmbPixelLayoutYUV444_UYV,          //!< Alias for ::VmbPixelLayoutYUV444_UYV

    VmbPixelLayoutLAST,
} VmbPixelLayout;

/**
 * \brief Image pixel layout information; for values see ::VmbPixelLayout
 */
typedef VmbUint32_t VmbPixelLayout_t;

/**
 * \brief Image color space information.
 */
typedef enum VmbColorSpace
{
    VmbColorSpaceUndefined,
    VmbColorSpaceITU_BT709, //!< \todo color space description
    VmbColorSpaceITU_BT601, //!< \todo color space description

} VmbColorSpace;

/**
 * \brief Image color space information; for values see ::VmbColorSpace
 */
typedef VmbUint32_t VmbColorSpace_t;

/**
 * \brief Image pixel information
 */
typedef struct VmbPixelInfo
{
    VmbUint32_t         BitsPerPixel;   //!< The number of bits used to store the data for one pixel
    VmbUint32_t         BitsUsed;       //!< The number of bits that actually contain data.
    VmbAlignment_t      Alignment;      //!< Indicates, if the most significant or the least significant bit is filled for pixel formats not using all bits of the buffer to store data.
    VmbEndianness_t     Endianness;     //!< Endianness of the pixel data
    VmbPixelLayout_t    PixelLayout;    //!< Channel order, and in case of YUV formats relative order.
    VmbBayerPattern_t   BayerPattern;   //!< The bayer pattern
    VmbColorSpace_t     Reserved;       //!< Unused member reserved for future use.
} VmbPixelInfo;

/**
 * \brief Struct containing information about the image data in the Data member of a ::VmbImage.
 */
typedef struct VmbImageInfo
{
    VmbUint32_t     Width;      //!< The width of the image in pixels
    VmbUint32_t     Height;     //!< The height of the image in pixels
    VmbInt32_t      Stride;     //!< \todo description; do we actually use this
    VmbPixelInfo    PixelInfo;  //!< Information about the pixel format
} VmbImageInfo;

/**
 * \brief vmb image type
 */
typedef struct VmbImage
{
    VmbUint32_t     Size;       //!< The size of this struct; If set incorrectly, API functions will return ::VmbErrorStructSize
    void*           Data;       //!< The image data
    VmbImageInfo    ImageInfo;  //!< Information about pixel format, size, and stride of the image.

} VmbImage;

/**
 * \brief Transform info for special debayering modes.
 */
typedef enum VmbDebayerMode
{
    VmbDebayerMode2x2,      //!< \todo description
    VmbDebayerMode3x3,      //!< \todo description
    VmbDebayerModeLCAA,     //!< \todo description
    VmbDebayerModeLCAAV,    //!< \todo description
    VmbDebayerModeYUV422,   //!< \todo description
} VmbDebayerMode;

/**
 * \brief Transform info for special debayering mode; for values see ::VmbDebayerMode
 */
typedef VmbUint32_t  VmbDebayerMode_t;

/**
 * \name Transformation Parameters
 * \defgroup Transformation Parameters
 * \{
 */

/**
 * \brief Transform parameter types.
 */
typedef enum VmbTransformType
{
    VmbTransformTypeNone,                   //!< Invalid type
    VmbTransformTypeDebayerMode,            //!< Debayering mode
    VmbTransformTypeColorCorrectionMatrix,  //!< Color correction matrix
    VmbTransformTypeGammaCorrection,        //!< Gamma correction
    VmbTransformTypeOffset,                 //!< Offset
    VmbTransformTypeGain,                   //!< Gain
} VmbTransformType;

/**
 * \brief Transform parameter type; for avalues see ::VmbTransformType
 */
typedef VmbUint32_t VmbTransformType_t;

/**
 * \brief Struct definition for holding the debayering mode.
 *
 * The struct is used to pass the data to ::VmbImageTransform via transform parameter.
 * It corresponds to the ::VmbTransformTypeDebayerMode parameter type.
 */
typedef struct VmbTransformParameteDebayer
{
    VmbDebayerMode_t  Method; //!< The DeBayering method to use.
} VmbTransformParameterDebayer;


/**
 * \brief Transform info for color correction using a 3x3 matrix multiplication.
 *
 * The struct is used to pass the data to ::VmbImageTransform via transform parameter.
 * It corresponds to the ::VmbTransformTypeColorCorrectionMatrix parameter type.
 *
 * \todo what does each index represent; how to get from 2d to 1d?
 */
typedef struct VmbTransformParameterMatrix3x3
{
    VmbFloat_t          Matrix[9]; //!< The color correction matrix to use for the transformation.
} VmbTransformParameterMatrix3x3;

/**
 * \brief Struct definition for a gamma value.
 *
 * This is currently not supported by ::VmbImageTransform.
 * It corresponds to the ::VmbTransformTypeGammaCorrection parameter type.
 */
typedef struct VmbTransformParameterGamma
{
    VmbFloat_t          Gamma; //!< The gamma value to use for the transformation
} VmbTransformParameterGamma;

/**
 * \brief Struct definition for holding the offset to pass via transform parameter.
 *
 * The struct is used to pass the data to ::VmbImageTransform via transform parameter.
 * It corresponds to the ::VmbTransformTypeOffset parameter type.
 */
typedef struct VmbTransformParameterOffset
{
    VmbInt32_t Offset; //!< The offset to use for the transformation.
} VmbTransformParameterOffset;

/**
 * \brief Struct definition for holding the gain value.
 *
 * The struct is used to pass the data to ::VmbImageTransform via transform parameter.
 * It corresponds to the ::VmbTransformTypeGain parameter type.
 */
typedef struct VmbTransformParameterGain
{
    VmbUint32_t Gain; //!< The gain to use for the transformation
} VmbTransformParameterGain;

/**
 * \brief Union for possible transformation parameter types.
 *
 * Each possible data type corresponds to a constant of ::VmbTransformType.
 */
typedef union VmbTransformParameter
{
    VmbTransformParameterMatrix3x3  Matrix3x3;  //!< A matrix with 3 rows and 3 columns.
    VmbTransformParameterDebayer    Debayer;    //!< A debayering mode
    VmbTransformParameterGamma      Gamma;      //!< A gamma value
    VmbTransformParameterOffset     Offset;     //!< \todo offset (is this even used?)
    VmbTransformParameterGain       Gain;       //!< A gain value
} VmbTransformParameter;

/**
 * \}
 */

/**
 * \brief Transform info interface structure.
 */
typedef struct VmbTransformInfo
{
    VmbTransformType_t      TransformType;  //!< The type of the information stored in the Parameter member.
    VmbTransformParameter   Parameter;      //!< The parameter data.
} VmbTransformInfo;

#endif // VMB_TRANSFORM_TYPES_H_
