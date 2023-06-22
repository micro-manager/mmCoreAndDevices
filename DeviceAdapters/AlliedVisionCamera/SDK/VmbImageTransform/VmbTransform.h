/*=============================================================================
  Copyright (C) 2012 - 2021 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        VmbTransform.h

  Description: Definition of image transform functions for the Vmb APIs.

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
#ifndef VMB_TRANSFORM_H_
#define VMB_TRANSFORM_H_
#ifndef VMB_TRANSFORM
#define VMB_TRANSFORM
#endif

#include <VmbImageTransform/VmbTransformTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VMBIMAGETRANSFORM_API
#   ifndef VMB_NO_EXPORT
#       ifdef VMB_EXPORTS
#           if defined(__ELF__) && (defined(__clang__) || defined(__GNUC__))
#               define VMBIMAGETRANSFORM_API __attribute__((visibility("default")))
#           elif defined( __APPLE__ ) || defined(__MACH__)
#               define VMBIMAGETRANSFORM_API __attribute__((visibility("default")))
#           else
#               ifndef _WIN64
#                   define VMBIMAGETRANSFORM_API __declspec(dllexport) __stdcall
#               else
#                   define VMBIMAGETRANSFORM_API __stdcall
#               endif
#           endif
#       else
#           if defined (__ELF__) && (defined(__clang__) || defined(__GNUC__))
#               define VMBIMAGETRANSFORM_API
#           elif defined( __APPLE__ ) || defined(__MACH__)
#               define VMBIMAGETRANSFORM_API
#           else
#               define VMBIMAGETRANSFORM_API __declspec(dllimport) __stdcall
#           endif
#       endif
#   else
#       define VMBIMAGETRANSFORM_API
#   endif
#endif

/**
 * \brief Inquire the library version.
 *
 * \param[out] value      Contains the library version (Major,Minor,Sub,Build).
 *
 * This function can be called at anytime, even before the library is initialized.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   \p value is null.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbGetImageTransformVersion ( VmbUint32_t*  value );

/**
 * \brief Get information about processor supported features.
 *
 * This should be called before using any SIMD (MMX,SSE) optimized functions.
 *
 * \param[out] technoInfo    Returns the supported SIMD technologies.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   If \p technoInfo is null.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbGetTechnoInfo( VmbTechInfo_t*  technoInfo );

/**
 * \brief Translate an Vmb error code to a human-readable string.
 *
 * \param[in]  errorCode       The error code to get a readable string for.
 * \param[out] info            Pointer to a zero terminated string the error description is written to.
 * \param[in]  maxInfoLength   The size of the \p info buffer.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   \p info is null, or if maxInfoLength is 0.
 *
 * \retval ::VmbErrorMoreData       If \p maxInfoLength is too small to hold the complete information.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbGetErrorInfo(VmbError_t     errorCode,
                                                 VmbANSIChar_t* info,
                                                 VmbUint32_t    maxInfoLength );

/**
 * \brief Get information about the currently loaded Vmb ImageTransform API.
 *
 *
 * \p infoType may be one of the following values:
 *  - ::VmbAPIInfoAll:         Returns all information about the API
 *  - ::VmbAPIInfoPlatform:    Returns information about the platform the API was built for (x86 or x64)
 *  - ::VmbAPIInfoBuild:       Returns info about the API built (debug or release)
 *  - ::VmbAPIInfoTechnology:  Returns info about the supported technologies the API was built for (OpenMP or OpenCL)
 *
 * \param[in]   infoType        Type of information to return
 * \param[out]  info            Pointer to a zero terminated string that
 * \param[in]   maxInfoLength   The length of the \p info buffer
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   \p info is null.
 *
 * \retval ::VmbErrorMoreData       If chars are insufficient \p maxInfoLength to hold the complete information.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbGetApiInfoString(VmbAPIInfo_t   infoType,
                                                     VmbANSIChar_t* info,
                                                     VmbUint32_t    maxInfoLength );

/**
 * \brief Set transformation options to a predefined debayering mode.
 *
 * The default mode is 2x2 debayering. Debayering modes only work for image widths and heights
 * divisible by two.
 *
 * \param[in]     debayerMode     The mode used for debayering the raw source image.
 *
 * \param[in,out] transformInfo   Parameter that contains information about special
 *                                transform functionality
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   \p transformInfo is null.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbSetDebayerMode(VmbDebayerMode_t     debayerMode,
                                                   VmbTransformInfo*    transformInfo );

/**
 * \brief Set transformation options to a 3x3 color matrix transformation.
 *
 * \param[in]       matrix          Color correction matrix.
 *
 * \param[in,out]   transformInfo   Parameter that is filled with information
 *                                  about special transform functionality.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   If \p matrix or \p transformInfo are null.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbSetColorCorrectionMatrix3x3(const VmbFloat_t*   matrix,
                                                                VmbTransformInfo*   transformInfo );

/**
 * \brief Initialize the give VmbTransformInfo with gamma correction information.
 *
 * \param[in]       gamma           Float gamma correction to set
 * \param[in,out]   transformInfo   Transform info to set gamma correction to
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   \p transformInfo is null.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbSetGammaCorrection(VmbFloat_t           gamma,
                                                       VmbTransformInfo*    transformInfo );

/**
 * \brief Set the pixel related info of a VmbImage to the values appropriate for the given pixel format.
 *
 * A VmbPixelFormat_t can be obtained from Vmb C/C++ APIs frame.
 * For displaying images, it is suggested to use ::VmbSetImageInfoFromString() or to look up
 * a matching VmbPixelFormat_t.
 *
 * \param[in]       pixelFormat     The pixel format describes the pixel format to be used.
 * \param[in]       width           The width of the image in pixels.
 * \param[in]       height          The height of the image in pixels.
 * \param[in,out]   image           A pointer to the image struct to write the info to.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter  If \p image is null or one of the members of \p image is invalid.
 *
 * \retval ::VmbErrorStructSize    If the Size member of the image is incorrect.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbSetImageInfoFromPixelFormat(VmbPixelFormat_t    pixelFormat,
                                                                VmbUint32_t         width,
                                                                VmbUint32_t         height,
                                                                VmbImage*           image);

/**
 * \brief Set image info member values in VmbImage from string.
 *
 * This function does not read or write to VmbImage::Data member.
 *
 * \param[in]       imageFormat     The string containing the image format. This parameter is case insensitive.
 * \param[in]       width           The width of the image in pixels.
 * \param[in]       height          The height of the image in pixels.
 * \param[in,out]   image           A pointer to the image struct to write the info to.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   \p imageFormat or \p image are null.
 *
 * \retval ::VmbErrorStructSize     The Size member of \p image contains an invalid value.
 *
 * \retval ::VmbErrorResources      The function ran out of memory while processing.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbSetImageInfoFromString(const VmbANSIChar_t* imageFormat,
                                                           VmbUint32_t          width,
                                                           VmbUint32_t          height,
                                                           VmbImage*            image);

/**
 * \brief Set output image dependent on the input image, user specifies pixel layout and bit depth of the out format.
 *
 * \param[in]    inputPixelFormat    Input Vmb pixel format
 * \param[in]    width               width of the output image
 * \param[in]    height              height of the output image
 * \param[in]    outputPixelLayout   pixel component layout for output image
 * \param[in]    bitsPerPixel        bit depth of output 8 and 16 supported
 * \param[out]   outputImage         The output image to write the compatible format to.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess            The call was successful.
 *
 * \retval ::VmbErrorBadParameter       \p outputImage is null.
 *
 * \retval ::VmbErrorStructSize         The Size member of \p outputImage contains an invalid size.
 *
 * \retval ::VmbErrorNotImplemented     No suitable transformation is implemented.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbSetImageInfoFromInputParameters(VmbPixelFormat_t    inputPixelFormat,
                                                                    VmbUint32_t         width,
                                                                    VmbUint32_t         height,
                                                                    VmbPixelLayout_t    outputPixelLayout,
                                                                    VmbUint32_t         bitsPerPixel,
                                                                    VmbImage*           outputImage);

/**
 * \brief Set output image compatible to input image with given layout and bit depth.
 *        The output image will have same dimensions as the input image.
 *
 * \param[in]   inputImage          The input image with fully initialized image info elements.
 * \param[in]   outputPixelLayout   The desired layout for the output image.
 * \param[in]   bitsPerPixel        The desided bit depth for output image. 8 bit and 16 bit are supported.
 * \param[out]  outputImage         The output image to write the compatible format to.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess            The call was successful.
 *
 * \retval ::VmbErrorBadParameter       \p inputImage or \p outputImage are null,
 *                                      or the PixelInfo member of the ImageInfo member of the \p inputImage does not correspond to a supported pixel format.
 *
 * \retval ::VmbErrorStructSize         The Size member of \p outputImage contains an invalid size.
 *
 * \retval ::VmbErrorNotImplemented     No suitable transformation is implemented.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbSetImageInfoFromInputImage(const VmbImage*  inputImage,
                                                               VmbPixelLayout_t outputPixelLayout,
                                                               VmbUint32_t      bitsPerPixel,
                                                               VmbImage*        outputImage);

/**
 * \brief Transform an image from one pixel format to another providing additional transformation options, if necessary.
 *
 * The transformation is defined by the provided images and the \p parameter.
 *
 * Create the source and destination image info structure with VmbSetImageInfoFromPixelFormat
 * or VmbSetimageInfoFromString and keep those structures as template.
 * For calls to transform, simply attach the image to the Data member.
 * The optional parameters, when set, are constraints on the transform.
 *
 * \param[in]       source          The pointer to source image.
 * \param[in,out]   destination     The pointer to destination image.
 * \param[in]       parameter       An array of transform parameters; may be null.
 * \param[in]       parameterCount  The number of transform parameters.
 *
 * \return An error code indicating success or the type of error.
 *
 * \retval ::VmbErrorSuccess        The call was successful.
 *
 * \retval ::VmbErrorBadParameter   if any image pointer or their "Data" members is NULL, or
 *                                  if "Width" or "Height" don't match between source and destination, or
 *                                  if one of the parameters for the conversion does not fit
 *
 * \retval ::VmbErrorStructSize     The Size member of \p source or \p destination contain an invalid value.
 *
 * \retval ::VmbErrorNotImplemented The transformation from the format of \p source to the format of \p destination is not implemented.
 */
VmbError_t VMBIMAGETRANSFORM_API VmbImageTransform(const VmbImage*          source,
                                                   VmbImage*                destination,
                                                   const VmbTransformInfo*  parameter,
                                                   VmbUint32_t              parameterCount);

#ifdef __cplusplus
}
#endif

#endif // VMB_TRANSFORM_H_
