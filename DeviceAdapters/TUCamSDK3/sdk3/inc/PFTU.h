/**
 * @copyright Copyright (C) Tucsen Photonics Co., Ltd. 2012-2025. All rights reserved.
 * @brief Tucsen Custom Pixel Format Definitions - PFTU (Pixel Format Tucsen)
 *
 * This file defines custom pixel format identifiers for Tucsen Photonics SDK,
 * providing extended pixel format support beyond the standard PFNC (Pixel Format
 * Naming Convention) definitions. These formats are used for specialized image
 * formats and camera interfaces such as CoaXPress (CXP).
 *
 * The file contains pixel format definitions including:
 * - CXP (CoaXPress) pixel formats - Various monochrome, color, and specialized formats
 * - Custom YUV formats - Extended YUV format definitions
 * - Format categories:
 *   * Monochrome formats (8/10/12/14/16-bit)
 *   * Planar formats (1, 2, and 15-plane configurations)
 *   * Bayer color formats (GR, RG, GB, BG patterns)
 *   * RGB/RGBA formats
 *   * YUV formats (411/422/444 sampling)
 *   * YCbCr formats (601/709 standards with various sampling)
 *
 * These pixel format identifiers are used in conjunction with the standard PFNC
 * definitions to provide comprehensive pixel format support across all supported
 * camera interfaces and protocols.
 *
 * For more information, please visit: https://www.tucsen.com/
 *
 * @file PFTU.h
 * @author fandayu
 * @date 2025-04-11
 * @version v1.0.0
 * @note This header extends PFNC.h with Tucsen-specific pixel format definitions
 * @see PFNC.h
 * @since v1.0.0
 */
#include "PFNC.h"

#ifndef _PFTU_H_
#define _PFTU_H_

// CXP Format
#define CXP_Raw              0x0000 
#define CXP_Mono8            0x0101 
#define CXP_Mono10           0x0102 
#define CXP_Mono12           0x0103 
#define CXP_Mono14           0x0104 
#define CXP_Mono16           0x0105 
#define CXP_Planar1_8        0x0211 
#define CXP_Planar1_10       0x0212 
#define CXP_Planar1_12       0x0213 
#define CXP_Planar1_14       0x0214 
#define CXP_Planar1_16       0x0215 
#define CXP_Planar2_8        0x0221 
#define CXP_Planar2_10       0x0222 
#define CXP_Planar2_12       0x0223 
#define CXP_Planar2_14       0x0224 
#define CXP_Planar2_16       0x0225 
#define CXP_Planar15_8       0x02F1 
#define CXP_Planar15_10      0x02F2 
#define CXP_Planar15_12      0x02F3 
#define CXP_Planar15_14      0x02F4 
#define CXP_Planar15_16      0x02F5 
#define CXP_BayerGR8         0x0311 
#define CXP_BayerGR10        0x0312 
#define CXP_BayerGR12        0x0313 
#define CXP_BayerGR14        0x0314 
#define CXP_BayerGR16        0x0315 
#define CXP_BayerRG8         0x0321 
#define CXP_BayerRG10        0x0322 
#define CXP_BayerRG12        0x0323 
#define CXP_BayerRG14        0x0324 
#define CXP_BayerRG16        0x0325 
#define CXP_BayerGB8         0x0331 
#define CXP_BayerGB10        0x0332 
#define CXP_BayerGB12        0x0333 
#define CXP_BayerGB14        0x0334 
#define CXP_BayerGB16        0x0335 
#define CXP_BayerBG8         0x0341 
#define CXP_BayerBG10        0x0342 
#define CXP_BayerBG12        0x0343 
#define CXP_BayerBG14        0x0344 
#define CXP_BayerBG16        0x0345 
#define CXP_RGB8             0x0401 
#define CXP_RGB10            0x0402 
#define CXP_RGB12            0x0403 
#define CXP_RGB14            0x0404 
#define CXP_RGB16            0x0405 
#define CXP_RGBA8            0x0501 
#define CXP_RGBA10           0x0502 
#define CXP_RGBA12           0x0503 
#define CXP_RGBA14           0x0504 
#define CXP_RGBA16           0x0505 
#define CXP_YUV411_8         0x0611 
#define CXP_YUV411_10        0x0612 
#define CXP_YUV411_12        0x0613 
#define CXP_YUV411_14        0x0614 
#define CXP_YUV411_16        0x0615 
#define CXP_YUV422_8         0x0621 
#define CXP_YUV422_10        0x0622 
#define CXP_YUV422_12        0x0623 
#define CXP_YUV422_14        0x0624 
#define CXP_YUV422_16        0x0625 
#define CXP_YUV444_8         0x0631 
#define CXP_YUV444_10        0x0632 
#define CXP_YUV444_12        0x0633 
#define CXP_YUV444_14        0x0634 
#define CXP_YUV444_16        0x0635 
#define CXP_YCbCr601_411_8   0x0711 
#define CXP_YCbCr601_411_10  0x0712 
#define CXP_YCbCr601_411_12  0x0713 
#define CXP_YCbCr601_411_14  0x0714 
#define CXP_YCbCr601_411_16  0x0715 
#define CXP_YCbCr601_422_8   0x0721 
#define CXP_YCbCr601_422_10  0x0722 
#define CXP_YCbCr601_422_12  0x0723 
#define CXP_YCbCr601_422_14  0x0724 
#define CXP_YCbCr601_422_16  0x0725 
#define CXP_YCbCr601_444_8   0x0731 
#define CXP_YCbCr601_444_10  0x0732 
#define CXP_YCbCr601_444_12  0x0733 
#define CXP_YCbCr601_444_14  0x0734 
#define CXP_YCbCr601_444_16  0x0735 
#define CXP_YCbCr709_411_8   0x0811 
#define CXP_YCbCr709_411_10  0x0812 
#define CXP_YCbCr709_411_12  0x0813 
#define CXP_YCbCr709_411_14  0x0814 
#define CXP_YCbCr709_411_16  0x0815 
#define CXP_YCbCr709_422_8   0x0821 
#define CXP_YCbCr709_422_10  0x0822 
#define CXP_YCbCr709_422_12  0x0823 
#define CXP_YCbCr709_422_14  0x0824 
#define CXP_YCbCr709_422_16  0x0825 
#define CXP_YCbCr709_444_8   0x0831 
#define CXP_YCbCr709_444_10  0x0832 
#define CXP_YCbCr709_444_12  0x0833 
#define CXP_YCbCr709_444_14  0x0834 
#define CXP_YCbCr709_444_16  0x0835 

// Pixel Format
#define PFNC_YUV422_16  0x02100033 /* YUV 4:2:2 16-bit */

#endif /* _PFTU_H_ */
