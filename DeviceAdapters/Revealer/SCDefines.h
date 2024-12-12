#ifndef __SC_DEFINES_H__
#define __SC_DEFINES_H__

#ifdef WIN32
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif

#ifndef IN
#define IN ///< \~chinese 输入型参数       \~english Input param
#endif

#ifndef OUT
#define OUT ///< \~chinese 输出型参数       \~english Output param
#endif

#ifndef IN_OUT
#define IN_OUT ///< \~chinese 输入/输出型参数  \~english Input/Output param
#endif

#ifndef __cplusplus
typedef char bool;
#define true 1
#define false 0
#endif

/// \~chinese
/// \brief 错误码
/// \~english
/// \brief Error code
#define SC_OK 0                      ///< \~chinese 成功，无错误                         \~english Successed, no error
#define SC_ERROR -101                ///< \~chinese 通用的错误                           \~english Generic error
#define SC_INVALID_HANDLE -102       ///< \~chinese 错误或无效的句柄                     \~english Error or invalid handle
#define SC_INVALID_PARAM -103        ///< \~chinese 错误的参数                           \~english Incorrect parameter
#define SC_INVALID_FRAME_HANDLE -104 ///< \~chinese 错误或无效的帧句柄                   \~english Error or invalid frame handle
#define SC_INVALID_FRAME -105        ///< \~chinese 无效的帧                             \~english Invalid frame
#define SC_INVALID_RESOURCE -106     ///< \~chinese 相机/事件/流等资源无效               \~english Camera/Event/Stream and so on resource invalid
#define SC_INVALID_IP -107           ///< \~chinese 设备与主机的IP网段不匹配             \~english Device's and PC's subnet is mismatch
#define SC_NO_MEMORY -108            ///< \~chinese 内存不足                             \~english Malloc memery failed
#define SC_INSUFFICIENT_MEMORY -109  ///< \~chinese 传入的内存空间不足                   \~english Insufficient memory
#define SC_ERROR_PROPERTY_TYPE -110  ///< \~chinese 属性类型错误                         \~english Property type error
#define SC_INVALID_ACCESS -111       ///< \~chinese 属性不可访问、或不能读/写、或读/写失败   \~english Property not accessible, or not be read/written, or read/written failed
#define SC_INVALID_RANGE -112        ///< \~chinese 属性值超出范围、或者不是步长整数倍       \~english The property's value is out of range, or is not integer multiple of the step
#define SC_NOT_SUPPORT -113          ///< \~chinese 设备不支持的功能                     \~english Device not supported function
#define SC_NOT_IMPLEMENTED -114          ///< \~chinese 功能未实现                     \~english  function not implement
//#define SC_RESTORE_STREAM -114       ///< \~chinese 取图恢复中                         \~english Device restore stream
//#define SC_RECONNECT_DEVICE -115     ///< \~chinese 重连恢复中                         \~english Device reconnect
#define SC_TIMEOUT -115              ///< \~chinese 超时                                 \~english Timeout
#define SC_BUSY -116                 ////  \~chinese 处于忙碌状态                        \~english busy
#define SC_ACCESS_DENIED -117        ////  \~chinese 访问设备被拒绝                      \~english access denied
#define SC_INVALID_NODEMAP -118           ////  \~chinese nodemap非法或不存在                 \~english nodemap invalid or missing
#define SC_INVALID_ERRCODE -300

#define SC_MAX_DEVICE_ENUM_NUM 100 ///< \~chinese 支持设备最大个数     \~english The maximum number of supported devices
#define SC_MAX_STRING_LENTH 256    ///< \~chinese 字符串最大长度       \~english The maximum length of string
#define SC_MAX_ERROR_LIST_NUM 128  ///< \~chinese 失败属性列表最大长度 \~english The maximum size of failed properties list
#define SC_MAX_PATH_LENGTH 1024
#define SC_MAX_EVENT_LEN  1024

typedef void* SC_DEV_HANDLE;   ///< \~chinese 设备句柄             \~english Device handle
typedef void* SC_FRAME_HANDLE; ///< \~chinese 帧句柄               \~english Frame handle

/// \~chinese
///枚举：属性类型
/// \~english
///Enumeration: property type
typedef enum _SC_EFeatureType
{
    featureInt = 0x10000000,     ///< \~chinese 整型数               \~english Integer
    featureFloat = 0x20000000,   ///< \~chinese 浮点数               \~english Float
    featureEnum = 0x30000000,    ///< \~chinese 枚举                 \~english Enumeration
    featureBool = 0x40000000,    ///< \~chinese 布尔                 \~english Bool
    featureString = 0x50000000,  ///< \~chinese 字符串               \~english String
    featureCommand = 0x60000000, ///< \~chinese 命令                 \~english Command
    featureGroup = 0x70000000,   ///< \~chinese 分组节点             \~english Group Node
    featureReg = 0x80000000,     ///< \~chinese 寄存器节点           \~english Register Node

    featureUndefined = 0x90000000 ///< \~chinese 未定义               \~english Undefined
} SC_EFeatureType;

/// \~chinese
///枚举：接口类型
/// \~english
///Enumeration: interface type
typedef enum _SC_EInterfaceType
{
    eInterfaceTypeCXP = 0x00000001,  ///< \~chinese 网卡接口类型                             \~english NIC type
    eInterfaceTypeUsb3 = 0x00000002, ///< \~chinese USB3.0接口类型                           \~english USB3.0 interface type
    //eIinterfaceTypeCL = 0x00000004,           ///< \~chinese CAMERALINK接口类型                       \~english Cameralink interface type
    //eIinterfaceTypePCIe = 0x00000008,         ///< \~chinese PCIe接口类型                             \~english PCIe interface type
    eInterfaceTypeCustom = 0x0000004,
    eInterfaceTypeAll = 0x00000000,     ///< \~chinese 忽略接口类型（CAMERALINK接口除外）       \~english All types interface type（Excluding CAMERALINK）
    eIInterfaceInvalidType = 0xFFFFFFFF ///< \~chinese 无效接口类型                             \~english Invalid interface type
} SC_EInterfaceType;

/// \~chinese
///枚举：设备类型
/// \~english
///Enumeration: device type
typedef enum _SC_ECameraType
{
    eTypeGigeCamera = 0, ///< \~chinese GIGE相机             \~english GigE Vision Camera
    eTypeU3vCamera = 1,  ///< \~chinese USB3.0相机           \~english USB3.0 Vision Camera
    eTypeCLCamera = 2,   ///< \~chinese CAMERALINK 相机      \~english Cameralink camera
    eTypePCIeCamera = 3, ///< \~chinese PCIe相机             \~english PCIe Camera
    eTypeCXPCamera = 4,
    eTypeCustom = 5,
    eTypeVirtual = 6,
    eTypeUndefinedCamera = 255 ///< \~chinese 未知类型             \~english Undefined Camera
} SC_ECameraType;

/// \~chinese
///枚举：创建句柄方式
/// \~english
///Enumeration: Create handle mode
typedef enum _SC_ECreateHandleMode
{
    eModeByIndex = 0,    ///< \~chinese 通过已枚举设备的索引(从0开始，比如 0, 1, 2...)   \~english By index of enumerated devices (Start from 0, such as 0, 1, 2...)
    eModeByCameraKey,    ///< \~chinese 通过设备键"厂商:序列号"                          \~english By device's key "vendor:serial number"
    eModeByDeviceUserID, ///< \~chinese 通过设备自定义名                                 \~english By device userID
    eModeByIPAddress,    ///< \~chinese 通过设备IP地址                                   \~english By device IP address.
} SC_ECreateHandleMode;

/// \~chinese
///枚举：访问权限
/// \~english
///Enumeration: access permission
typedef enum _SC_ECameraAccessPermission
{
    eAccessPermissionUnknown = 0, ///< \~chinese 无法确定                                 \~english Value not known; indeterminate.
    eAccessNone = 1,
    eAccessPermissionMonitor,               ///< \~chinese 非独占访问权限,以读的模式打开设备        \~english Non-Exclusive Read Permission, open device with read mode
    eAccessPermissionControl,               ///< \~chinese 非独占控制权限,其他App允许读取所有寄存器 \~english Non-Exclusive Control Permission, allows other APP reading all registers
    eAccessPermissionExclusive,             ///< \~chinese 独占访问权限                             \~english Exclusive Access Permission
} SC_ECameraAccessPermission;

/// \~chinese
///枚举：抓图策略
/// \~english
///Enumeration: grab strartegy
typedef enum _SC_EGrabStrategy
{
    grabStrartegySequential = 0,    ///< \~chinese 按到达顺序处理图片   \~english The images are processed in the order of their arrival
    grabStrartegyLatestImage = 1,   ///< \~chinese 获取最新的图片       \~english Get latest image
    grabStrartegyUpcomingImage = 2, ///< \~chinese 等待获取下一张图片(只针对GigE相机)   \~english Waiting for next image(GigE only)
    grabStrartegyUndefined          ///< \~chinese 未定义               \~english Undefined
} SC_EGrabStrategy;

/// \~chinese
///枚举：流事件状态
/// \~english
/// Enumeration:stream event status
typedef enum _SC_EEventStatus
{
    streamEventNormal = 1,                    ///< \~chinese 正常流事件       \~english Normal stream event
    streamEventLostFrame = 2,                 ///< \~chinese 丢帧事件         \~english Lost frame event
    streamEventLostPacket = 3,                ///< \~chinese 丢包事件         \~english Lost packet event
    streamEventImageError = 4,                ///< \~chinese 图像错误事件     \~english Error image event
    streamEventStreamChannelError = 5,        ///< \~chinese 取流错误事件     \~english Stream channel error event
    streamEventTooManyConsecutiveResends = 6, ///< \~chinese 太多连续重传     \~english Too many consecutive resends event
    streamEventTooManyLostPacket = 7          ///< \~chinese 太多丢包         \~english Too many lost packet event
} SC_EEventStatus;

/// \~chinese
///枚举：图像转换Bayer格式所用的算法
/// \~english
/// Enumeration:alorithm used for Bayer demosaic
typedef enum _SC_EBayerDemosaic
{
    demosaicNearestNeighbor,  ///< \~chinese 最近邻           \~english Nearest neighbor
    demosaicBilinear,         ///< \~chinese 双线性           \~english Bilinear
    demosaicEdgeSensing,      ///< \~chinese 边缘检测         \~english Edge sensing
    demosaicNotSupport = 255, ///< \~chinese 不支持           \~english Not support
} SC_EBayerDemosaic;

/// \~chinese
///枚举：事件类型
/// \~english
/// Enumeration:event type
typedef enum _SC_EVType
{
    offLine, ///< \~chinese 设备离线通知     \~english device offline notification
    onLine,   ///< \~chinese 设备在线通知     \~english device online notification
} SC_EVType;

/// \~chinese
///枚举：视频格式
/// \~english
/// Enumeration:Video format
typedef enum _SC_EVideoType
{
    typeVideoFormatTIFF = 0,        /// <  \~chinese TIFF格式           \~english support TIFF
    typeVideoFormatBMP = 0,         /// <  \~chinese  BMP格式           \~english support  BMP
    typeVideoFormatRHVD = 0,        /// <  \~chinese RHVD格式           \~english support RHVD
    typeVideoFormatNotSupport = 255 ///< \~chinese 不支持           \~english Not support
} SC_EVideoType;

/// \~chinese
///枚举：图像翻转类型
/// \~english
/// Enumeration:Image flip type
typedef enum _SC_EFlipType
{
    typeFlipVertical,  ///< \~chinese 垂直(Y轴)翻转    \~english Vertical(Y-axis) flip
    typeFlipHorizontal ///< \~chinese 水平(X轴)翻转    \~english Horizontal(X-axis) flip
} SC_EFlipType;

/// \~chinese
///枚举：顺时针旋转角度
/// \~english
/// Enumeration:Rotation angle clockwise
typedef enum _SC_ERotationAngle
{
    rotationAngle90,  ///< \~chinese 顺时针旋转90度   \~english Rotate 90 degree clockwise
    rotationAngle180, ///< \~chinese 顺时针旋转180度  \~english Rotate 180 degree clockwise
    rotationAngle270, ///< \~chinese 顺时针旋转270度  \~english Rotate 270 degree clockwise
} SC_ERotationAngle;


typedef enum _SC_MessageType
{
    eFrameLost,
    eExportStatistic,
    eGenericMessage,

} MessageType;

#define SC_GVSP_PIX_MONO 0x01000000
#define SC_GVSP_PIX_RGB 0x02000000
#define SC_GVSP_PIX_COLOR 0x02000000
#define SC_GVSP_PIX_CUSTOM 0x80000000
#define SC_GVSP_PIX_COLOR_MASK 0xFF000000

// Indicate effective number of bits occupied by the pixel (including padding).
// This can be used to compute amount of memory required to store an image.
#define SC_GVSP_PIX_OCCUPY1BIT 0x00010000
#define SC_GVSP_PIX_OCCUPY2BIT 0x00020000
#define SC_GVSP_PIX_OCCUPY4BIT 0x00040000
#define SC_GVSP_PIX_OCCUPY8BIT 0x00080000
#define SC_GVSP_PIX_OCCUPY12BIT 0x000C0000
#define SC_GVSP_PIX_OCCUPY16BIT 0x00100000
#define SC_GVSP_PIX_OCCUPY24BIT 0x00180000
#define SC_GVSP_PIX_OCCUPY32BIT 0x00200000
#define SC_GVSP_PIX_OCCUPY36BIT 0x00240000
#define SC_GVSP_PIX_OCCUPY48BIT 0x00300000

/// \~chinese
///枚举：图像格式
/// \~english
/// Enumeration:image format
typedef enum _SC_EPixelType
{
    // Undefined pixel type
    gvspPixelTypeUndefined = -1,

    // Mono Format
    gvspPixelMono1p = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY1BIT | 0x0037),
    gvspPixelMono2p = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY2BIT | 0x0038),
    gvspPixelMono4p = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY4BIT | 0x0039),
    gvspPixelMono8 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY8BIT | 0x0001),
    gvspPixelMono8S = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY8BIT | 0x0002),
    gvspPixelMono10 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0003),
    gvspPixelMono10Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x0004),
    gvspPixelMono12 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0005),
    gvspPixelMono12Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x0006),
    gvspPixelMono14 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0025),
    gvspPixelMono16 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0007),

    // Bayer Format
    gvspPixelBayGR8 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY8BIT | 0x0008),
    gvspPixelBayRG8 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY8BIT | 0x0009),
    gvspPixelBayGB8 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY8BIT | 0x000A),
    gvspPixelBayBG8 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY8BIT | 0x000B),
    gvspPixelBayGR10 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x000C),
    gvspPixelBayRG10 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x000D),
    gvspPixelBayGB10 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x000E),
    gvspPixelBayBG10 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x000F),
    gvspPixelBayGR12 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0010),
    gvspPixelBayRG12 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0011),
    gvspPixelBayGB12 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0012),
    gvspPixelBayBG12 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0013),
    gvspPixelBayGR10Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x0026),
    gvspPixelBayRG10Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x0027),
    gvspPixelBayGB10Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x0028),
    gvspPixelBayBG10Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x0029),
    gvspPixelBayGR12Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x002A),
    gvspPixelBayRG12Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x002B),
    gvspPixelBayGB12Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x002C),
    gvspPixelBayBG12Packed = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY12BIT | 0x002D),
    gvspPixelBayGR16 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x002E),
    gvspPixelBayRG16 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x002F),
    gvspPixelBayGB16 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0030),
    gvspPixelBayBG16 = (SC_GVSP_PIX_MONO | SC_GVSP_PIX_OCCUPY16BIT | 0x0031),

    // RGB Format
    gvspPixelRGB8 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY24BIT | 0x0014),
    gvspPixelBGR8 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY24BIT | 0x0015),
    gvspPixelRGBA8 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY32BIT | 0x0016),
    gvspPixelBGRA8 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY32BIT | 0x0017),
    gvspPixelRGB10 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY48BIT | 0x0018),
    gvspPixelBGR10 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY48BIT | 0x0019),
    gvspPixelRGB12 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY48BIT | 0x001A),
    gvspPixelBGR12 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY48BIT | 0x001B),
    gvspPixelRGB16 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY48BIT | 0x0033),
    gvspPixelRGB10V1Packed = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY32BIT | 0x001C),
    gvspPixelRGB10P32 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY32BIT | 0x001D),
    gvspPixelRGB12V1Packed = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY36BIT | 0X0034),
    gvspPixelRGB565P = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x0035),
    gvspPixelBGR565P = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0X0036),

    // YVR Format
    gvspPixelYUV411_8_UYYVYY = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY12BIT | 0x001E),
    gvspPixelYUV422_8_UYVY = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x001F),
    gvspPixelYUV422_8 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x0032),
    gvspPixelYUV8_UYV = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY24BIT | 0x0020),
    gvspPixelYCbCr8CbYCr = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY24BIT | 0x003A),
    gvspPixelYCbCr422_8 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x003B),
    gvspPixelYCbCr422_8_CbYCrY = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x0043),
    gvspPixelYCbCr411_8_CbYYCrYY = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY12BIT | 0x003C),
    gvspPixelYCbCr601_8_CbYCr = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY24BIT | 0x003D),
    gvspPixelYCbCr601_422_8 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x003E),
    gvspPixelYCbCr601_422_8_CbYCrY = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x0044),
    gvspPixelYCbCr601_411_8_CbYYCrYY = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY12BIT | 0x003F),
    gvspPixelYCbCr709_8_CbYCr = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY24BIT | 0x0040),
    gvspPixelYCbCr709_422_8 = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x0041),
    gvspPixelYCbCr709_422_8_CbYCrY = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY16BIT | 0x0045),
    gvspPixelYCbCr709_411_8_CbYYCrYY = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY12BIT | 0x0042),

    // RGB Planar
    gvspPixelRGB8Planar = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY24BIT | 0x0021),
    gvspPixelRGB10Planar = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY48BIT | 0x0022),
    gvspPixelRGB12Planar = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY48BIT | 0x0023),
    gvspPixelRGB16Planar = (SC_GVSP_PIX_COLOR | SC_GVSP_PIX_OCCUPY48BIT | 0x0024),

    //BayerRG10p和BayerRG12p格式，针对特定项目临时添加,请不要使用
    //BayerRG10p and BayerRG12p, currently used for specific project, please do not use them
    gvspPixelBayRG10p = 0x010A0058,
    gvspPixelBayRG12p = 0x010c0059,

    //mono1c格式，自定义格式
    //mono1c, customized image format, used for binary output
    gvspPixelMono1c = 0x012000FF,

    //mono1e格式，自定义格式，用来显示连通域
    //mono1e, customized image format, used for displaying connected domain
    gvspPixelMono1e = 0x01080FFF
} SC_EPixelType;

/// \~chinese
/// \brief 传输模式(gige)
/// \~english
/// \brief Transmission type (gige)
typedef enum _SC_TransmissionType
{
    transTypeUnicast,  ///< \~chinese 单播模式     \~english Unicast Mode
    transTypeMulticast ///< \~chinese 组播模式     \~english Multicast Mode
} SC_TransmissionType;

/// \~chinese
/// \brief 字符串信息
/// \~english
/// \brief String information
typedef struct _SC_String
{
    char str[SC_MAX_STRING_LENTH]; ///< \~chinese  字符串.长度不超过256  \~english Strings and the maximum length of strings is 255.
} SC_String;

/// \~chinese
/// \brief USB接口信息
/// \~english
/// \brief USB interface information
typedef struct _SC_UsbInterfaceInfo
{
    char description[SC_MAX_STRING_LENTH];   ///< \~chinese  USB接口描述信息       \~english USB interface description
    char vendorID[SC_MAX_STRING_LENTH];      ///< \~chinese  USB接口Vendor ID  \~english USB interface Vendor ID
    char deviceID[SC_MAX_STRING_LENTH];      ///< \~chinese  USB接口设备ID     \~english USB interface Device ID
    char subsystemID[SC_MAX_STRING_LENTH];   ///< \~chinese  USB接口Subsystem ID   \~english USB interface Subsystem ID
    char revision[SC_MAX_STRING_LENTH];      ///< \~chinese  USB接口Revision       \~english USB interface Revision
    char speed[SC_MAX_STRING_LENTH];         ///< \~chinese  USB接口speed      \~english USB interface speed
    char chReserved[4][SC_MAX_STRING_LENTH]; ///< 保留                         \~english Reserved field
} SC_UsbInterfaceInfo;

/// \~chinese
/// \brief Usb设备信息
/// \~english
/// \brief Usb device information
typedef struct _SC_UsbDeviceInfo
{
    bool bLowSpeedSupported;   ///< \~chinese true支持，false不支持，其他值 非法。  \~english true support,false not supported,other invalid
    bool bFullSpeedSupported;  ///< \~chinese true支持，false不支持，其他值 非法。  \~english true support,false not supported,other invalid
    bool bHighSpeedSupported;  ///< \~chinese true支持，false不支持，其他值 非法。  \~english true support,false not supported,other invalid
    bool bSuperSpeedSupported; ///< \~chinese true支持，false不支持，其他值 非法。  \~english true support,false not supported,other invalid
    bool bDriverInstalled;     ///< \~chinese true安装，false未安装，其他值 非法。  \~english true support,false not supported,other invalid
    bool boolReserved[3];      ///< \~chinese 保留
    unsigned int Reserved[4];  ///< \~chinese 保留                                  \~english Reserved field

    char configurationValid[SC_MAX_STRING_LENTH]; ///< \~chinese 配置有效性                            \~english Configuration Valid
    char genCPVersion[SC_MAX_STRING_LENTH];       ///< \~chinese GenCP 版本                            \~english GenCP Version
    char u3vVersion[SC_MAX_STRING_LENTH];         ///< \~chinese U3V 版本号                            \~english U3v Version
    char deviceGUID[SC_MAX_STRING_LENTH];         ///< \~chinese 设备引导号                            \~english Device guid number
    char familyName[SC_MAX_STRING_LENTH];         ///< \~chinese 设备系列号                            \~english Device serial number
    char u3vSerialNumber[SC_MAX_STRING_LENTH];    ///< \~chinese 设备序列号                            \~english Device SerialNumber
    char speed[SC_MAX_STRING_LENTH];              ///< \~chinese 设备传输速度                          \~english Device transmission speed
    char maxPower[SC_MAX_STRING_LENTH];           ///< \~chinese 设备最大供电量                        \~english Maximum power supply of device
    char usbProtocol[SC_MAX_STRING_LENTH];        ///< \~chinese 设备USB协议版本                       \~english usb protocol
    char chReserved[3][SC_MAX_STRING_LENTH];      ///< \~chinese 保留                                  \~english Reserved field

} SC_UsbDeviceInfo;

/// \~chinese
/// \brief 设备通用信息
/// \~english
/// \brief Device general information
typedef struct _SC_DeviceInfo
{
    SC_ECameraType nCameraType; ///< \~chinese 设备类别         \~english Camera type
    int nCameraReserved[5];     ///< \~chinese 保留             \~english Reserved field

    char cameraKey[SC_MAX_STRING_LENTH];         ///< \~chinese 厂商:序列号      \~english Camera key
    char cameraName[SC_MAX_STRING_LENTH];        ///< \~chinese 用户自定义名     \~english UserDefinedName
    char serialNumber[SC_MAX_STRING_LENTH];      ///< \~chinese 设备序列号       \~english Device SerialNumber
    char vendorName[SC_MAX_STRING_LENTH];        ///< \~chinese 厂商             \~english Camera Vendor
    char modelName[SC_MAX_STRING_LENTH];         ///< \~chinese 设备型号         \~english Device model
    char manufactureInfo[SC_MAX_STRING_LENTH];   ///< \~chinese 设备制造信息     \~english Device ManufactureInfo
    char deviceVersion[SC_MAX_STRING_LENTH];     ///< \~chinese 设备版本         \~english Device Version
    char cameraReserved[5][SC_MAX_STRING_LENTH]; ///< \~chinese 保留             \~english Reserved field
    char ctiPath[SC_MAX_PATH_LENGTH];
    SC_UsbDeviceInfo usbDeviceInfo;              ///< \~chinese  Usb设备信息     \~english Usb  Device Information

    SC_EInterfaceType nInterfaceType;               ///< \~chinese 接口类别         \~english Interface type
    int nInterfaceReserved[5];                      ///< \~chinese 保留             \~english Reserved field
    char interfaceName[SC_MAX_STRING_LENTH];        ///< \~chinese 接口名           \~english Interface Name
    char interfaceReserved[5][SC_MAX_STRING_LENTH]; ///< \~chinese 保留             \~english Reserved field
    SC_UsbInterfaceInfo usbInterfaceInfo; ///< \~chinese  Usb接口信息     \~english Usb interface Information
} SC_DeviceInfo;

/// \~chinese
/// \brief 网络传输模式
/// \~english
/// \brief Transmission type
typedef struct _SC_TRANSMISSION_TYPE
{
    SC_TransmissionType eTransmissionType; ///< \~chinese 传输模式     \~english Transmission type
    unsigned int nDesIp;                   ///< \~chinese 目标ip地址   \~english Destination IP address
    unsigned short nDesPort;               ///< \~chinese 目标端口号   \~english Destination port

    unsigned int nReserved[32]; ///< \~chinese 预留位       \~english Reserved field
} SC_TRANSMISSION_TYPE;

/// \~chinese
/// \brief 加载失败的属性信息
/// \~english
/// \brief Load failed properties information
typedef struct _SC_ErrorList
{
    unsigned int nParamCnt;                         ///< \~chinese 加载失败的属性个数           \~english The count of load failed properties
    SC_String paramNameList[SC_MAX_ERROR_LIST_NUM]; ///< \~chinese 加载失败的属性集合，上限128  \~english Array of load failed properties, up to 128
} SC_ErrorList;

/// \~chinese
/// \brief 设备信息列表
/// \~english
/// \brief Device information list
typedef struct _SC_DeviceList
{
    unsigned int nDevNum;    ///< \~chinese 设备数量                                 \~english Device Number
    SC_DeviceInfo* pDevInfo; ///< \~chinese 设备息列表(SDK内部缓存)，最多100设备     \~english Device information list(cached within the SDK), up to 100
} SC_DeviceList;

/// \~chinese
/// \brief 连接事件信息
/// \~english
/// \brief connection event information
typedef struct _SC_SConnectArg
{
    SC_EVType event;           ///< \~chinese 事件类型                         \~english event type
    char serialNumber[SC_MAX_STRING_LENTH];  
    unsigned int nReserve[10]; ///< \~chinese 预留字段                         \~english Reserved field
} SC_SConnectArg;

/// \~chinese
/// \brief 参数更新事件信息
/// \~english
/// \brief Updating parameters event information
typedef struct _SC_SParamUpdateArg
{
    bool isPoll;               ///< \~chinese 是否是定时更新,true:表示是定时更新，false:表示非定时更新 \~english Update periodically or not. true:update periodically, true:not update periodically
    unsigned int nReserve[10]; ///< \~chinese 预留字段                         \~english Reserved field
    unsigned int nParamCnt;    ///< \~chinese 更新的参数个数                   \~english The number of parameters which need update
    SC_String* pParamNameList; ///< \~chinese 更新的参数名称集合(SDK内部缓存)  \~english Array of parameter's name which need to be updated(cached within the SDK)
} SC_SParamUpdateArg;

/// \~chinese
/// \brief 流事件信息
/// \~english
/// \brief Stream event information
typedef struct _SC_SStreamArg
{
    unsigned int channel;               ///< \~chinese 流通道号         \~english Channel no.
    uint64_t blockId;                   ///< \~chinese 流数据BlockID    \~english Block ID of stream data
    uint64_t timeStamp;                 ///< \~chinese 时间戳           \~english Event time stamp
    SC_EEventStatus eStreamEventStatus; ///< \~chinese 流事件状态码     \~english Stream event status code
    unsigned int status;                ///< \~chinese 事件状态错误码   \~english Status error code
    unsigned int nReserve[9];           ///< \~chinese 预留字段         \~english Reserved field
} SC_SStreamArg;

/// \~chinese
/// 消息通道事件ID列表
/// \~english
/// message channel event id list
#define SC_MSG_EVENT_ID_EXPOSURE_END 0x9001
#define SC_MSG_EVENT_ID_FRAME_TRIGGER 0x9002
#define SC_MSG_EVENT_ID_FRAME_START 0x9003
#define SC_MSG_EVENT_ID_ACQ_START 0x9004
#define SC_MSG_EVENT_ID_ACQ_TRIGGER 0x9005
#define SC_MSG_EVENT_ID_DATA_READ_OUT 0x9006

/// \~chinese
/// \brief 消息通道事件信息
/// \~english
/// \brief Message channel event information
typedef struct _SC_SMsgChannelArg
{
    unsigned short eventId;    ///< \~chinese 事件Id                               \~english Event id
    unsigned short channelId;  ///< \~chinese 消息通道号                           \~english Channel id
    uint64_t blockId;          ///< \~chinese 流数据BlockID                        \~english Block ID of stream data
    uint64_t timeStamp;        ///< \~chinese 时间戳                               \~english Event timestamp
    unsigned int nReserve[8];  ///< \~chinese 预留字段                             \~english Reserved field
    unsigned int nParamCnt;    ///< \~chinese 参数个数                             \~english The number of parameters which need update
    SC_String* pParamNameList; ///< \~chinese 事件相关的属性名列集合(SDK内部缓存)  \~english Array of parameter's name which is related(cached within the SDK)
} SC_SMsgChannelArg;

/// \~chinese
/// \brief 消息通道事件信息（通用）
/// \~english
/// \brief Message channel event information(Common)
typedef struct _SC_SMsgEventArg
{
    unsigned short eventId;      ///< \~chinese 事件Id                                     \~english Event id
    unsigned short channelId;    ///< \~chinese 消息通道号                                 \~english Channel id
    uint64_t blockId;            ///< \~chinese 流数据BlockID                              \~english Block ID of stream data
    uint64_t timeStamp;          ///< \~chinese 时间戳                                     \~english Event timestamp
    void* pEventData;            ///< \~chinese 事件数据，内部缓存，需要及时进行数据处理   \~english Event data, internal buffer, need to be processed in time
    unsigned int nEventDataSize; ///< \~chinese 事件数据长度                               \~english Event data size
    unsigned int reserve[8];     ///< \~chinese 预留字段                                   \~english Reserved field
} SC_SMsgEventArg;


/// \~chinese
/// \brief 帧图像信息
/// \~english
/// \brief The frame image information
typedef struct _SC_FrameInfo
{
    uint64_t blockId;          ///< \~chinese 帧Id(仅对GigE/Usb/PCIe相机有效)                  \~english The block ID(GigE/Usb/PCIe camera only)
    unsigned int status;       ///< \~chinese 数据帧状态(0是正常状态)                          \~english The status of frame(0 is normal status)
    unsigned int width;        ///< \~chinese 图像宽度                                         \~english The width of image
    unsigned int height;       ///< \~chinese 图像高度                                         \~english The height of image
    unsigned int size;         ///< \~chinese 图像大小                                         \~english The size of image
    SC_EPixelType pixelFormat; ///< \~chinese 图像像素格式                                     \~english The pixel format of image
    uint64_t timeStamp;        ///< \~chinese 图像时间戳                                       \~english The timestamp of image)
    uint64_t exposureTime;     ///< \~chinese 曝光时间                                       \~english The exposure time of image)
    unsigned int paddingX;     ///< \~chinese 图像paddingX                                     \~english The paddingX of image
    unsigned int paddingY;     ///< \~chinese 图像paddingY                                     \~english The paddingY of image

    // debug 
    unsigned int usbSendCnt;
    unsigned int sendTimestampsec;
    unsigned int sendTimestampnas;

    unsigned int nReserved[7]; ///< \~chinese 预留字段                                        \~english Reserved field
    //unsigned int nReserved[19]; ///< \~chinese 预留字段                                        \~english Reserved field
} SC_FrameInfo;

/// \~chinese
/// \brief 帧图像数据信息
/// \~english
/// \brief Frame image data information
typedef struct _SC_Frame
{
    SC_FRAME_HANDLE frameHandle; ///< \~chinese 帧图像句柄(SDK内部帧管理用)                      \~english Frame image handle(used for managing frame within the SDK)
    unsigned char* pData;        ///< \~chinese 帧图像数据的内存首地址                           \~english The starting address of memory of image data
    SC_FrameInfo frameInfo;      ///< \~chinese 帧信息                                           \~english Frame information
    unsigned int nReserved[10];  ///< \~chinese 预留字段                                         \~english Reserved field
} SC_Frame;


typedef struct _SC_StatsInfo
{
    unsigned int imageError;      ///< \~chinese 图像错误的帧数           \~english  Number of images error frames
    unsigned int lostPacketBlock; ///< \~chinese 丢包的帧数               \~english  Number of frames lost
    unsigned int nReserved0[10];  ///< \~chinese 预留                     \~english  Reserved field

    unsigned int imageReceived; ///< \~chinese 正常获取的帧数           \~english  Number of images error frames
    double fps;                 ///< \~chinese 帧率                     \~english  Frame rate
    double bandwidth;           ///< \~chinese 带宽(Mbps)               \~english  Bandwidth(Mbps)
    unsigned int nReserved[8];  ///< \~chinese 预留                     \~english  Reserved field
} SC_StreamStatsInfo;

typedef struct _SC_SimulationStreamStatsInfo
{
    unsigned int imageError; ///< \~chinese 图像错误的帧数           \~english  Number of images error frames
    unsigned int lostFrame;  ///< \~chinese 丢的帧数               \~english  Number of frames lost

    unsigned int imageReceived; ///< \~chinese 正常获取的帧数           \~english  Number of images error frames
    double fps;                 ///< \~chinese 帧率                     \~english  Frame rate

} SC_SimulationStreamStatsInfo;

typedef struct _SC_CXPStreamStatsInfo
{
    unsigned int imageError; ///< \~chinese 图像错误的帧数           \~english  Number of images error frames
    unsigned int lostFrame;  ///< \~chinese 丢的帧数               \~english  Number of frames lost

    unsigned int imageReceived; ///< \~chinese 正常获取的帧数           \~english  Number of images error frames
    double fps;                 ///< \~chinese 帧率                     \~english  Frame rate

} SC_CXPStreamStatsInfo;


typedef struct _SC_ExportStatisticInfo
{
    unsigned int mExportInMemCount;
    unsigned int mSaveCount;
    unsigned int mSaveFps;
    unsigned int mLostCount;
    unsigned int mSumRevFrameCount;
} SC_ExportStatisticInfo;

/// \~chinese
/// \brief 统计流信息
/// \~english
/// \brief Stream statistics information
typedef struct _SC_StreamStatisticsInfo
{
    SC_ECameraType nCameraType; ///< \~chinese 设备类型             \~english  Device type
    SC_StreamStatsInfo streamStatisticInfo;
} SC_StreamStatisticsInfo;

/// \~chinese
/// \brief 枚举属性的枚举值信息
/// \~english
/// \brief Enumeration property 's enumeration value information
typedef struct _SC_EnumEntryInfo
{
    uint64_t value;                 ///< \~chinese 枚举值               \~english  Enumeration value
    char name[SC_MAX_STRING_LENTH]; ///< \~chinese symbol名             \~english  Symbol name
} SC_EnumEntryInfo;

/// \~chinese
/// \brief 枚举属性的可设枚举值列表信息
/// \~english
/// \brief Enumeration property 's settable enumeration value list information
typedef struct _SC_EnumEntryList
{
    unsigned int nEnumEntryBufferSize; ///< \~chinese 存放枚举值内存大小                   \~english The size of saving enumeration value
    SC_EnumEntryInfo* pEnumEntryInfo;  ///< \~chinese 存放可设枚举值列表(调用者分配缓存)   \~english Save the list of settable enumeration value(allocated cache by the caller)
} SC_EnumEntryList;

typedef struct _SC_FrameLostInfo
{
    unsigned int nTotalLost;
    unsigned long long nLastFrameId;
    unsigned long long nCurrentFrameId;
} SC_FrameLostInfo;

/// \~chinese
/// \brief 像素转换结构体
/// \~english
/// \brief Pixel convert structure
typedef struct _SC_PixelConvertParam
{
    unsigned int nWidth;              ///< [IN]   \~chinese 图像宽                        \~english Width
    unsigned int nHeight;             ///< [IN]   \~chinese 图像高                        \~english Height
    SC_EPixelType ePixelFormat;       ///< [IN]   \~chinese 像素格式                      \~english Pixel format
    unsigned char* pSrcData;          ///< [IN]   \~chinese 输入图像数据                  \~english Input image data
    unsigned int nSrcDataLen;         ///< [IN]   \~chinese 输入图像长度                  \~english Input image length
    unsigned int nPaddingX;           ///< [IN]   \~chinese 图像宽填充                    \~english Padding X
    unsigned int nPaddingY;           ///< [IN]   \~chinese 图像高填充                    \~english Padding Y
    SC_EBayerDemosaic eBayerDemosaic; ///< [IN]   \~chinese 转换Bayer格式算法             \~english Alorithm used for Bayer demosaic
    SC_EPixelType eDstPixelFormat;    ///< [IN]   \~chinese 目标像素格式                  \~english Destination pixel format
    unsigned char* pDstBuf;           ///< [OUT]  \~chinese 输出数据缓存(调用者分配缓存)  \~english Output data buffer(allocated cache by the caller)
    unsigned int nDstBufSize;         ///< [IN]   \~chinese 提供的输出缓冲区大小          \~english Provided output buffer size
    unsigned int nDstDataLen;         ///< [OUT]  \~chinese 输出数据长度                  \~english Output data length
    unsigned int nReserved[8];        ///<        \~chinese 预留                          \~english Reserved field
} SC_PixelConvertParam;


/// \~chinese
/// \brief 图像翻转结构体
/// \~english
/// \brief Flip image structure
typedef struct _SC_FlipImageParam
{
    unsigned int nWidth;        ///< [IN]   \~chinese 图像宽                        \~english Width
    unsigned int nHeight;       ///< [IN]   \~chinese 图像高                        \~english Height
    SC_EPixelType ePixelFormat; ///< [IN]   \~chinese 像素格式                      \~english Pixel format
    SC_EFlipType eFlipType;     ///< [IN]   \~chinese 翻转类型                      \~english Flip type
    unsigned char* pSrcData;    ///< [IN]   \~chinese 输入图像数据                  \~english Input image data
    unsigned int nSrcDataLen;   ///< [IN]   \~chinese 输入图像长度                  \~english Input image length
    unsigned char* pDstBuf;     ///< [OUT]  \~chinese 输出数据缓存(调用者分配缓存)  \~english Output data buffer(allocated cache by the caller)
    unsigned int nDstBufSize;   ///< [IN]   \~chinese 提供的输出缓冲区大小          \~english Provided output buffer size
    unsigned int nDstDataLen;   ///< [OUT]  \~chinese 输出数据长度                  \~english Output data length
    unsigned int nReserved[8];  ///<        \~chinese 预留                          \~english Reserved
} SC_FlipImageParam;

/// \~chinese
/// \brief 图像旋转结构体
/// \~english
/// \brief Rotate image structure
typedef struct _SC_RotateImageParam
{
    unsigned int nWidth;              ///< [IN][OUT]  \~chinese 图像宽                        \~english Width
    unsigned int nHeight;             ///< [IN][OUT]  \~chinese 图像高                        \~english Height
    SC_EPixelType ePixelFormat;       ///< [IN]       \~chinese 像素格式                      \~english Pixel format
    SC_ERotationAngle eRotationAngle; ///< [IN]       \~chinese 旋转角度                      \~english Rotation angle
    unsigned char* pSrcData;          ///< [IN]       \~chinese 输入图像数据                  \~english Input image data
    unsigned int nSrcDataLen;         ///< [IN]       \~chinese 输入图像长度                  \~english Input image length
    unsigned char* pDstBuf;           ///< [OUT]      \~chinese 输出数据缓存(调用者分配缓存)  \~english Output data buffer(allocated cache by the caller)
    unsigned int nDstBufSize;         ///< [IN]       \~chinese 提供的输出缓冲区大小          \~english Provided output buffer size
    unsigned int nDstDataLen;         ///< [OUT]      \~chinese 输出数据长度                  \~english Output data length
    unsigned int nReserved[8];        ///<            \~chinese 预留                          \~english Reserved
} SC_RotateImageParam;

/// \~chinese
/// \brief 录像结构体
/// \~english
/// \brief Record structure
typedef struct _SC_RecordParam
{
    unsigned int nStartFrame;    ///< [IN]  \~chinese 起始帧                        \~english start frame
    unsigned int nCount;         ///< [IN] \~chinese 采集帧数                      \~english number of frame
    float fFameRate;             ///< [IN]  \~chinese 帧率(大于0)                   \~english Frame rate(greater than 0)
    unsigned int nQuality;       ///< [IN]  \~chinese 视频质量(1-100)               \~english Video quality(1-100)
    SC_EVideoType recordFormat;  ///< [IN]  \~chinese 视频格式                      \~english Video format
    char recordFilePath[SC_MAX_PATH_LENGTH];     ///< [IN]  \~chinese 保存路径                  \~english Save video path
    unsigned int nReserved[5];   ///<       \~chinese 预留                          \~english Reserved
} SC_RecordParam;

typedef struct _SC_MessageReport
{
    MessageType MsgType;
    unsigned int MsgLen;
    char data[SC_MAX_EVENT_LEN];
} SC_MessageReport;

// Log level
// 日志等级
enum SCLogLevel
{
    eDefault = 0,
    eDebug,
    eInfo,
    eWarn,
    eError,
    eFatal
};

#include <string.h>
struct CXPRegInfo
{
    uint64_t addr = 0;
    size_t len = 0;
    SC_EFeatureType type = SC_EFeatureType::featureUndefined;
    char regName[128];
    CXPRegInfo() {}
    CXPRegInfo(uint64_t address, size_t size, SC_EFeatureType t, char* name)
    {
        addr = address;
        len = size;
        type = t;
        strcpy_s(regName, name);
    }
};

/// \~chinese
/// \brief 设备连接状态事件回调函数声明
/// \param pParamUpdateArg [in] 回调时主动推送的设备连接状态事件信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of device connection status event
/// \param pStreamArg [in] The device connection status event which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void (*SC_ConnectCallBack)(const SC_SConnectArg* pConnectArg, void* pUser);

/// \~chinese
/// \brief 参数更新事件回调函数声明
/// \param pParamUpdateArg [in] 回调时主动推送的参数更新事件信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of parameter update event
/// \param pStreamArg [in] The parameter update event which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void (*SC_ParamUpdateCallBack)(const SC_SParamUpdateArg* pParamUpdateArg, void* pUser);

/// \~chinese
/// \brief 流事件回调函数声明
/// \param pStreamArg [in] 回调时主动推送的流事件信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of stream event
/// \param pStreamArg [in] The stream event which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void (*SC_StreamCallBack)(const SC_SStreamArg* pStreamArg, void* pUser);


/// \~chinese
/// \brief 消息通道事件回调函数声明（通用）
/// \param pMsgChannelArg [in] 回调时主动推送的消息通道事件信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of message channel event(Common)
/// \param pMsgChannelArg [in] The message channel event which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void (*SC_MsgChannelCallBackEx)(const SC_SMsgEventArg* pMsgChannelArg, void* pUser);

/// \~chinese
/// \brief 帧数据信息回调函数声明
/// \param pFrame [in] 回调时主动推送的帧信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of frame data information
/// \param pFrame [in] The frame information which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void (*SC_FrameCallBack)(SC_Frame* pFrame, void* pUser);

/// \~chinese
/// \brief 信息回调函数声明
/// \param pFrame [in] 回调时主动推送信息报告
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of message
/// \param pFrame [in] The message which will be active pushed out during the callback
/// \param pUser [in] User defined data

typedef void (*SC_MessageCallBack)(SC_MessageReport* pMessage, void* pUser);

// for upgrade
enum UpgradeNotify
{
    eUpgradeCompleted,
    eProcessing,
    eErasing,
    eUpgradeFailed,
    eFileCheckOK,

    eFileCheckFailed,
    eEraseFailed,
    eSendPackFailed,
    eFileLengthError,
    eCRCError,
    eReadRegFailed,
    eWriteRegFailed,

    eUploadFTPFailed,
    eSendRequestFailed,
    
    eOpenDeviceFailed,
    eBreakProcess,
};

enum ExportNotify
{
    eExportStart,
    eExportProcessing,
    eExportFinish,
    eExportClose
};

enum UpgradeFileType
{
    eFPGA,
    eARM,
    eBoot,
    eFPGAImageParam,
    eFPGAXml
};

typedef void (*UpgradeProcessCB)(int progress, const char* msgText, int notify, void *user);
typedef void (*ExportEventCB)(int notify, void* data, int len);


typedef struct _SC_DeviceInfoLabView
{
    char cameraKey[SC_MAX_STRING_LENTH];         ///< \~chinese 厂商:序列号      \~english Camera key
    char cameraName[SC_MAX_STRING_LENTH];        ///< \~chinese 用户自定义名     \~english UserDefinedName
    char serialNumber[SC_MAX_STRING_LENTH];      ///< \~chinese 设备序列号       \~english Device SerialNumber
    char vendorName[SC_MAX_STRING_LENTH];        ///< \~chinese 厂商             \~english Camera Vendor
    char deviceVersion[SC_MAX_STRING_LENTH];     ///< \~chinese 设备版本         \~english Device Version
    char ctiPath[SC_MAX_PATH_LENGTH];
} SC_DeviceInfoLabView;

//#pragma pack(push,1)
typedef struct _SC_FrameLabView
{
    unsigned int offsetBit;        
    unsigned int width;        ///< \~chinese 图像宽度                                         \~english The width of image
    unsigned int height;       ///< \~chinese 图像高度                                         \~english The height of image
    unsigned int paddingX;     ///< \~chinese 图像paddingX                                     \~english The paddingX of image
    unsigned int paddingY;     ///< \~chinese 图像paddingY  
    unsigned int size;         ///< \~chinese 图像大小                                         \~english The size of image
    unsigned int pixelFormat;  ///< \~chinese 图像像素格式                                     \~english The pixel format of image
    unsigned int reserve[3];   ///< \~chinese 图像像素格式                                     \~english The pixel format of image
    uint64_t timeStamp;        ///< \~chinese 图像时间戳                                       \~english The timestamp of image)
    uint64_t exposureTime;     ///< \~chinese 曝光时间                                    \~english The exposure time of image)
    uint64_t blockId;          ///< \~chinese 帧Id(仅对GigE/Usb/PCIe相机有效)                  \~english The block ID(GigE/Usb/PCIe camera only)
    uint64_t extent;           
} SC_FrameLabView;

typedef struct _SC_ROI
{
    unsigned int width;        ///< \~chinese 图像宽度                                         \~english The width of image
    unsigned int height;       ///< \~chinese 图像高度                                         \~english The height of image
    unsigned int paddingX;     ///< \~chinese 图像paddingX                                     \~english The paddingX of image
    unsigned int paddingY;     ///< \~chinese 图像paddingY  
} SC_ROI;
//#pragma pack(pop)
#endif // __SC_DEFINES_H__