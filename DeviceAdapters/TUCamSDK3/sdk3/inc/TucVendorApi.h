/**************************************************************************************************
 *
 *  Copyright (C) Tucsen Photonics Co.,Ltd. 2012-2025. All rights reserved.
 *
 *  @file      TucVendorApi.h
 *
 *  @brief     TUCam V3 厂商扩展接口 - 固件升级等厂商/工具用 API，后续可扩展其它接口
 *
 *  @version   3.0.0.0
 *
 *  @note      V2 兼容层 TUCAM_Vendor_Update 调用 TucVendorUpdateFirmware；命名无下划线。
 *
 **************************************************************************************************/
#ifndef _TUC_VENDOR_API_H_
#define _TUC_VENDOR_API_H_

#include "TucDefine.h"
#include "TucVendorDefine.h"
#include "TucSDKExports.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 固件升级（V3 原生接口）
 *
 * @param hDevice    V3 设备句柄 (TUC_HDEVICE)，由 TUCamAdapter 从 TUC_HDEVICE 转换得到
 * @param pstrFwFile 固件文件路径（已加密格式：.efi/.efb/.zip/.xml 等）
 * @param fwType   固件类型，见 TUFW_TYPE：TUFW_IIC(.efi)、TUFW_FPGA(.efb)、TUFW_DATA(.zip/.xml)
 * @return TUC_STATUS_OK 成功，否则为错误码
 *
 * @note 调用前需已打开设备；升级完成后设备会重启，需等待重新枚举后再使用。
 * @note **进度事件**：包含 `TucVendorDefine.h`，注册时使用 `(TUC_EVENT_TYPE)TUC_VENDOR_EVENT_UPDATE_PROGRESS`，
 *       载荷为 `TUC_VENDOR_UPDATE_PROGRESS`（phase / percent / bytesWritten / totalBytes / fwType；percent 由字节推导）。
 *       传输期亦可读 GenICam 整数节点 `UpgradeBytesWritten`、`FileDataLength`。升级进行中 **禁止开始采集**，
 *       回调内请勿在同设备上调用会长时间占用设备锁的 API。
 */
TUC_API TUC_STATUS TucVendorUpdateFirmware(TUC_HDEVICE hDevice, const char *pstrFwFile, int32_t fwType);

/**
 * @brief 请求中止进行中的固件升级（与 GenICam 命令节点 `UpgradeAbort` 等价，由 `TucDevice::firmwareUpdateAbort` 执行）
 *
 * @param hDevice 已打开且正在执行 `TucVendorUpdateFirmware` 的设备句柄
 * @return TUC_STATUS_OK 已投递请求；TUC_STATUS_INVALID_OPERATION 当前未在升级；其它为参数或节点错误
 * @note 协作式中止：在传输循环可检测后停止下发；`TucVendorUpdateFirmware` 可能返回 TUC_STATUS_ABORTED；进度 phase 见 TUC_VENDOR_UPDATE_PHASE_ABORTED。
 *       符号名与 `TucVendorUpdateFirmware` 前缀一致（历史名 `TucVendorFirmwareUpdateAbort` 已弃用）。
 */
TUC_API TUC_STATUS TucVendorUpdateFirmwareAbort(TUC_HDEVICE hDevice);

/**
 * @brief 按设备地址读取数据（厂商/工具用，不经过 Xml 节点）
 *
 * @param hDevice 设备句柄
 * @param buffer 缓冲区
 * @param address 指令地址
 * @param length 数据长度，字节数
 * @return TUC_STATUS_OK 成功，否则为错误码
 */
TUC_API TUC_STATUS TucVendorAddressRead(TUC_HDEVICE hDevice, void *buffer, int64_t address, int64_t length);

/**
 * @brief 按设备地址写入数据（厂商/工具用，不经过 Xml 节点）
 *
 * @param hDevice 设备句柄
 * @param buffer 缓冲区
 * @param address 指令地址
 * @param length 数据长度，字节数
 * @return TUC_STATUS_OK 成功，否则为错误码
 */
TUC_API TUC_STATUS TucVendorAddressWrite(TUC_HDEVICE hDevice, const void *buffer, int64_t address, int64_t length);

#ifdef __cplusplus
}
#endif

#endif /* _TUC_VENDOR_API_H_ */
