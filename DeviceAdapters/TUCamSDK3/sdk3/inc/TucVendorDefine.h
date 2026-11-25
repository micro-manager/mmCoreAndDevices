/**
 * @copyright Copyright (C) Tucsen Photonics Co., Ltd. 2012-2025. All rights reserved.
 * @brief 厂商扩展类型定义（与主 SDK `TucDefine.h` 分离，避免污染通用 ABI）
 *
 * @file TucVendorDefine.h
 */
#ifndef TUC_VENDOR_DEFINE_H
#define TUC_VENDOR_DEFINE_H

#include "TucDefine.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 厂商扩展事件数值起点（与核心 SDK 对外 `TUC_EVENT_TYPE` 拉开间隔）。
 * @remarks `TucDefine.h` 中核心事件当前最大为 `TUC_EVENT_TYPE_FEATURE_CHANGE`（9）。回调入参经 `(TUC_EVENT_TYPE)` 传递，
 *          若厂商事件也从 10 递增，会与核心日后新增类型撞号。后续厂商事件建议使用 `TUC_VENDOR_EVENT_BASE + 偏移`。
 *          若需更大隔离（例如预留大量核心枚举），可改为 1000 等量级并同步文档。
 */
#ifndef TUC_VENDOR_EVENT_BASE
#define TUC_VENDOR_EVENT_BASE (100)
#endif

/**
 * @brief 厂商侧事件类型（注册回调时转换为 TUC_EVENT_TYPE：`(TUC_EVENT_TYPE)值`）
 */
typedef enum _TUC_VENDOR_EVENT_TYPE
{
    /** 升级进度：载荷为 TUC_VENDOR_UPDATE_PROGRESS */
    TUC_VENDOR_EVENT_UPDATE_PROGRESS = TUC_VENDOR_EVENT_BASE
} TUC_VENDOR_EVENT_TYPE;

/**
 * @brief 厂商升级进度阶段（对应 TUC_VENDOR_UPDATE_PROGRESS.phase）
 */
typedef enum _TUC_VENDOR_UPDATE_PHASE
{
    TUC_VENDOR_UPDATE_PHASE_STARTED = 0,
    TUC_VENDOR_UPDATE_PHASE_TRANSFERRING = 1,
    TUC_VENDOR_UPDATE_PHASE_COMPLETED = 2,
    TUC_VENDOR_UPDATE_PHASE_FAILED = 3,
    TUC_VENDOR_UPDATE_PHASE_ABORTED = 4 /**< User/host requested abort (e.g. UpgradeAbort node) */
} TUC_VENDOR_UPDATE_PHASE;

/**
 * @brief `(TUC_EVENT_TYPE)TUC_VENDOR_EVENT_UPDATE_PROGRESS` 事件的载荷
 * @remarks percent 由 SDK 填充：(totalBytes > 0) ? min(100, (bytesWritten * 100) / totalBytes) : 0；
 *          COMPLETED 且 totalBytes > 0 时 bytesWritten == totalBytes、percent == 100。
 */
typedef struct _TUC_VENDOR_UPDATE_PROGRESS
{
    int32_t phase;        /**< TUC_VENDOR_UPDATE_PHASE */
    int32_t percent;      /**< 0..100 */
    int32_t fwType;       /**< TucVendorUpdateFirmware 的 fwType */
    uint32_t reserved32;  /**< 预留：保持布局与对齐（其后 int64 自偏移 16 起）；将来可承载 flags、子阶段等而不破坏 ABI，当前恒为 0 */
    int64_t bytesWritten; /**< 节点 UpgradeBytesWritten 或里程碑 */
    int64_t totalBytes;   /**< FileDataLength 或文件大小 */
    uint8_t reserved[16];
} TUC_VENDOR_UPDATE_PROGRESS;

#ifdef __cplusplus
}
#endif

#endif /* TUC_VENDOR_DEFINE_H */
