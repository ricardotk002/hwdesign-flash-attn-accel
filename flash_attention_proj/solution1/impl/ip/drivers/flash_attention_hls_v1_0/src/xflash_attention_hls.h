// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XFLASH_ATTENTION_HLS_H
#define XFLASH_ATTENTION_HLS_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xflash_attention_hls_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Ctrl_BaseAddress;
} XFlash_attention_hls_Config;
#endif

typedef struct {
    u64 Ctrl_BaseAddress;
    u32 IsReady;
} XFlash_attention_hls;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XFlash_attention_hls_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XFlash_attention_hls_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XFlash_attention_hls_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XFlash_attention_hls_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XFlash_attention_hls_Initialize(XFlash_attention_hls *InstancePtr, UINTPTR BaseAddress);
XFlash_attention_hls_Config* XFlash_attention_hls_LookupConfig(UINTPTR BaseAddress);
#else
int XFlash_attention_hls_Initialize(XFlash_attention_hls *InstancePtr, u16 DeviceId);
XFlash_attention_hls_Config* XFlash_attention_hls_LookupConfig(u16 DeviceId);
#endif
int XFlash_attention_hls_CfgInitialize(XFlash_attention_hls *InstancePtr, XFlash_attention_hls_Config *ConfigPtr);
#else
int XFlash_attention_hls_Initialize(XFlash_attention_hls *InstancePtr, const char* InstanceName);
int XFlash_attention_hls_Release(XFlash_attention_hls *InstancePtr);
#endif

void XFlash_attention_hls_Start(XFlash_attention_hls *InstancePtr);
u32 XFlash_attention_hls_IsDone(XFlash_attention_hls *InstancePtr);
u32 XFlash_attention_hls_IsIdle(XFlash_attention_hls *InstancePtr);
u32 XFlash_attention_hls_IsReady(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_EnableAutoRestart(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_DisableAutoRestart(XFlash_attention_hls *InstancePtr);

void XFlash_attention_hls_Set_Q(XFlash_attention_hls *InstancePtr, u64 Data);
u64 XFlash_attention_hls_Get_Q(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_Set_K(XFlash_attention_hls *InstancePtr, u64 Data);
u64 XFlash_attention_hls_Get_K(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_Set_V(XFlash_attention_hls *InstancePtr, u64 Data);
u64 XFlash_attention_hls_Get_V(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_Set_O(XFlash_attention_hls *InstancePtr, u64 Data);
u64 XFlash_attention_hls_Get_O(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_Set_N(XFlash_attention_hls *InstancePtr, u32 Data);
u32 XFlash_attention_hls_Get_N(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_Set_d(XFlash_attention_hls *InstancePtr, u32 Data);
u32 XFlash_attention_hls_Get_d(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_Set_causal(XFlash_attention_hls *InstancePtr, u32 Data);
u32 XFlash_attention_hls_Get_causal(XFlash_attention_hls *InstancePtr);

void XFlash_attention_hls_InterruptGlobalEnable(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_InterruptGlobalDisable(XFlash_attention_hls *InstancePtr);
void XFlash_attention_hls_InterruptEnable(XFlash_attention_hls *InstancePtr, u32 Mask);
void XFlash_attention_hls_InterruptDisable(XFlash_attention_hls *InstancePtr, u32 Mask);
void XFlash_attention_hls_InterruptClear(XFlash_attention_hls *InstancePtr, u32 Mask);
u32 XFlash_attention_hls_InterruptGetEnabled(XFlash_attention_hls *InstancePtr);
u32 XFlash_attention_hls_InterruptGetStatus(XFlash_attention_hls *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
