// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xflash_attention_hls.h"

extern XFlash_attention_hls_Config XFlash_attention_hls_ConfigTable[];

#ifdef SDT
XFlash_attention_hls_Config *XFlash_attention_hls_LookupConfig(UINTPTR BaseAddress) {
	XFlash_attention_hls_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XFlash_attention_hls_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XFlash_attention_hls_ConfigTable[Index].Ctrl_BaseAddress == BaseAddress) {
			ConfigPtr = &XFlash_attention_hls_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XFlash_attention_hls_Initialize(XFlash_attention_hls *InstancePtr, UINTPTR BaseAddress) {
	XFlash_attention_hls_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XFlash_attention_hls_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XFlash_attention_hls_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XFlash_attention_hls_Config *XFlash_attention_hls_LookupConfig(u16 DeviceId) {
	XFlash_attention_hls_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XFLASH_ATTENTION_HLS_NUM_INSTANCES; Index++) {
		if (XFlash_attention_hls_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XFlash_attention_hls_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XFlash_attention_hls_Initialize(XFlash_attention_hls *InstancePtr, u16 DeviceId) {
	XFlash_attention_hls_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XFlash_attention_hls_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XFlash_attention_hls_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

