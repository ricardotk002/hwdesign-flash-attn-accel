// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xflash_attention_hls.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XFlash_attention_hls_CfgInitialize(XFlash_attention_hls *InstancePtr, XFlash_attention_hls_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Ctrl_BaseAddress = ConfigPtr->Ctrl_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XFlash_attention_hls_Start(XFlash_attention_hls *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_AP_CTRL) & 0x80;
    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XFlash_attention_hls_IsDone(XFlash_attention_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XFlash_attention_hls_IsIdle(XFlash_attention_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XFlash_attention_hls_IsReady(XFlash_attention_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XFlash_attention_hls_EnableAutoRestart(XFlash_attention_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_AP_CTRL, 0x80);
}

void XFlash_attention_hls_DisableAutoRestart(XFlash_attention_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_AP_CTRL, 0);
}

void XFlash_attention_hls_Set_Q(XFlash_attention_hls *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_Q_DATA, (u32)(Data));
    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_Q_DATA + 4, (u32)(Data >> 32));
}

u64 XFlash_attention_hls_Get_Q(XFlash_attention_hls *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_Q_DATA);
    Data += (u64)XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_Q_DATA + 4) << 32;
    return Data;
}

void XFlash_attention_hls_Set_K(XFlash_attention_hls *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_K_DATA, (u32)(Data));
    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_K_DATA + 4, (u32)(Data >> 32));
}

u64 XFlash_attention_hls_Get_K(XFlash_attention_hls *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_K_DATA);
    Data += (u64)XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_K_DATA + 4) << 32;
    return Data;
}

void XFlash_attention_hls_Set_V(XFlash_attention_hls *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_V_DATA, (u32)(Data));
    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_V_DATA + 4, (u32)(Data >> 32));
}

u64 XFlash_attention_hls_Get_V(XFlash_attention_hls *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_V_DATA);
    Data += (u64)XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_V_DATA + 4) << 32;
    return Data;
}

void XFlash_attention_hls_Set_O(XFlash_attention_hls *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_O_DATA, (u32)(Data));
    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_O_DATA + 4, (u32)(Data >> 32));
}

u64 XFlash_attention_hls_Get_O(XFlash_attention_hls *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_O_DATA);
    Data += (u64)XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_O_DATA + 4) << 32;
    return Data;
}

void XFlash_attention_hls_Set_N(XFlash_attention_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_N_DATA, Data);
}

u32 XFlash_attention_hls_Get_N(XFlash_attention_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_N_DATA);
    return Data;
}

void XFlash_attention_hls_Set_d(XFlash_attention_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_D_DATA, Data);
}

u32 XFlash_attention_hls_Get_d(XFlash_attention_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_D_DATA);
    return Data;
}

void XFlash_attention_hls_Set_causal(XFlash_attention_hls *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_CAUSAL_DATA, Data);
}

u32 XFlash_attention_hls_Get_causal(XFlash_attention_hls *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_CAUSAL_DATA);
    return Data;
}

void XFlash_attention_hls_InterruptGlobalEnable(XFlash_attention_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_GIE, 1);
}

void XFlash_attention_hls_InterruptGlobalDisable(XFlash_attention_hls *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_GIE, 0);
}

void XFlash_attention_hls_InterruptEnable(XFlash_attention_hls *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_IER);
    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_IER, Register | Mask);
}

void XFlash_attention_hls_InterruptDisable(XFlash_attention_hls *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_IER);
    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_IER, Register & (~Mask));
}

void XFlash_attention_hls_InterruptClear(XFlash_attention_hls *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XFlash_attention_hls_WriteReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_ISR, Mask);
}

u32 XFlash_attention_hls_InterruptGetEnabled(XFlash_attention_hls *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_IER);
}

u32 XFlash_attention_hls_InterruptGetStatus(XFlash_attention_hls *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XFlash_attention_hls_ReadReg(InstancePtr->Ctrl_BaseAddress, XFLASH_ATTENTION_HLS_CTRL_ADDR_ISR);
}

