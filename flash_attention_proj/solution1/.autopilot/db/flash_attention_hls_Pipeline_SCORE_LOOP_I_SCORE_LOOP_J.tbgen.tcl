set moduleName flash_attention_hls_Pipeline_SCORE_LOOP_I_SCORE_LOOP_J
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {flash_attention_hls_Pipeline_SCORE_LOOP_I_SCORE_LOOP_J}
set C_modelType { void 0 }
set C_modelArgList {
	{ k_lim int 32 regular  }
	{ mul_ln104 int 63 regular  }
	{ Qbuf float 32 regular {array 64 { 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 } 1 1 }  }
	{ Qbuf_1 float 32 regular {array 64 { 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 } 1 1 }  }
	{ Qbuf_2 float 32 regular {array 64 { 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 } 1 1 }  }
	{ Qbuf_3 float 32 regular {array 64 { 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 } 1 1 }  }
	{ q0 int 32 regular  }
	{ Kbuf float 32 regular {array 64 { 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 } 1 1 }  }
	{ Kbuf_1 float 32 regular {array 64 { 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 } 1 1 }  }
	{ Kbuf_2 float 32 regular {array 64 { 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 } 1 1 }  }
	{ Kbuf_3 float 32 regular {array 64 { 1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3 3 } 1 1 }  }
	{ scores float 32 regular {array 64 { 0 3 } 0 1 }  }
	{ cmp715 int 1 regular  }
	{ icmp int 1 regular  }
	{ cmp102_2 int 1 regular  }
	{ icmp49 int 1 regular  }
	{ cmp102_4 int 1 regular  }
	{ cmp102_5 int 1 regular  }
	{ cmp102_6 int 1 regular  }
	{ icmp52 int 1 regular  }
	{ cmp102_8 int 1 regular  }
	{ cmp102_9 int 1 regular  }
	{ cmp102_10 int 1 regular  }
	{ cmp102_11 int 1 regular  }
	{ cmp102_12 int 1 regular  }
	{ cmp102_13 int 1 regular  }
	{ cmp102_14 int 1 regular  }
	{ icmp55 int 1 regular  }
	{ cmp102_16 int 1 regular  }
	{ cmp102_17 int 1 regular  }
	{ cmp102_18 int 1 regular  }
	{ cmp102_19 int 1 regular  }
	{ cmp102_20 int 1 regular  }
	{ cmp102_21 int 1 regular  }
	{ cmp102_22 int 1 regular  }
	{ cmp102_23 int 1 regular  }
	{ cmp102_24 int 1 regular  }
	{ cmp102_25 int 1 regular  }
	{ cmp102_26 int 1 regular  }
	{ cmp102_27 int 1 regular  }
	{ cmp102_28 int 1 regular  }
	{ cmp102_29 int 1 regular  }
	{ cmp102_30 int 1 regular  }
	{ icmp58 int 1 regular  }
	{ inv_sqrt_d float 32 regular  }
	{ k0 int 32 regular  }
	{ tobool int 1 regular  }
}
set hasAXIMCache 0
set AXIMCacheInstList { }
set C_modelArgMapList {[ 
	{ "Name" : "k_lim", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "mul_ln104", "interface" : "wire", "bitwidth" : 63, "direction" : "READONLY"} , 
 	{ "Name" : "Qbuf", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Qbuf_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Qbuf_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Qbuf_3", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "q0", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Kbuf", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Kbuf_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Kbuf_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "Kbuf_3", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "scores", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "cmp715", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "icmp", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_2", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "icmp49", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_4", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_5", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_6", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "icmp52", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_8", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_9", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_10", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_11", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_12", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_13", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_14", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "icmp55", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_16", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_17", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_18", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_19", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_20", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_21", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_22", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_23", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_24", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_25", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_26", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_27", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_28", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_29", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "cmp102_30", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "icmp58", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} , 
 	{ "Name" : "inv_sqrt_d", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "k0", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "tobool", "interface" : "wire", "bitwidth" : 1, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 281
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ k_lim sc_in sc_lv 32 signal 0 } 
	{ mul_ln104 sc_in sc_lv 63 signal 1 } 
	{ Qbuf_address0 sc_out sc_lv 6 signal 2 } 
	{ Qbuf_ce0 sc_out sc_logic 1 signal 2 } 
	{ Qbuf_q0 sc_in sc_lv 32 signal 2 } 
	{ Qbuf_address1 sc_out sc_lv 6 signal 2 } 
	{ Qbuf_ce1 sc_out sc_logic 1 signal 2 } 
	{ Qbuf_q1 sc_in sc_lv 32 signal 2 } 
	{ Qbuf_address2 sc_out sc_lv 6 signal 2 } 
	{ Qbuf_ce2 sc_out sc_logic 1 signal 2 } 
	{ Qbuf_q2 sc_in sc_lv 32 signal 2 } 
	{ Qbuf_address3 sc_out sc_lv 6 signal 2 } 
	{ Qbuf_ce3 sc_out sc_logic 1 signal 2 } 
	{ Qbuf_q3 sc_in sc_lv 32 signal 2 } 
	{ Qbuf_address4 sc_out sc_lv 6 signal 2 } 
	{ Qbuf_ce4 sc_out sc_logic 1 signal 2 } 
	{ Qbuf_q4 sc_in sc_lv 32 signal 2 } 
	{ Qbuf_address5 sc_out sc_lv 6 signal 2 } 
	{ Qbuf_ce5 sc_out sc_logic 1 signal 2 } 
	{ Qbuf_q5 sc_in sc_lv 32 signal 2 } 
	{ Qbuf_address6 sc_out sc_lv 6 signal 2 } 
	{ Qbuf_ce6 sc_out sc_logic 1 signal 2 } 
	{ Qbuf_q6 sc_in sc_lv 32 signal 2 } 
	{ Qbuf_address7 sc_out sc_lv 6 signal 2 } 
	{ Qbuf_ce7 sc_out sc_logic 1 signal 2 } 
	{ Qbuf_q7 sc_in sc_lv 32 signal 2 } 
	{ Qbuf_1_address0 sc_out sc_lv 6 signal 3 } 
	{ Qbuf_1_ce0 sc_out sc_logic 1 signal 3 } 
	{ Qbuf_1_q0 sc_in sc_lv 32 signal 3 } 
	{ Qbuf_1_address1 sc_out sc_lv 6 signal 3 } 
	{ Qbuf_1_ce1 sc_out sc_logic 1 signal 3 } 
	{ Qbuf_1_q1 sc_in sc_lv 32 signal 3 } 
	{ Qbuf_1_address2 sc_out sc_lv 6 signal 3 } 
	{ Qbuf_1_ce2 sc_out sc_logic 1 signal 3 } 
	{ Qbuf_1_q2 sc_in sc_lv 32 signal 3 } 
	{ Qbuf_1_address3 sc_out sc_lv 6 signal 3 } 
	{ Qbuf_1_ce3 sc_out sc_logic 1 signal 3 } 
	{ Qbuf_1_q3 sc_in sc_lv 32 signal 3 } 
	{ Qbuf_1_address4 sc_out sc_lv 6 signal 3 } 
	{ Qbuf_1_ce4 sc_out sc_logic 1 signal 3 } 
	{ Qbuf_1_q4 sc_in sc_lv 32 signal 3 } 
	{ Qbuf_1_address5 sc_out sc_lv 6 signal 3 } 
	{ Qbuf_1_ce5 sc_out sc_logic 1 signal 3 } 
	{ Qbuf_1_q5 sc_in sc_lv 32 signal 3 } 
	{ Qbuf_1_address6 sc_out sc_lv 6 signal 3 } 
	{ Qbuf_1_ce6 sc_out sc_logic 1 signal 3 } 
	{ Qbuf_1_q6 sc_in sc_lv 32 signal 3 } 
	{ Qbuf_1_address7 sc_out sc_lv 6 signal 3 } 
	{ Qbuf_1_ce7 sc_out sc_logic 1 signal 3 } 
	{ Qbuf_1_q7 sc_in sc_lv 32 signal 3 } 
	{ Qbuf_2_address0 sc_out sc_lv 6 signal 4 } 
	{ Qbuf_2_ce0 sc_out sc_logic 1 signal 4 } 
	{ Qbuf_2_q0 sc_in sc_lv 32 signal 4 } 
	{ Qbuf_2_address1 sc_out sc_lv 6 signal 4 } 
	{ Qbuf_2_ce1 sc_out sc_logic 1 signal 4 } 
	{ Qbuf_2_q1 sc_in sc_lv 32 signal 4 } 
	{ Qbuf_2_address2 sc_out sc_lv 6 signal 4 } 
	{ Qbuf_2_ce2 sc_out sc_logic 1 signal 4 } 
	{ Qbuf_2_q2 sc_in sc_lv 32 signal 4 } 
	{ Qbuf_2_address3 sc_out sc_lv 6 signal 4 } 
	{ Qbuf_2_ce3 sc_out sc_logic 1 signal 4 } 
	{ Qbuf_2_q3 sc_in sc_lv 32 signal 4 } 
	{ Qbuf_2_address4 sc_out sc_lv 6 signal 4 } 
	{ Qbuf_2_ce4 sc_out sc_logic 1 signal 4 } 
	{ Qbuf_2_q4 sc_in sc_lv 32 signal 4 } 
	{ Qbuf_2_address5 sc_out sc_lv 6 signal 4 } 
	{ Qbuf_2_ce5 sc_out sc_logic 1 signal 4 } 
	{ Qbuf_2_q5 sc_in sc_lv 32 signal 4 } 
	{ Qbuf_2_address6 sc_out sc_lv 6 signal 4 } 
	{ Qbuf_2_ce6 sc_out sc_logic 1 signal 4 } 
	{ Qbuf_2_q6 sc_in sc_lv 32 signal 4 } 
	{ Qbuf_2_address7 sc_out sc_lv 6 signal 4 } 
	{ Qbuf_2_ce7 sc_out sc_logic 1 signal 4 } 
	{ Qbuf_2_q7 sc_in sc_lv 32 signal 4 } 
	{ Qbuf_3_address0 sc_out sc_lv 6 signal 5 } 
	{ Qbuf_3_ce0 sc_out sc_logic 1 signal 5 } 
	{ Qbuf_3_q0 sc_in sc_lv 32 signal 5 } 
	{ Qbuf_3_address1 sc_out sc_lv 6 signal 5 } 
	{ Qbuf_3_ce1 sc_out sc_logic 1 signal 5 } 
	{ Qbuf_3_q1 sc_in sc_lv 32 signal 5 } 
	{ Qbuf_3_address2 sc_out sc_lv 6 signal 5 } 
	{ Qbuf_3_ce2 sc_out sc_logic 1 signal 5 } 
	{ Qbuf_3_q2 sc_in sc_lv 32 signal 5 } 
	{ Qbuf_3_address3 sc_out sc_lv 6 signal 5 } 
	{ Qbuf_3_ce3 sc_out sc_logic 1 signal 5 } 
	{ Qbuf_3_q3 sc_in sc_lv 32 signal 5 } 
	{ Qbuf_3_address4 sc_out sc_lv 6 signal 5 } 
	{ Qbuf_3_ce4 sc_out sc_logic 1 signal 5 } 
	{ Qbuf_3_q4 sc_in sc_lv 32 signal 5 } 
	{ Qbuf_3_address5 sc_out sc_lv 6 signal 5 } 
	{ Qbuf_3_ce5 sc_out sc_logic 1 signal 5 } 
	{ Qbuf_3_q5 sc_in sc_lv 32 signal 5 } 
	{ Qbuf_3_address6 sc_out sc_lv 6 signal 5 } 
	{ Qbuf_3_ce6 sc_out sc_logic 1 signal 5 } 
	{ Qbuf_3_q6 sc_in sc_lv 32 signal 5 } 
	{ Qbuf_3_address7 sc_out sc_lv 6 signal 5 } 
	{ Qbuf_3_ce7 sc_out sc_logic 1 signal 5 } 
	{ Qbuf_3_q7 sc_in sc_lv 32 signal 5 } 
	{ q0 sc_in sc_lv 32 signal 6 } 
	{ Kbuf_address0 sc_out sc_lv 6 signal 7 } 
	{ Kbuf_ce0 sc_out sc_logic 1 signal 7 } 
	{ Kbuf_q0 sc_in sc_lv 32 signal 7 } 
	{ Kbuf_address1 sc_out sc_lv 6 signal 7 } 
	{ Kbuf_ce1 sc_out sc_logic 1 signal 7 } 
	{ Kbuf_q1 sc_in sc_lv 32 signal 7 } 
	{ Kbuf_address2 sc_out sc_lv 6 signal 7 } 
	{ Kbuf_ce2 sc_out sc_logic 1 signal 7 } 
	{ Kbuf_q2 sc_in sc_lv 32 signal 7 } 
	{ Kbuf_address3 sc_out sc_lv 6 signal 7 } 
	{ Kbuf_ce3 sc_out sc_logic 1 signal 7 } 
	{ Kbuf_q3 sc_in sc_lv 32 signal 7 } 
	{ Kbuf_address4 sc_out sc_lv 6 signal 7 } 
	{ Kbuf_ce4 sc_out sc_logic 1 signal 7 } 
	{ Kbuf_q4 sc_in sc_lv 32 signal 7 } 
	{ Kbuf_address5 sc_out sc_lv 6 signal 7 } 
	{ Kbuf_ce5 sc_out sc_logic 1 signal 7 } 
	{ Kbuf_q5 sc_in sc_lv 32 signal 7 } 
	{ Kbuf_address6 sc_out sc_lv 6 signal 7 } 
	{ Kbuf_ce6 sc_out sc_logic 1 signal 7 } 
	{ Kbuf_q6 sc_in sc_lv 32 signal 7 } 
	{ Kbuf_address7 sc_out sc_lv 6 signal 7 } 
	{ Kbuf_ce7 sc_out sc_logic 1 signal 7 } 
	{ Kbuf_q7 sc_in sc_lv 32 signal 7 } 
	{ Kbuf_1_address0 sc_out sc_lv 6 signal 8 } 
	{ Kbuf_1_ce0 sc_out sc_logic 1 signal 8 } 
	{ Kbuf_1_q0 sc_in sc_lv 32 signal 8 } 
	{ Kbuf_1_address1 sc_out sc_lv 6 signal 8 } 
	{ Kbuf_1_ce1 sc_out sc_logic 1 signal 8 } 
	{ Kbuf_1_q1 sc_in sc_lv 32 signal 8 } 
	{ Kbuf_1_address2 sc_out sc_lv 6 signal 8 } 
	{ Kbuf_1_ce2 sc_out sc_logic 1 signal 8 } 
	{ Kbuf_1_q2 sc_in sc_lv 32 signal 8 } 
	{ Kbuf_1_address3 sc_out sc_lv 6 signal 8 } 
	{ Kbuf_1_ce3 sc_out sc_logic 1 signal 8 } 
	{ Kbuf_1_q3 sc_in sc_lv 32 signal 8 } 
	{ Kbuf_1_address4 sc_out sc_lv 6 signal 8 } 
	{ Kbuf_1_ce4 sc_out sc_logic 1 signal 8 } 
	{ Kbuf_1_q4 sc_in sc_lv 32 signal 8 } 
	{ Kbuf_1_address5 sc_out sc_lv 6 signal 8 } 
	{ Kbuf_1_ce5 sc_out sc_logic 1 signal 8 } 
	{ Kbuf_1_q5 sc_in sc_lv 32 signal 8 } 
	{ Kbuf_1_address6 sc_out sc_lv 6 signal 8 } 
	{ Kbuf_1_ce6 sc_out sc_logic 1 signal 8 } 
	{ Kbuf_1_q6 sc_in sc_lv 32 signal 8 } 
	{ Kbuf_1_address7 sc_out sc_lv 6 signal 8 } 
	{ Kbuf_1_ce7 sc_out sc_logic 1 signal 8 } 
	{ Kbuf_1_q7 sc_in sc_lv 32 signal 8 } 
	{ Kbuf_2_address0 sc_out sc_lv 6 signal 9 } 
	{ Kbuf_2_ce0 sc_out sc_logic 1 signal 9 } 
	{ Kbuf_2_q0 sc_in sc_lv 32 signal 9 } 
	{ Kbuf_2_address1 sc_out sc_lv 6 signal 9 } 
	{ Kbuf_2_ce1 sc_out sc_logic 1 signal 9 } 
	{ Kbuf_2_q1 sc_in sc_lv 32 signal 9 } 
	{ Kbuf_2_address2 sc_out sc_lv 6 signal 9 } 
	{ Kbuf_2_ce2 sc_out sc_logic 1 signal 9 } 
	{ Kbuf_2_q2 sc_in sc_lv 32 signal 9 } 
	{ Kbuf_2_address3 sc_out sc_lv 6 signal 9 } 
	{ Kbuf_2_ce3 sc_out sc_logic 1 signal 9 } 
	{ Kbuf_2_q3 sc_in sc_lv 32 signal 9 } 
	{ Kbuf_2_address4 sc_out sc_lv 6 signal 9 } 
	{ Kbuf_2_ce4 sc_out sc_logic 1 signal 9 } 
	{ Kbuf_2_q4 sc_in sc_lv 32 signal 9 } 
	{ Kbuf_2_address5 sc_out sc_lv 6 signal 9 } 
	{ Kbuf_2_ce5 sc_out sc_logic 1 signal 9 } 
	{ Kbuf_2_q5 sc_in sc_lv 32 signal 9 } 
	{ Kbuf_2_address6 sc_out sc_lv 6 signal 9 } 
	{ Kbuf_2_ce6 sc_out sc_logic 1 signal 9 } 
	{ Kbuf_2_q6 sc_in sc_lv 32 signal 9 } 
	{ Kbuf_2_address7 sc_out sc_lv 6 signal 9 } 
	{ Kbuf_2_ce7 sc_out sc_logic 1 signal 9 } 
	{ Kbuf_2_q7 sc_in sc_lv 32 signal 9 } 
	{ Kbuf_3_address0 sc_out sc_lv 6 signal 10 } 
	{ Kbuf_3_ce0 sc_out sc_logic 1 signal 10 } 
	{ Kbuf_3_q0 sc_in sc_lv 32 signal 10 } 
	{ Kbuf_3_address1 sc_out sc_lv 6 signal 10 } 
	{ Kbuf_3_ce1 sc_out sc_logic 1 signal 10 } 
	{ Kbuf_3_q1 sc_in sc_lv 32 signal 10 } 
	{ Kbuf_3_address2 sc_out sc_lv 6 signal 10 } 
	{ Kbuf_3_ce2 sc_out sc_logic 1 signal 10 } 
	{ Kbuf_3_q2 sc_in sc_lv 32 signal 10 } 
	{ Kbuf_3_address3 sc_out sc_lv 6 signal 10 } 
	{ Kbuf_3_ce3 sc_out sc_logic 1 signal 10 } 
	{ Kbuf_3_q3 sc_in sc_lv 32 signal 10 } 
	{ Kbuf_3_address4 sc_out sc_lv 6 signal 10 } 
	{ Kbuf_3_ce4 sc_out sc_logic 1 signal 10 } 
	{ Kbuf_3_q4 sc_in sc_lv 32 signal 10 } 
	{ Kbuf_3_address5 sc_out sc_lv 6 signal 10 } 
	{ Kbuf_3_ce5 sc_out sc_logic 1 signal 10 } 
	{ Kbuf_3_q5 sc_in sc_lv 32 signal 10 } 
	{ Kbuf_3_address6 sc_out sc_lv 6 signal 10 } 
	{ Kbuf_3_ce6 sc_out sc_logic 1 signal 10 } 
	{ Kbuf_3_q6 sc_in sc_lv 32 signal 10 } 
	{ Kbuf_3_address7 sc_out sc_lv 6 signal 10 } 
	{ Kbuf_3_ce7 sc_out sc_logic 1 signal 10 } 
	{ Kbuf_3_q7 sc_in sc_lv 32 signal 10 } 
	{ scores_address0 sc_out sc_lv 6 signal 11 } 
	{ scores_ce0 sc_out sc_logic 1 signal 11 } 
	{ scores_we0 sc_out sc_logic 1 signal 11 } 
	{ scores_d0 sc_out sc_lv 32 signal 11 } 
	{ cmp715 sc_in sc_lv 1 signal 12 } 
	{ icmp sc_in sc_lv 1 signal 13 } 
	{ cmp102_2 sc_in sc_lv 1 signal 14 } 
	{ icmp49 sc_in sc_lv 1 signal 15 } 
	{ cmp102_4 sc_in sc_lv 1 signal 16 } 
	{ cmp102_5 sc_in sc_lv 1 signal 17 } 
	{ cmp102_6 sc_in sc_lv 1 signal 18 } 
	{ icmp52 sc_in sc_lv 1 signal 19 } 
	{ cmp102_8 sc_in sc_lv 1 signal 20 } 
	{ cmp102_9 sc_in sc_lv 1 signal 21 } 
	{ cmp102_10 sc_in sc_lv 1 signal 22 } 
	{ cmp102_11 sc_in sc_lv 1 signal 23 } 
	{ cmp102_12 sc_in sc_lv 1 signal 24 } 
	{ cmp102_13 sc_in sc_lv 1 signal 25 } 
	{ cmp102_14 sc_in sc_lv 1 signal 26 } 
	{ icmp55 sc_in sc_lv 1 signal 27 } 
	{ cmp102_16 sc_in sc_lv 1 signal 28 } 
	{ cmp102_17 sc_in sc_lv 1 signal 29 } 
	{ cmp102_18 sc_in sc_lv 1 signal 30 } 
	{ cmp102_19 sc_in sc_lv 1 signal 31 } 
	{ cmp102_20 sc_in sc_lv 1 signal 32 } 
	{ cmp102_21 sc_in sc_lv 1 signal 33 } 
	{ cmp102_22 sc_in sc_lv 1 signal 34 } 
	{ cmp102_23 sc_in sc_lv 1 signal 35 } 
	{ cmp102_24 sc_in sc_lv 1 signal 36 } 
	{ cmp102_25 sc_in sc_lv 1 signal 37 } 
	{ cmp102_26 sc_in sc_lv 1 signal 38 } 
	{ cmp102_27 sc_in sc_lv 1 signal 39 } 
	{ cmp102_28 sc_in sc_lv 1 signal 40 } 
	{ cmp102_29 sc_in sc_lv 1 signal 41 } 
	{ cmp102_30 sc_in sc_lv 1 signal 42 } 
	{ icmp58 sc_in sc_lv 1 signal 43 } 
	{ inv_sqrt_d sc_in sc_lv 32 signal 44 } 
	{ k0 sc_in sc_lv 32 signal 45 } 
	{ tobool sc_in sc_lv 1 signal 46 } 
	{ grp_fu_2294_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2294_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2294_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_2294_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_2294_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_2298_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2298_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2298_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_2298_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_2298_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_2302_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2302_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2302_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_2302_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_2302_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_2306_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2306_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2306_p_opcode sc_out sc_lv 2 signal -1 } 
	{ grp_fu_2306_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_2306_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_930_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_930_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_930_p_opcode sc_out sc_lv 1 signal -1 } 
	{ grp_fu_930_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_930_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_936_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_936_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_936_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_936_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_2310_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2310_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2310_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_2310_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_2314_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2314_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2314_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_2314_p_ce sc_out sc_logic 1 signal -1 } 
	{ grp_fu_2318_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2318_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_2318_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_2318_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "k_lim", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "k_lim", "role": "default" }} , 
 	{ "name": "mul_ln104", "direction": "in", "datatype": "sc_lv", "bitwidth":63, "type": "signal", "bundle":{"name": "mul_ln104", "role": "default" }} , 
 	{ "name": "Qbuf_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf", "role": "address0" }} , 
 	{ "name": "Qbuf_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf", "role": "ce0" }} , 
 	{ "name": "Qbuf_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf", "role": "q0" }} , 
 	{ "name": "Qbuf_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf", "role": "address1" }} , 
 	{ "name": "Qbuf_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf", "role": "ce1" }} , 
 	{ "name": "Qbuf_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf", "role": "q1" }} , 
 	{ "name": "Qbuf_address2", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf", "role": "address2" }} , 
 	{ "name": "Qbuf_ce2", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf", "role": "ce2" }} , 
 	{ "name": "Qbuf_q2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf", "role": "q2" }} , 
 	{ "name": "Qbuf_address3", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf", "role": "address3" }} , 
 	{ "name": "Qbuf_ce3", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf", "role": "ce3" }} , 
 	{ "name": "Qbuf_q3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf", "role": "q3" }} , 
 	{ "name": "Qbuf_address4", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf", "role": "address4" }} , 
 	{ "name": "Qbuf_ce4", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf", "role": "ce4" }} , 
 	{ "name": "Qbuf_q4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf", "role": "q4" }} , 
 	{ "name": "Qbuf_address5", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf", "role": "address5" }} , 
 	{ "name": "Qbuf_ce5", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf", "role": "ce5" }} , 
 	{ "name": "Qbuf_q5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf", "role": "q5" }} , 
 	{ "name": "Qbuf_address6", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf", "role": "address6" }} , 
 	{ "name": "Qbuf_ce6", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf", "role": "ce6" }} , 
 	{ "name": "Qbuf_q6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf", "role": "q6" }} , 
 	{ "name": "Qbuf_address7", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf", "role": "address7" }} , 
 	{ "name": "Qbuf_ce7", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf", "role": "ce7" }} , 
 	{ "name": "Qbuf_q7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf", "role": "q7" }} , 
 	{ "name": "Qbuf_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "address0" }} , 
 	{ "name": "Qbuf_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "ce0" }} , 
 	{ "name": "Qbuf_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "q0" }} , 
 	{ "name": "Qbuf_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "address1" }} , 
 	{ "name": "Qbuf_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "ce1" }} , 
 	{ "name": "Qbuf_1_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "q1" }} , 
 	{ "name": "Qbuf_1_address2", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "address2" }} , 
 	{ "name": "Qbuf_1_ce2", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "ce2" }} , 
 	{ "name": "Qbuf_1_q2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "q2" }} , 
 	{ "name": "Qbuf_1_address3", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "address3" }} , 
 	{ "name": "Qbuf_1_ce3", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "ce3" }} , 
 	{ "name": "Qbuf_1_q3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "q3" }} , 
 	{ "name": "Qbuf_1_address4", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "address4" }} , 
 	{ "name": "Qbuf_1_ce4", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "ce4" }} , 
 	{ "name": "Qbuf_1_q4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "q4" }} , 
 	{ "name": "Qbuf_1_address5", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "address5" }} , 
 	{ "name": "Qbuf_1_ce5", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "ce5" }} , 
 	{ "name": "Qbuf_1_q5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "q5" }} , 
 	{ "name": "Qbuf_1_address6", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "address6" }} , 
 	{ "name": "Qbuf_1_ce6", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "ce6" }} , 
 	{ "name": "Qbuf_1_q6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "q6" }} , 
 	{ "name": "Qbuf_1_address7", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "address7" }} , 
 	{ "name": "Qbuf_1_ce7", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "ce7" }} , 
 	{ "name": "Qbuf_1_q7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_1", "role": "q7" }} , 
 	{ "name": "Qbuf_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "address0" }} , 
 	{ "name": "Qbuf_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "ce0" }} , 
 	{ "name": "Qbuf_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "q0" }} , 
 	{ "name": "Qbuf_2_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "address1" }} , 
 	{ "name": "Qbuf_2_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "ce1" }} , 
 	{ "name": "Qbuf_2_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "q1" }} , 
 	{ "name": "Qbuf_2_address2", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "address2" }} , 
 	{ "name": "Qbuf_2_ce2", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "ce2" }} , 
 	{ "name": "Qbuf_2_q2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "q2" }} , 
 	{ "name": "Qbuf_2_address3", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "address3" }} , 
 	{ "name": "Qbuf_2_ce3", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "ce3" }} , 
 	{ "name": "Qbuf_2_q3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "q3" }} , 
 	{ "name": "Qbuf_2_address4", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "address4" }} , 
 	{ "name": "Qbuf_2_ce4", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "ce4" }} , 
 	{ "name": "Qbuf_2_q4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "q4" }} , 
 	{ "name": "Qbuf_2_address5", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "address5" }} , 
 	{ "name": "Qbuf_2_ce5", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "ce5" }} , 
 	{ "name": "Qbuf_2_q5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "q5" }} , 
 	{ "name": "Qbuf_2_address6", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "address6" }} , 
 	{ "name": "Qbuf_2_ce6", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "ce6" }} , 
 	{ "name": "Qbuf_2_q6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "q6" }} , 
 	{ "name": "Qbuf_2_address7", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "address7" }} , 
 	{ "name": "Qbuf_2_ce7", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "ce7" }} , 
 	{ "name": "Qbuf_2_q7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_2", "role": "q7" }} , 
 	{ "name": "Qbuf_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "address0" }} , 
 	{ "name": "Qbuf_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "ce0" }} , 
 	{ "name": "Qbuf_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "q0" }} , 
 	{ "name": "Qbuf_3_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "address1" }} , 
 	{ "name": "Qbuf_3_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "ce1" }} , 
 	{ "name": "Qbuf_3_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "q1" }} , 
 	{ "name": "Qbuf_3_address2", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "address2" }} , 
 	{ "name": "Qbuf_3_ce2", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "ce2" }} , 
 	{ "name": "Qbuf_3_q2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "q2" }} , 
 	{ "name": "Qbuf_3_address3", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "address3" }} , 
 	{ "name": "Qbuf_3_ce3", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "ce3" }} , 
 	{ "name": "Qbuf_3_q3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "q3" }} , 
 	{ "name": "Qbuf_3_address4", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "address4" }} , 
 	{ "name": "Qbuf_3_ce4", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "ce4" }} , 
 	{ "name": "Qbuf_3_q4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "q4" }} , 
 	{ "name": "Qbuf_3_address5", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "address5" }} , 
 	{ "name": "Qbuf_3_ce5", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "ce5" }} , 
 	{ "name": "Qbuf_3_q5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "q5" }} , 
 	{ "name": "Qbuf_3_address6", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "address6" }} , 
 	{ "name": "Qbuf_3_ce6", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "ce6" }} , 
 	{ "name": "Qbuf_3_q6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "q6" }} , 
 	{ "name": "Qbuf_3_address7", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "address7" }} , 
 	{ "name": "Qbuf_3_ce7", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "ce7" }} , 
 	{ "name": "Qbuf_3_q7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Qbuf_3", "role": "q7" }} , 
 	{ "name": "q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "q0", "role": "default" }} , 
 	{ "name": "Kbuf_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf", "role": "address0" }} , 
 	{ "name": "Kbuf_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf", "role": "ce0" }} , 
 	{ "name": "Kbuf_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf", "role": "q0" }} , 
 	{ "name": "Kbuf_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf", "role": "address1" }} , 
 	{ "name": "Kbuf_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf", "role": "ce1" }} , 
 	{ "name": "Kbuf_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf", "role": "q1" }} , 
 	{ "name": "Kbuf_address2", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf", "role": "address2" }} , 
 	{ "name": "Kbuf_ce2", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf", "role": "ce2" }} , 
 	{ "name": "Kbuf_q2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf", "role": "q2" }} , 
 	{ "name": "Kbuf_address3", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf", "role": "address3" }} , 
 	{ "name": "Kbuf_ce3", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf", "role": "ce3" }} , 
 	{ "name": "Kbuf_q3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf", "role": "q3" }} , 
 	{ "name": "Kbuf_address4", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf", "role": "address4" }} , 
 	{ "name": "Kbuf_ce4", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf", "role": "ce4" }} , 
 	{ "name": "Kbuf_q4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf", "role": "q4" }} , 
 	{ "name": "Kbuf_address5", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf", "role": "address5" }} , 
 	{ "name": "Kbuf_ce5", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf", "role": "ce5" }} , 
 	{ "name": "Kbuf_q5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf", "role": "q5" }} , 
 	{ "name": "Kbuf_address6", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf", "role": "address6" }} , 
 	{ "name": "Kbuf_ce6", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf", "role": "ce6" }} , 
 	{ "name": "Kbuf_q6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf", "role": "q6" }} , 
 	{ "name": "Kbuf_address7", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf", "role": "address7" }} , 
 	{ "name": "Kbuf_ce7", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf", "role": "ce7" }} , 
 	{ "name": "Kbuf_q7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf", "role": "q7" }} , 
 	{ "name": "Kbuf_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "address0" }} , 
 	{ "name": "Kbuf_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "ce0" }} , 
 	{ "name": "Kbuf_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "q0" }} , 
 	{ "name": "Kbuf_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "address1" }} , 
 	{ "name": "Kbuf_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "ce1" }} , 
 	{ "name": "Kbuf_1_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "q1" }} , 
 	{ "name": "Kbuf_1_address2", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "address2" }} , 
 	{ "name": "Kbuf_1_ce2", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "ce2" }} , 
 	{ "name": "Kbuf_1_q2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "q2" }} , 
 	{ "name": "Kbuf_1_address3", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "address3" }} , 
 	{ "name": "Kbuf_1_ce3", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "ce3" }} , 
 	{ "name": "Kbuf_1_q3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "q3" }} , 
 	{ "name": "Kbuf_1_address4", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "address4" }} , 
 	{ "name": "Kbuf_1_ce4", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "ce4" }} , 
 	{ "name": "Kbuf_1_q4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "q4" }} , 
 	{ "name": "Kbuf_1_address5", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "address5" }} , 
 	{ "name": "Kbuf_1_ce5", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "ce5" }} , 
 	{ "name": "Kbuf_1_q5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "q5" }} , 
 	{ "name": "Kbuf_1_address6", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "address6" }} , 
 	{ "name": "Kbuf_1_ce6", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "ce6" }} , 
 	{ "name": "Kbuf_1_q6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "q6" }} , 
 	{ "name": "Kbuf_1_address7", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "address7" }} , 
 	{ "name": "Kbuf_1_ce7", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "ce7" }} , 
 	{ "name": "Kbuf_1_q7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_1", "role": "q7" }} , 
 	{ "name": "Kbuf_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "address0" }} , 
 	{ "name": "Kbuf_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "ce0" }} , 
 	{ "name": "Kbuf_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "q0" }} , 
 	{ "name": "Kbuf_2_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "address1" }} , 
 	{ "name": "Kbuf_2_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "ce1" }} , 
 	{ "name": "Kbuf_2_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "q1" }} , 
 	{ "name": "Kbuf_2_address2", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "address2" }} , 
 	{ "name": "Kbuf_2_ce2", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "ce2" }} , 
 	{ "name": "Kbuf_2_q2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "q2" }} , 
 	{ "name": "Kbuf_2_address3", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "address3" }} , 
 	{ "name": "Kbuf_2_ce3", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "ce3" }} , 
 	{ "name": "Kbuf_2_q3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "q3" }} , 
 	{ "name": "Kbuf_2_address4", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "address4" }} , 
 	{ "name": "Kbuf_2_ce4", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "ce4" }} , 
 	{ "name": "Kbuf_2_q4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "q4" }} , 
 	{ "name": "Kbuf_2_address5", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "address5" }} , 
 	{ "name": "Kbuf_2_ce5", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "ce5" }} , 
 	{ "name": "Kbuf_2_q5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "q5" }} , 
 	{ "name": "Kbuf_2_address6", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "address6" }} , 
 	{ "name": "Kbuf_2_ce6", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "ce6" }} , 
 	{ "name": "Kbuf_2_q6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "q6" }} , 
 	{ "name": "Kbuf_2_address7", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "address7" }} , 
 	{ "name": "Kbuf_2_ce7", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "ce7" }} , 
 	{ "name": "Kbuf_2_q7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_2", "role": "q7" }} , 
 	{ "name": "Kbuf_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "address0" }} , 
 	{ "name": "Kbuf_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "ce0" }} , 
 	{ "name": "Kbuf_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "q0" }} , 
 	{ "name": "Kbuf_3_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "address1" }} , 
 	{ "name": "Kbuf_3_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "ce1" }} , 
 	{ "name": "Kbuf_3_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "q1" }} , 
 	{ "name": "Kbuf_3_address2", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "address2" }} , 
 	{ "name": "Kbuf_3_ce2", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "ce2" }} , 
 	{ "name": "Kbuf_3_q2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "q2" }} , 
 	{ "name": "Kbuf_3_address3", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "address3" }} , 
 	{ "name": "Kbuf_3_ce3", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "ce3" }} , 
 	{ "name": "Kbuf_3_q3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "q3" }} , 
 	{ "name": "Kbuf_3_address4", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "address4" }} , 
 	{ "name": "Kbuf_3_ce4", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "ce4" }} , 
 	{ "name": "Kbuf_3_q4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "q4" }} , 
 	{ "name": "Kbuf_3_address5", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "address5" }} , 
 	{ "name": "Kbuf_3_ce5", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "ce5" }} , 
 	{ "name": "Kbuf_3_q5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "q5" }} , 
 	{ "name": "Kbuf_3_address6", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "address6" }} , 
 	{ "name": "Kbuf_3_ce6", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "ce6" }} , 
 	{ "name": "Kbuf_3_q6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "q6" }} , 
 	{ "name": "Kbuf_3_address7", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "address7" }} , 
 	{ "name": "Kbuf_3_ce7", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "ce7" }} , 
 	{ "name": "Kbuf_3_q7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "Kbuf_3", "role": "q7" }} , 
 	{ "name": "scores_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "scores", "role": "address0" }} , 
 	{ "name": "scores_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "scores", "role": "ce0" }} , 
 	{ "name": "scores_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "scores", "role": "we0" }} , 
 	{ "name": "scores_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "scores", "role": "d0" }} , 
 	{ "name": "cmp715", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp715", "role": "default" }} , 
 	{ "name": "icmp", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "icmp", "role": "default" }} , 
 	{ "name": "cmp102_2", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_2", "role": "default" }} , 
 	{ "name": "icmp49", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "icmp49", "role": "default" }} , 
 	{ "name": "cmp102_4", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_4", "role": "default" }} , 
 	{ "name": "cmp102_5", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_5", "role": "default" }} , 
 	{ "name": "cmp102_6", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_6", "role": "default" }} , 
 	{ "name": "icmp52", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "icmp52", "role": "default" }} , 
 	{ "name": "cmp102_8", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_8", "role": "default" }} , 
 	{ "name": "cmp102_9", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_9", "role": "default" }} , 
 	{ "name": "cmp102_10", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_10", "role": "default" }} , 
 	{ "name": "cmp102_11", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_11", "role": "default" }} , 
 	{ "name": "cmp102_12", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_12", "role": "default" }} , 
 	{ "name": "cmp102_13", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_13", "role": "default" }} , 
 	{ "name": "cmp102_14", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_14", "role": "default" }} , 
 	{ "name": "icmp55", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "icmp55", "role": "default" }} , 
 	{ "name": "cmp102_16", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_16", "role": "default" }} , 
 	{ "name": "cmp102_17", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_17", "role": "default" }} , 
 	{ "name": "cmp102_18", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_18", "role": "default" }} , 
 	{ "name": "cmp102_19", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_19", "role": "default" }} , 
 	{ "name": "cmp102_20", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_20", "role": "default" }} , 
 	{ "name": "cmp102_21", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_21", "role": "default" }} , 
 	{ "name": "cmp102_22", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_22", "role": "default" }} , 
 	{ "name": "cmp102_23", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_23", "role": "default" }} , 
 	{ "name": "cmp102_24", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_24", "role": "default" }} , 
 	{ "name": "cmp102_25", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_25", "role": "default" }} , 
 	{ "name": "cmp102_26", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_26", "role": "default" }} , 
 	{ "name": "cmp102_27", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_27", "role": "default" }} , 
 	{ "name": "cmp102_28", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_28", "role": "default" }} , 
 	{ "name": "cmp102_29", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_29", "role": "default" }} , 
 	{ "name": "cmp102_30", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "cmp102_30", "role": "default" }} , 
 	{ "name": "icmp58", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "icmp58", "role": "default" }} , 
 	{ "name": "inv_sqrt_d", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inv_sqrt_d", "role": "default" }} , 
 	{ "name": "k0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "k0", "role": "default" }} , 
 	{ "name": "tobool", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "tobool", "role": "default" }} , 
 	{ "name": "grp_fu_2294_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2294_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_2294_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2294_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_2294_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_2294_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_2294_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2294_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_2294_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2294_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_2298_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2298_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_2298_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2298_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_2298_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_2298_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_2298_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2298_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_2298_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2298_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_2302_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2302_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_2302_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2302_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_2302_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_2302_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_2302_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2302_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_2302_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2302_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_2306_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2306_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_2306_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2306_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_2306_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "grp_fu_2306_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_2306_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2306_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_2306_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2306_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_930_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_930_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_930_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_930_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_930_p_opcode", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_930_p_opcode", "role": "default" }} , 
 	{ "name": "grp_fu_930_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_930_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_930_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_930_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_936_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_936_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_936_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_936_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_936_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_936_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_936_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_936_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_2310_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2310_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_2310_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2310_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_2310_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2310_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_2310_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2310_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_2314_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2314_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_2314_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2314_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_2314_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2314_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_2314_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2314_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_2318_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2318_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_2318_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2318_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_2318_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_2318_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_2318_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2318_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57"],
		"CDFG" : "flash_attention_hls_Pipeline_SCORE_LOOP_I_SCORE_LOOP_J",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "k_lim", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln104", "Type" : "None", "Direction" : "I"},
			{"Name" : "Qbuf", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Qbuf_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Qbuf_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Qbuf_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "q0", "Type" : "None", "Direction" : "I"},
			{"Name" : "Kbuf", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Kbuf_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Kbuf_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "Kbuf_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "scores", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "cmp715", "Type" : "None", "Direction" : "I"},
			{"Name" : "icmp", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "icmp49", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_4", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_6", "Type" : "None", "Direction" : "I"},
			{"Name" : "icmp52", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_8", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_9", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_10", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_11", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_12", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_13", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_14", "Type" : "None", "Direction" : "I"},
			{"Name" : "icmp55", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_16", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_17", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_18", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_19", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_20", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_21", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_22", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_23", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_24", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_25", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_26", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_27", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_28", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_29", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp102_30", "Type" : "None", "Direction" : "I"},
			{"Name" : "icmp58", "Type" : "None", "Direction" : "I"},
			{"Name" : "inv_sqrt_d", "Type" : "None", "Direction" : "I"},
			{"Name" : "k0", "Type" : "None", "Direction" : "I"},
			{"Name" : "tobool", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "SCORE_LOOP_I_SCORE_LOOP_J", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter202", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter202", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U38", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U39", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U40", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U41", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U42", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U43", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U44", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U45", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U46", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U47", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U48", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U49", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U50", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U51", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U52", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U53", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U54", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U55", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U56", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U57", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U58", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U59", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U60", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U61", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U62", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U63", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_5_full_dsp_1_U64", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U69", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U70", "Parent" : "0"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U71", "Parent" : "0"},
	{"ID" : "31", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U72", "Parent" : "0"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U73", "Parent" : "0"},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U74", "Parent" : "0"},
	{"ID" : "34", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U75", "Parent" : "0"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U76", "Parent" : "0"},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U77", "Parent" : "0"},
	{"ID" : "37", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U78", "Parent" : "0"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U79", "Parent" : "0"},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U80", "Parent" : "0"},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U81", "Parent" : "0"},
	{"ID" : "41", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U82", "Parent" : "0"},
	{"ID" : "42", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U83", "Parent" : "0"},
	{"ID" : "43", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U84", "Parent" : "0"},
	{"ID" : "44", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U85", "Parent" : "0"},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U86", "Parent" : "0"},
	{"ID" : "46", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U87", "Parent" : "0"},
	{"ID" : "47", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U88", "Parent" : "0"},
	{"ID" : "48", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U89", "Parent" : "0"},
	{"ID" : "49", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U90", "Parent" : "0"},
	{"ID" : "50", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U91", "Parent" : "0"},
	{"ID" : "51", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U92", "Parent" : "0"},
	{"ID" : "52", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U93", "Parent" : "0"},
	{"ID" : "53", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U94", "Parent" : "0"},
	{"ID" : "54", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U95", "Parent" : "0"},
	{"ID" : "55", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U96", "Parent" : "0"},
	{"ID" : "56", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_4_max_dsp_1_U97", "Parent" : "0"},
	{"ID" : "57", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	flash_attention_hls_Pipeline_SCORE_LOOP_I_SCORE_LOOP_J {
		k_lim {Type I LastRead 0 FirstWrite -1}
		mul_ln104 {Type I LastRead 0 FirstWrite -1}
		Qbuf {Type I LastRead 2 FirstWrite -1}
		Qbuf_1 {Type I LastRead 2 FirstWrite -1}
		Qbuf_2 {Type I LastRead 2 FirstWrite -1}
		Qbuf_3 {Type I LastRead 2 FirstWrite -1}
		q0 {Type I LastRead 0 FirstWrite -1}
		Kbuf {Type I LastRead 2 FirstWrite -1}
		Kbuf_1 {Type I LastRead 2 FirstWrite -1}
		Kbuf_2 {Type I LastRead 2 FirstWrite -1}
		Kbuf_3 {Type I LastRead 2 FirstWrite -1}
		scores {Type O LastRead -1 FirstWrite 202}
		cmp715 {Type I LastRead 0 FirstWrite -1}
		icmp {Type I LastRead 0 FirstWrite -1}
		cmp102_2 {Type I LastRead 0 FirstWrite -1}
		icmp49 {Type I LastRead 0 FirstWrite -1}
		cmp102_4 {Type I LastRead 0 FirstWrite -1}
		cmp102_5 {Type I LastRead 0 FirstWrite -1}
		cmp102_6 {Type I LastRead 0 FirstWrite -1}
		icmp52 {Type I LastRead 0 FirstWrite -1}
		cmp102_8 {Type I LastRead 0 FirstWrite -1}
		cmp102_9 {Type I LastRead 0 FirstWrite -1}
		cmp102_10 {Type I LastRead 0 FirstWrite -1}
		cmp102_11 {Type I LastRead 0 FirstWrite -1}
		cmp102_12 {Type I LastRead 0 FirstWrite -1}
		cmp102_13 {Type I LastRead 0 FirstWrite -1}
		cmp102_14 {Type I LastRead 0 FirstWrite -1}
		icmp55 {Type I LastRead 0 FirstWrite -1}
		cmp102_16 {Type I LastRead 0 FirstWrite -1}
		cmp102_17 {Type I LastRead 0 FirstWrite -1}
		cmp102_18 {Type I LastRead 0 FirstWrite -1}
		cmp102_19 {Type I LastRead 0 FirstWrite -1}
		cmp102_20 {Type I LastRead 0 FirstWrite -1}
		cmp102_21 {Type I LastRead 0 FirstWrite -1}
		cmp102_22 {Type I LastRead 0 FirstWrite -1}
		cmp102_23 {Type I LastRead 0 FirstWrite -1}
		cmp102_24 {Type I LastRead 0 FirstWrite -1}
		cmp102_25 {Type I LastRead 0 FirstWrite -1}
		cmp102_26 {Type I LastRead 0 FirstWrite -1}
		cmp102_27 {Type I LastRead 0 FirstWrite -1}
		cmp102_28 {Type I LastRead 0 FirstWrite -1}
		cmp102_29 {Type I LastRead 0 FirstWrite -1}
		cmp102_30 {Type I LastRead 0 FirstWrite -1}
		icmp58 {Type I LastRead 0 FirstWrite -1}
		inv_sqrt_d {Type I LastRead 0 FirstWrite -1}
		k0 {Type I LastRead 0 FirstWrite -1}
		tobool {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	k_lim { ap_none {  { k_lim in_data 0 32 } } }
	mul_ln104 { ap_none {  { mul_ln104 in_data 0 63 } } }
	Qbuf { ap_memory {  { Qbuf_address0 mem_address 1 6 }  { Qbuf_ce0 mem_ce 1 1 }  { Qbuf_q0 mem_dout 0 32 }  { Qbuf_address1 MemPortADDR2 1 6 }  { Qbuf_ce1 MemPortCE2 1 1 }  { Qbuf_q1 MemPortDOUT2 0 32 }  { Qbuf_address2 MemPortADDR2 1 6 }  { Qbuf_ce2 MemPortCE2 1 1 }  { Qbuf_q2 MemPortDOUT2 0 32 }  { Qbuf_address3 MemPortADDR2 1 6 }  { Qbuf_ce3 MemPortCE2 1 1 }  { Qbuf_q3 MemPortDOUT2 0 32 }  { Qbuf_address4 MemPortADDR2 1 6 }  { Qbuf_ce4 MemPortCE2 1 1 }  { Qbuf_q4 MemPortDOUT2 0 32 }  { Qbuf_address5 MemPortADDR2 1 6 }  { Qbuf_ce5 MemPortCE2 1 1 }  { Qbuf_q5 MemPortDOUT2 0 32 }  { Qbuf_address6 MemPortADDR2 1 6 }  { Qbuf_ce6 MemPortCE2 1 1 }  { Qbuf_q6 MemPortDOUT2 0 32 }  { Qbuf_address7 MemPortADDR2 1 6 }  { Qbuf_ce7 MemPortCE2 1 1 }  { Qbuf_q7 MemPortDOUT2 0 32 } } }
	Qbuf_1 { ap_memory {  { Qbuf_1_address0 mem_address 1 6 }  { Qbuf_1_ce0 mem_ce 1 1 }  { Qbuf_1_q0 mem_dout 0 32 }  { Qbuf_1_address1 MemPortADDR2 1 6 }  { Qbuf_1_ce1 MemPortCE2 1 1 }  { Qbuf_1_q1 MemPortDOUT2 0 32 }  { Qbuf_1_address2 MemPortADDR2 1 6 }  { Qbuf_1_ce2 MemPortCE2 1 1 }  { Qbuf_1_q2 MemPortDOUT2 0 32 }  { Qbuf_1_address3 MemPortADDR2 1 6 }  { Qbuf_1_ce3 MemPortCE2 1 1 }  { Qbuf_1_q3 MemPortDOUT2 0 32 }  { Qbuf_1_address4 MemPortADDR2 1 6 }  { Qbuf_1_ce4 MemPortCE2 1 1 }  { Qbuf_1_q4 MemPortDOUT2 0 32 }  { Qbuf_1_address5 MemPortADDR2 1 6 }  { Qbuf_1_ce5 MemPortCE2 1 1 }  { Qbuf_1_q5 MemPortDOUT2 0 32 }  { Qbuf_1_address6 MemPortADDR2 1 6 }  { Qbuf_1_ce6 MemPortCE2 1 1 }  { Qbuf_1_q6 MemPortDOUT2 0 32 }  { Qbuf_1_address7 MemPortADDR2 1 6 }  { Qbuf_1_ce7 MemPortCE2 1 1 }  { Qbuf_1_q7 MemPortDOUT2 0 32 } } }
	Qbuf_2 { ap_memory {  { Qbuf_2_address0 mem_address 1 6 }  { Qbuf_2_ce0 mem_ce 1 1 }  { Qbuf_2_q0 mem_dout 0 32 }  { Qbuf_2_address1 MemPortADDR2 1 6 }  { Qbuf_2_ce1 MemPortCE2 1 1 }  { Qbuf_2_q1 MemPortDOUT2 0 32 }  { Qbuf_2_address2 MemPortADDR2 1 6 }  { Qbuf_2_ce2 MemPortCE2 1 1 }  { Qbuf_2_q2 MemPortDOUT2 0 32 }  { Qbuf_2_address3 MemPortADDR2 1 6 }  { Qbuf_2_ce3 MemPortCE2 1 1 }  { Qbuf_2_q3 MemPortDOUT2 0 32 }  { Qbuf_2_address4 MemPortADDR2 1 6 }  { Qbuf_2_ce4 MemPortCE2 1 1 }  { Qbuf_2_q4 MemPortDOUT2 0 32 }  { Qbuf_2_address5 MemPortADDR2 1 6 }  { Qbuf_2_ce5 MemPortCE2 1 1 }  { Qbuf_2_q5 MemPortDOUT2 0 32 }  { Qbuf_2_address6 MemPortADDR2 1 6 }  { Qbuf_2_ce6 MemPortCE2 1 1 }  { Qbuf_2_q6 MemPortDOUT2 0 32 }  { Qbuf_2_address7 MemPortADDR2 1 6 }  { Qbuf_2_ce7 MemPortCE2 1 1 }  { Qbuf_2_q7 MemPortDOUT2 0 32 } } }
	Qbuf_3 { ap_memory {  { Qbuf_3_address0 mem_address 1 6 }  { Qbuf_3_ce0 mem_ce 1 1 }  { Qbuf_3_q0 mem_dout 0 32 }  { Qbuf_3_address1 MemPortADDR2 1 6 }  { Qbuf_3_ce1 MemPortCE2 1 1 }  { Qbuf_3_q1 MemPortDOUT2 0 32 }  { Qbuf_3_address2 MemPortADDR2 1 6 }  { Qbuf_3_ce2 MemPortCE2 1 1 }  { Qbuf_3_q2 MemPortDOUT2 0 32 }  { Qbuf_3_address3 MemPortADDR2 1 6 }  { Qbuf_3_ce3 MemPortCE2 1 1 }  { Qbuf_3_q3 MemPortDOUT2 0 32 }  { Qbuf_3_address4 MemPortADDR2 1 6 }  { Qbuf_3_ce4 MemPortCE2 1 1 }  { Qbuf_3_q4 MemPortDOUT2 0 32 }  { Qbuf_3_address5 MemPortADDR2 1 6 }  { Qbuf_3_ce5 MemPortCE2 1 1 }  { Qbuf_3_q5 MemPortDOUT2 0 32 }  { Qbuf_3_address6 MemPortADDR2 1 6 }  { Qbuf_3_ce6 MemPortCE2 1 1 }  { Qbuf_3_q6 MemPortDOUT2 0 32 }  { Qbuf_3_address7 MemPortADDR2 1 6 }  { Qbuf_3_ce7 MemPortCE2 1 1 }  { Qbuf_3_q7 MemPortDOUT2 0 32 } } }
	q0 { ap_none {  { q0 in_data 0 32 } } }
	Kbuf { ap_memory {  { Kbuf_address0 mem_address 1 6 }  { Kbuf_ce0 mem_ce 1 1 }  { Kbuf_q0 mem_dout 0 32 }  { Kbuf_address1 MemPortADDR2 1 6 }  { Kbuf_ce1 MemPortCE2 1 1 }  { Kbuf_q1 MemPortDOUT2 0 32 }  { Kbuf_address2 MemPortADDR2 1 6 }  { Kbuf_ce2 MemPortCE2 1 1 }  { Kbuf_q2 MemPortDOUT2 0 32 }  { Kbuf_address3 MemPortADDR2 1 6 }  { Kbuf_ce3 MemPortCE2 1 1 }  { Kbuf_q3 MemPortDOUT2 0 32 }  { Kbuf_address4 MemPortADDR2 1 6 }  { Kbuf_ce4 MemPortCE2 1 1 }  { Kbuf_q4 MemPortDOUT2 0 32 }  { Kbuf_address5 MemPortADDR2 1 6 }  { Kbuf_ce5 MemPortCE2 1 1 }  { Kbuf_q5 MemPortDOUT2 0 32 }  { Kbuf_address6 MemPortADDR2 1 6 }  { Kbuf_ce6 MemPortCE2 1 1 }  { Kbuf_q6 MemPortDOUT2 0 32 }  { Kbuf_address7 MemPortADDR2 1 6 }  { Kbuf_ce7 MemPortCE2 1 1 }  { Kbuf_q7 MemPortDOUT2 0 32 } } }
	Kbuf_1 { ap_memory {  { Kbuf_1_address0 mem_address 1 6 }  { Kbuf_1_ce0 mem_ce 1 1 }  { Kbuf_1_q0 mem_dout 0 32 }  { Kbuf_1_address1 MemPortADDR2 1 6 }  { Kbuf_1_ce1 MemPortCE2 1 1 }  { Kbuf_1_q1 MemPortDOUT2 0 32 }  { Kbuf_1_address2 MemPortADDR2 1 6 }  { Kbuf_1_ce2 MemPortCE2 1 1 }  { Kbuf_1_q2 MemPortDOUT2 0 32 }  { Kbuf_1_address3 MemPortADDR2 1 6 }  { Kbuf_1_ce3 MemPortCE2 1 1 }  { Kbuf_1_q3 MemPortDOUT2 0 32 }  { Kbuf_1_address4 MemPortADDR2 1 6 }  { Kbuf_1_ce4 MemPortCE2 1 1 }  { Kbuf_1_q4 MemPortDOUT2 0 32 }  { Kbuf_1_address5 MemPortADDR2 1 6 }  { Kbuf_1_ce5 MemPortCE2 1 1 }  { Kbuf_1_q5 MemPortDOUT2 0 32 }  { Kbuf_1_address6 MemPortADDR2 1 6 }  { Kbuf_1_ce6 MemPortCE2 1 1 }  { Kbuf_1_q6 MemPortDOUT2 0 32 }  { Kbuf_1_address7 MemPortADDR2 1 6 }  { Kbuf_1_ce7 MemPortCE2 1 1 }  { Kbuf_1_q7 MemPortDOUT2 0 32 } } }
	Kbuf_2 { ap_memory {  { Kbuf_2_address0 mem_address 1 6 }  { Kbuf_2_ce0 mem_ce 1 1 }  { Kbuf_2_q0 mem_dout 0 32 }  { Kbuf_2_address1 MemPortADDR2 1 6 }  { Kbuf_2_ce1 MemPortCE2 1 1 }  { Kbuf_2_q1 MemPortDOUT2 0 32 }  { Kbuf_2_address2 MemPortADDR2 1 6 }  { Kbuf_2_ce2 MemPortCE2 1 1 }  { Kbuf_2_q2 MemPortDOUT2 0 32 }  { Kbuf_2_address3 MemPortADDR2 1 6 }  { Kbuf_2_ce3 MemPortCE2 1 1 }  { Kbuf_2_q3 MemPortDOUT2 0 32 }  { Kbuf_2_address4 MemPortADDR2 1 6 }  { Kbuf_2_ce4 MemPortCE2 1 1 }  { Kbuf_2_q4 MemPortDOUT2 0 32 }  { Kbuf_2_address5 MemPortADDR2 1 6 }  { Kbuf_2_ce5 MemPortCE2 1 1 }  { Kbuf_2_q5 MemPortDOUT2 0 32 }  { Kbuf_2_address6 MemPortADDR2 1 6 }  { Kbuf_2_ce6 MemPortCE2 1 1 }  { Kbuf_2_q6 MemPortDOUT2 0 32 }  { Kbuf_2_address7 MemPortADDR2 1 6 }  { Kbuf_2_ce7 MemPortCE2 1 1 }  { Kbuf_2_q7 MemPortDOUT2 0 32 } } }
	Kbuf_3 { ap_memory {  { Kbuf_3_address0 mem_address 1 6 }  { Kbuf_3_ce0 mem_ce 1 1 }  { Kbuf_3_q0 mem_dout 0 32 }  { Kbuf_3_address1 MemPortADDR2 1 6 }  { Kbuf_3_ce1 MemPortCE2 1 1 }  { Kbuf_3_q1 MemPortDOUT2 0 32 }  { Kbuf_3_address2 MemPortADDR2 1 6 }  { Kbuf_3_ce2 MemPortCE2 1 1 }  { Kbuf_3_q2 MemPortDOUT2 0 32 }  { Kbuf_3_address3 MemPortADDR2 1 6 }  { Kbuf_3_ce3 MemPortCE2 1 1 }  { Kbuf_3_q3 MemPortDOUT2 0 32 }  { Kbuf_3_address4 MemPortADDR2 1 6 }  { Kbuf_3_ce4 MemPortCE2 1 1 }  { Kbuf_3_q4 MemPortDOUT2 0 32 }  { Kbuf_3_address5 MemPortADDR2 1 6 }  { Kbuf_3_ce5 MemPortCE2 1 1 }  { Kbuf_3_q5 MemPortDOUT2 0 32 }  { Kbuf_3_address6 MemPortADDR2 1 6 }  { Kbuf_3_ce6 MemPortCE2 1 1 }  { Kbuf_3_q6 MemPortDOUT2 0 32 }  { Kbuf_3_address7 MemPortADDR2 1 6 }  { Kbuf_3_ce7 MemPortCE2 1 1 }  { Kbuf_3_q7 MemPortDOUT2 0 32 } } }
	scores { ap_memory {  { scores_address0 mem_address 1 6 }  { scores_ce0 mem_ce 1 1 }  { scores_we0 mem_we 1 1 }  { scores_d0 mem_din 1 32 } } }
	cmp715 { ap_none {  { cmp715 in_data 0 1 } } }
	icmp { ap_none {  { icmp in_data 0 1 } } }
	cmp102_2 { ap_none {  { cmp102_2 in_data 0 1 } } }
	icmp49 { ap_none {  { icmp49 in_data 0 1 } } }
	cmp102_4 { ap_none {  { cmp102_4 in_data 0 1 } } }
	cmp102_5 { ap_none {  { cmp102_5 in_data 0 1 } } }
	cmp102_6 { ap_none {  { cmp102_6 in_data 0 1 } } }
	icmp52 { ap_none {  { icmp52 in_data 0 1 } } }
	cmp102_8 { ap_none {  { cmp102_8 in_data 0 1 } } }
	cmp102_9 { ap_none {  { cmp102_9 in_data 0 1 } } }
	cmp102_10 { ap_none {  { cmp102_10 in_data 0 1 } } }
	cmp102_11 { ap_none {  { cmp102_11 in_data 0 1 } } }
	cmp102_12 { ap_none {  { cmp102_12 in_data 0 1 } } }
	cmp102_13 { ap_none {  { cmp102_13 in_data 0 1 } } }
	cmp102_14 { ap_none {  { cmp102_14 in_data 0 1 } } }
	icmp55 { ap_none {  { icmp55 in_data 0 1 } } }
	cmp102_16 { ap_none {  { cmp102_16 in_data 0 1 } } }
	cmp102_17 { ap_none {  { cmp102_17 in_data 0 1 } } }
	cmp102_18 { ap_none {  { cmp102_18 in_data 0 1 } } }
	cmp102_19 { ap_none {  { cmp102_19 in_data 0 1 } } }
	cmp102_20 { ap_none {  { cmp102_20 in_data 0 1 } } }
	cmp102_21 { ap_none {  { cmp102_21 in_data 0 1 } } }
	cmp102_22 { ap_none {  { cmp102_22 in_data 0 1 } } }
	cmp102_23 { ap_none {  { cmp102_23 in_data 0 1 } } }
	cmp102_24 { ap_none {  { cmp102_24 in_data 0 1 } } }
	cmp102_25 { ap_none {  { cmp102_25 in_data 0 1 } } }
	cmp102_26 { ap_none {  { cmp102_26 in_data 0 1 } } }
	cmp102_27 { ap_none {  { cmp102_27 in_data 0 1 } } }
	cmp102_28 { ap_none {  { cmp102_28 in_data 0 1 } } }
	cmp102_29 { ap_none {  { cmp102_29 in_data 0 1 } } }
	cmp102_30 { ap_none {  { cmp102_30 in_data 0 1 } } }
	icmp58 { ap_none {  { icmp58 in_data 0 1 } } }
	inv_sqrt_d { ap_none {  { inv_sqrt_d in_data 0 32 } } }
	k0 { ap_none {  { k0 in_data 0 32 } } }
	tobool { ap_none {  { tobool in_data 0 1 } } }
}
