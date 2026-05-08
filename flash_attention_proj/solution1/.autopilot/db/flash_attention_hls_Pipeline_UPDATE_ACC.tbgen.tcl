set moduleName flash_attention_hls_Pipeline_UPDATE_ACC
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
set C_modelName {flash_attention_hls_Pipeline_UPDATE_ACC}
set C_modelType { void 0 }
set C_modelArgList {
	{ d int 32 regular  }
	{ i_4 int 3 regular  }
	{ acc float 32 regular {array 64 { 0 1 } 1 1 }  }
	{ acc_1 float 32 regular {array 64 { 0 1 } 1 1 }  }
	{ acc_2 float 32 regular {array 64 { 0 1 } 1 1 }  }
	{ acc_3 float 32 regular {array 64 { 0 1 } 1 1 }  }
	{ mul1 float 32 regular  }
	{ weighted float 32 regular {array 8 { 1 3 } 1 1 }  }
	{ l_new float 32 regular  }
	{ weighted_1 float 32 regular {array 8 { 1 3 } 1 1 }  }
	{ weighted_2 float 32 regular {array 8 { 1 3 } 1 1 }  }
	{ weighted_3 float 32 regular {array 8 { 1 3 } 1 1 }  }
}
set hasAXIMCache 0
set AXIMCacheInstList { }
set C_modelArgMapList {[ 
	{ "Name" : "d", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "i_4", "interface" : "wire", "bitwidth" : 3, "direction" : "READONLY"} , 
 	{ "Name" : "acc", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} , 
 	{ "Name" : "acc_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} , 
 	{ "Name" : "acc_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} , 
 	{ "Name" : "acc_3", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} , 
 	{ "Name" : "mul1", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weighted", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "l_new", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weighted_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weighted_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weighted_3", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 90
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ d sc_in sc_lv 32 signal 0 } 
	{ i_4 sc_in sc_lv 3 signal 1 } 
	{ acc_address0 sc_out sc_lv 6 signal 2 } 
	{ acc_ce0 sc_out sc_logic 1 signal 2 } 
	{ acc_we0 sc_out sc_logic 1 signal 2 } 
	{ acc_d0 sc_out sc_lv 32 signal 2 } 
	{ acc_address1 sc_out sc_lv 6 signal 2 } 
	{ acc_ce1 sc_out sc_logic 1 signal 2 } 
	{ acc_q1 sc_in sc_lv 32 signal 2 } 
	{ acc_1_address0 sc_out sc_lv 6 signal 3 } 
	{ acc_1_ce0 sc_out sc_logic 1 signal 3 } 
	{ acc_1_we0 sc_out sc_logic 1 signal 3 } 
	{ acc_1_d0 sc_out sc_lv 32 signal 3 } 
	{ acc_1_address1 sc_out sc_lv 6 signal 3 } 
	{ acc_1_ce1 sc_out sc_logic 1 signal 3 } 
	{ acc_1_q1 sc_in sc_lv 32 signal 3 } 
	{ acc_2_address0 sc_out sc_lv 6 signal 4 } 
	{ acc_2_ce0 sc_out sc_logic 1 signal 4 } 
	{ acc_2_we0 sc_out sc_logic 1 signal 4 } 
	{ acc_2_d0 sc_out sc_lv 32 signal 4 } 
	{ acc_2_address1 sc_out sc_lv 6 signal 4 } 
	{ acc_2_ce1 sc_out sc_logic 1 signal 4 } 
	{ acc_2_q1 sc_in sc_lv 32 signal 4 } 
	{ acc_3_address0 sc_out sc_lv 6 signal 5 } 
	{ acc_3_ce0 sc_out sc_logic 1 signal 5 } 
	{ acc_3_we0 sc_out sc_logic 1 signal 5 } 
	{ acc_3_d0 sc_out sc_lv 32 signal 5 } 
	{ acc_3_address1 sc_out sc_lv 6 signal 5 } 
	{ acc_3_ce1 sc_out sc_logic 1 signal 5 } 
	{ acc_3_q1 sc_in sc_lv 32 signal 5 } 
	{ mul1 sc_in sc_lv 32 signal 6 } 
	{ weighted_address0 sc_out sc_lv 3 signal 7 } 
	{ weighted_ce0 sc_out sc_logic 1 signal 7 } 
	{ weighted_q0 sc_in sc_lv 32 signal 7 } 
	{ l_new sc_in sc_lv 32 signal 8 } 
	{ weighted_1_address0 sc_out sc_lv 3 signal 9 } 
	{ weighted_1_ce0 sc_out sc_logic 1 signal 9 } 
	{ weighted_1_q0 sc_in sc_lv 32 signal 9 } 
	{ weighted_2_address0 sc_out sc_lv 3 signal 10 } 
	{ weighted_2_ce0 sc_out sc_logic 1 signal 10 } 
	{ weighted_2_q0 sc_in sc_lv 32 signal 10 } 
	{ weighted_3_address0 sc_out sc_lv 3 signal 11 } 
	{ weighted_3_ce0 sc_out sc_logic 1 signal 11 } 
	{ weighted_3_q0 sc_in sc_lv 32 signal 11 } 
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
	{ grp_fu_940_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_940_p_din1 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_940_p_dout0 sc_in sc_lv 32 signal -1 } 
	{ grp_fu_940_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "d", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "d", "role": "default" }} , 
 	{ "name": "i_4", "direction": "in", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "i_4", "role": "default" }} , 
 	{ "name": "acc_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "acc", "role": "address0" }} , 
 	{ "name": "acc_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc", "role": "ce0" }} , 
 	{ "name": "acc_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc", "role": "we0" }} , 
 	{ "name": "acc_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "acc", "role": "d0" }} , 
 	{ "name": "acc_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "acc", "role": "address1" }} , 
 	{ "name": "acc_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc", "role": "ce1" }} , 
 	{ "name": "acc_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "acc", "role": "q1" }} , 
 	{ "name": "acc_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "acc_1", "role": "address0" }} , 
 	{ "name": "acc_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_1", "role": "ce0" }} , 
 	{ "name": "acc_1_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_1", "role": "we0" }} , 
 	{ "name": "acc_1_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "acc_1", "role": "d0" }} , 
 	{ "name": "acc_1_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "acc_1", "role": "address1" }} , 
 	{ "name": "acc_1_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_1", "role": "ce1" }} , 
 	{ "name": "acc_1_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "acc_1", "role": "q1" }} , 
 	{ "name": "acc_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "acc_2", "role": "address0" }} , 
 	{ "name": "acc_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_2", "role": "ce0" }} , 
 	{ "name": "acc_2_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_2", "role": "we0" }} , 
 	{ "name": "acc_2_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "acc_2", "role": "d0" }} , 
 	{ "name": "acc_2_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "acc_2", "role": "address1" }} , 
 	{ "name": "acc_2_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_2", "role": "ce1" }} , 
 	{ "name": "acc_2_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "acc_2", "role": "q1" }} , 
 	{ "name": "acc_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "acc_3", "role": "address0" }} , 
 	{ "name": "acc_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_3", "role": "ce0" }} , 
 	{ "name": "acc_3_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_3", "role": "we0" }} , 
 	{ "name": "acc_3_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "acc_3", "role": "d0" }} , 
 	{ "name": "acc_3_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "acc_3", "role": "address1" }} , 
 	{ "name": "acc_3_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "acc_3", "role": "ce1" }} , 
 	{ "name": "acc_3_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "acc_3", "role": "q1" }} , 
 	{ "name": "mul1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mul1", "role": "default" }} , 
 	{ "name": "weighted_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "weighted", "role": "address0" }} , 
 	{ "name": "weighted_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weighted", "role": "ce0" }} , 
 	{ "name": "weighted_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weighted", "role": "q0" }} , 
 	{ "name": "l_new", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "l_new", "role": "default" }} , 
 	{ "name": "weighted_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "weighted_1", "role": "address0" }} , 
 	{ "name": "weighted_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weighted_1", "role": "ce0" }} , 
 	{ "name": "weighted_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weighted_1", "role": "q0" }} , 
 	{ "name": "weighted_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "weighted_2", "role": "address0" }} , 
 	{ "name": "weighted_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weighted_2", "role": "ce0" }} , 
 	{ "name": "weighted_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weighted_2", "role": "q0" }} , 
 	{ "name": "weighted_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "weighted_3", "role": "address0" }} , 
 	{ "name": "weighted_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weighted_3", "role": "ce0" }} , 
 	{ "name": "weighted_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weighted_3", "role": "q0" }} , 
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
 	{ "name": "grp_fu_2318_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_2318_p_ce", "role": "default" }} , 
 	{ "name": "grp_fu_940_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_940_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_940_p_din1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_940_p_din1", "role": "default" }} , 
 	{ "name": "grp_fu_940_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_940_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_940_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_940_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "flash_attention_hls_Pipeline_UPDATE_ACC",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "36", "EstimateLatencyMax" : "36",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "d", "Type" : "None", "Direction" : "I"},
			{"Name" : "i_4", "Type" : "None", "Direction" : "I"},
			{"Name" : "acc", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "acc_1", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "acc_2", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "acc_3", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "mul1", "Type" : "None", "Direction" : "I"},
			{"Name" : "weighted", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "l_new", "Type" : "None", "Direction" : "I"},
			{"Name" : "weighted_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weighted_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weighted_3", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "UPDATE_ACC", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter27", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter27", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_16_no_dsp_1_U180", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_16_no_dsp_1_U181", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fdiv_32ns_32ns_32_16_no_dsp_1_U182", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	flash_attention_hls_Pipeline_UPDATE_ACC {
		d {Type I LastRead 0 FirstWrite -1}
		i_4 {Type I LastRead 0 FirstWrite -1}
		acc {Type IO LastRead 0 FirstWrite 27}
		acc_1 {Type IO LastRead 0 FirstWrite 27}
		acc_2 {Type IO LastRead 0 FirstWrite 27}
		acc_3 {Type IO LastRead 0 FirstWrite 27}
		mul1 {Type I LastRead 0 FirstWrite -1}
		weighted {Type I LastRead 4 FirstWrite -1}
		l_new {Type I LastRead 0 FirstWrite -1}
		weighted_1 {Type I LastRead 4 FirstWrite -1}
		weighted_2 {Type I LastRead 4 FirstWrite -1}
		weighted_3 {Type I LastRead 4 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "36", "Max" : "36"}
	, {"Name" : "Interval", "Min" : "36", "Max" : "36"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	d { ap_none {  { d in_data 0 32 } } }
	i_4 { ap_none {  { i_4 in_data 0 3 } } }
	acc { ap_memory {  { acc_address0 mem_address 1 6 }  { acc_ce0 mem_ce 1 1 }  { acc_we0 mem_we 1 1 }  { acc_d0 mem_din 1 32 }  { acc_address1 MemPortADDR2 1 6 }  { acc_ce1 MemPortCE2 1 1 }  { acc_q1 MemPortDOUT2 0 32 } } }
	acc_1 { ap_memory {  { acc_1_address0 mem_address 1 6 }  { acc_1_ce0 mem_ce 1 1 }  { acc_1_we0 mem_we 1 1 }  { acc_1_d0 mem_din 1 32 }  { acc_1_address1 MemPortADDR2 1 6 }  { acc_1_ce1 MemPortCE2 1 1 }  { acc_1_q1 MemPortDOUT2 0 32 } } }
	acc_2 { ap_memory {  { acc_2_address0 mem_address 1 6 }  { acc_2_ce0 mem_ce 1 1 }  { acc_2_we0 mem_we 1 1 }  { acc_2_d0 mem_din 1 32 }  { acc_2_address1 MemPortADDR2 1 6 }  { acc_2_ce1 MemPortCE2 1 1 }  { acc_2_q1 MemPortDOUT2 0 32 } } }
	acc_3 { ap_memory {  { acc_3_address0 mem_address 1 6 }  { acc_3_ce0 mem_ce 1 1 }  { acc_3_we0 mem_we 1 1 }  { acc_3_d0 mem_din 1 32 }  { acc_3_address1 MemPortADDR2 1 6 }  { acc_3_ce1 MemPortCE2 1 1 }  { acc_3_q1 MemPortDOUT2 0 32 } } }
	mul1 { ap_none {  { mul1 in_data 0 32 } } }
	weighted { ap_memory {  { weighted_address0 mem_address 1 3 }  { weighted_ce0 mem_ce 1 1 }  { weighted_q0 mem_dout 0 32 } } }
	l_new { ap_none {  { l_new in_data 0 32 } } }
	weighted_1 { ap_memory {  { weighted_1_address0 mem_address 1 3 }  { weighted_1_ce0 mem_ce 1 1 }  { weighted_1_q0 mem_dout 0 32 } } }
	weighted_2 { ap_memory {  { weighted_2_address0 mem_address 1 3 }  { weighted_2_ce0 mem_ce 1 1 }  { weighted_2_q0 mem_dout 0 32 } } }
	weighted_3 { ap_memory {  { weighted_3_address0 mem_address 1 3 }  { weighted_3_ce0 mem_ce 1 1 }  { weighted_3_q0 mem_dout 0 32 } } }
}
