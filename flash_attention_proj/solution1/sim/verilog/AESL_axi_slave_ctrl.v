// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================

`timescale 1 ns / 1 ps

module AESL_axi_slave_ctrl (
    clk,
    reset,
    TRAN_s_axi_ctrl_AWADDR,
    TRAN_s_axi_ctrl_AWVALID,
    TRAN_s_axi_ctrl_AWREADY,
    TRAN_s_axi_ctrl_WVALID,
    TRAN_s_axi_ctrl_WREADY,
    TRAN_s_axi_ctrl_WDATA,
    TRAN_s_axi_ctrl_WSTRB,
    TRAN_s_axi_ctrl_ARADDR,
    TRAN_s_axi_ctrl_ARVALID,
    TRAN_s_axi_ctrl_ARREADY,
    TRAN_s_axi_ctrl_RVALID,
    TRAN_s_axi_ctrl_RREADY,
    TRAN_s_axi_ctrl_RDATA,
    TRAN_s_axi_ctrl_RRESP,
    TRAN_s_axi_ctrl_BVALID,
    TRAN_s_axi_ctrl_BREADY,
    TRAN_s_axi_ctrl_BRESP,
    TRAN_ctrl_write_data_finish,
    TRAN_ctrl_start_in,
    TRAN_ctrl_idle_out,
    TRAN_ctrl_ready_out,
    TRAN_ctrl_ready_in,
    TRAN_ctrl_done_out,
    TRAN_ctrl_write_start_in   ,
    TRAN_ctrl_write_start_finish,
    TRAN_ctrl_interrupt,
    TRAN_ctrl_transaction_done_in
    );

//------------------------Parameter----------------------
`define TV_IN_Q "../tv/cdatafile/c.flash_attention_hls.autotvin_Q.dat"
`define TV_IN_K "../tv/cdatafile/c.flash_attention_hls.autotvin_K.dat"
`define TV_IN_V "../tv/cdatafile/c.flash_attention_hls.autotvin_V.dat"
`define TV_IN_O "../tv/cdatafile/c.flash_attention_hls.autotvin_O.dat"
`define TV_IN_N "../tv/cdatafile/c.flash_attention_hls.autotvin_N.dat"
`define TV_IN_d "../tv/cdatafile/c.flash_attention_hls.autotvin_d.dat"
`define TV_IN_causal "../tv/cdatafile/c.flash_attention_hls.autotvin_causal.dat"
parameter ADDR_WIDTH = 7;
parameter DATA_WIDTH = 32;
parameter Q_DEPTH = 1;
reg [31 : 0] Q_OPERATE_DEPTH = 0;
parameter Q_c_bitwidth = 64;
parameter K_DEPTH = 1;
reg [31 : 0] K_OPERATE_DEPTH = 0;
parameter K_c_bitwidth = 64;
parameter V_DEPTH = 1;
reg [31 : 0] V_OPERATE_DEPTH = 0;
parameter V_c_bitwidth = 64;
parameter O_DEPTH = 1;
reg [31 : 0] O_OPERATE_DEPTH = 0;
parameter O_c_bitwidth = 64;
parameter N_DEPTH = 1;
reg [31 : 0] N_OPERATE_DEPTH = 0;
parameter N_c_bitwidth = 32;
parameter d_DEPTH = 1;
reg [31 : 0] d_OPERATE_DEPTH = 0;
parameter d_c_bitwidth = 32;
parameter causal_DEPTH = 1;
reg [31 : 0] causal_OPERATE_DEPTH = 0;
parameter causal_c_bitwidth = 32;
parameter START_ADDR = 0;
parameter flash_attention_hls_continue_addr = 0;
parameter flash_attention_hls_auto_start_addr = 0;
parameter Q_data_in_addr = 16;
parameter K_data_in_addr = 28;
parameter V_data_in_addr = 40;
parameter O_data_in_addr = 52;
parameter N_data_in_addr = 64;
parameter d_data_in_addr = 72;
parameter causal_data_in_addr = 80;
parameter STATUS_ADDR = 0;

output [ADDR_WIDTH - 1 : 0] TRAN_s_axi_ctrl_AWADDR;
output  TRAN_s_axi_ctrl_AWVALID;
input  TRAN_s_axi_ctrl_AWREADY;
output  TRAN_s_axi_ctrl_WVALID;
input  TRAN_s_axi_ctrl_WREADY;
output [DATA_WIDTH - 1 : 0] TRAN_s_axi_ctrl_WDATA;
output [DATA_WIDTH/8 - 1 : 0] TRAN_s_axi_ctrl_WSTRB;
output [ADDR_WIDTH - 1 : 0] TRAN_s_axi_ctrl_ARADDR;
output  TRAN_s_axi_ctrl_ARVALID;
input  TRAN_s_axi_ctrl_ARREADY;
input  TRAN_s_axi_ctrl_RVALID;
output  TRAN_s_axi_ctrl_RREADY;
input [DATA_WIDTH - 1 : 0] TRAN_s_axi_ctrl_RDATA;
input [2 - 1 : 0] TRAN_s_axi_ctrl_RRESP;
input  TRAN_s_axi_ctrl_BVALID;
output  TRAN_s_axi_ctrl_BREADY;
input [2 - 1 : 0] TRAN_s_axi_ctrl_BRESP;
output TRAN_ctrl_write_data_finish;
input     clk;
input     reset;
input     TRAN_ctrl_start_in;
output    TRAN_ctrl_done_out;
output    TRAN_ctrl_ready_out;
input     TRAN_ctrl_ready_in;
output    TRAN_ctrl_idle_out;
input  TRAN_ctrl_write_start_in   ;
output TRAN_ctrl_write_start_finish;
input     TRAN_ctrl_interrupt;
input     TRAN_ctrl_transaction_done_in;

reg [ADDR_WIDTH - 1 : 0] AWADDR_reg = 0;
reg  AWVALID_reg = 0;
reg  WVALID_reg = 0;
reg [DATA_WIDTH - 1 : 0] WDATA_reg = 0;
reg [DATA_WIDTH/8 - 1 : 0] WSTRB_reg = 0;
reg [ADDR_WIDTH - 1 : 0] ARADDR_reg = 0;
reg  ARVALID_reg = 0;
reg  RREADY_reg = 0;
reg [DATA_WIDTH - 1 : 0] RDATA_reg = 0;
reg  BREADY_reg = 0;
reg [Q_c_bitwidth - 1 : 0] mem_Q [Q_DEPTH - 1 : 0] = '{default : 'h0};
reg [DATA_WIDTH-1 : 0] image_mem_Q [ (Q_c_bitwidth+DATA_WIDTH-1)/DATA_WIDTH * Q_DEPTH -1 : 0] = '{default : 'hz};
reg Q_write_data_finish;
reg [K_c_bitwidth - 1 : 0] mem_K [K_DEPTH - 1 : 0] = '{default : 'h0};
reg [DATA_WIDTH-1 : 0] image_mem_K [ (K_c_bitwidth+DATA_WIDTH-1)/DATA_WIDTH * K_DEPTH -1 : 0] = '{default : 'hz};
reg K_write_data_finish;
reg [V_c_bitwidth - 1 : 0] mem_V [V_DEPTH - 1 : 0] = '{default : 'h0};
reg [DATA_WIDTH-1 : 0] image_mem_V [ (V_c_bitwidth+DATA_WIDTH-1)/DATA_WIDTH * V_DEPTH -1 : 0] = '{default : 'hz};
reg V_write_data_finish;
reg [O_c_bitwidth - 1 : 0] mem_O [O_DEPTH - 1 : 0] = '{default : 'h0};
reg [DATA_WIDTH-1 : 0] image_mem_O [ (O_c_bitwidth+DATA_WIDTH-1)/DATA_WIDTH * O_DEPTH -1 : 0] = '{default : 'hz};
reg O_write_data_finish;
reg [DATA_WIDTH - 1 : 0] mem_N [N_DEPTH - 1 : 0] = '{default : 'h0};
reg [DATA_WIDTH-1 : 0] image_mem_N [ (N_c_bitwidth+DATA_WIDTH-1)/DATA_WIDTH * N_DEPTH -1 : 0] = '{default : 'hz};
reg N_write_data_finish;
reg [DATA_WIDTH - 1 : 0] mem_d [d_DEPTH - 1 : 0] = '{default : 'h0};
reg [DATA_WIDTH-1 : 0] image_mem_d [ (d_c_bitwidth+DATA_WIDTH-1)/DATA_WIDTH * d_DEPTH -1 : 0] = '{default : 'hz};
reg d_write_data_finish;
reg [DATA_WIDTH - 1 : 0] mem_causal [causal_DEPTH - 1 : 0] = '{default : 'h0};
reg [DATA_WIDTH-1 : 0] image_mem_causal [ (causal_c_bitwidth+DATA_WIDTH-1)/DATA_WIDTH * causal_DEPTH -1 : 0] = '{default : 'hz};
reg causal_write_data_finish;
reg AESL_ready_out_index_reg = 0;
reg AESL_write_start_finish = 0;
reg AESL_ready_reg;
reg ready_initial;
reg AESL_done_index_reg = 0;
reg AESL_idle_index_reg = 0;
reg AESL_auto_restart_index_reg;
reg process_0_finish = 0;
reg process_1_finish = 0;
reg process_2_finish = 0;
reg process_3_finish = 0;
reg process_4_finish = 0;
reg process_5_finish = 0;
reg process_6_finish = 0;
reg process_7_finish = 0;
reg process_8_finish = 0;
//write Q reg
reg [31 : 0] write_Q_count = 0;
reg [31 : 0] Q_diff_count = 0;
reg write_Q_run_flag = 0;
reg write_one_Q_data_done = 0;
//write K reg
reg [31 : 0] write_K_count = 0;
reg [31 : 0] K_diff_count = 0;
reg write_K_run_flag = 0;
reg write_one_K_data_done = 0;
//write V reg
reg [31 : 0] write_V_count = 0;
reg [31 : 0] V_diff_count = 0;
reg write_V_run_flag = 0;
reg write_one_V_data_done = 0;
//write O reg
reg [31 : 0] write_O_count = 0;
reg [31 : 0] O_diff_count = 0;
reg write_O_run_flag = 0;
reg write_one_O_data_done = 0;
//write N reg
reg [31 : 0] write_N_count = 0;
reg [31 : 0] N_diff_count = 0;
reg write_N_run_flag = 0;
reg write_one_N_data_done = 0;
//write d reg
reg [31 : 0] write_d_count = 0;
reg [31 : 0] d_diff_count = 0;
reg write_d_run_flag = 0;
reg write_one_d_data_done = 0;
//write causal reg
reg [31 : 0] write_causal_count = 0;
reg [31 : 0] causal_diff_count = 0;
reg write_causal_run_flag = 0;
reg write_one_causal_data_done = 0;
reg [31 : 0] write_start_count = 0;
reg write_start_run_flag = 0;

//===================process control=================
reg [31 : 0] ongoing_process_number = 0;
//process number depends on how much processes needed.
reg process_busy = 0;

//=================== signal connection ==============
assign TRAN_s_axi_ctrl_AWADDR = AWADDR_reg;
assign TRAN_s_axi_ctrl_AWVALID = AWVALID_reg;
assign TRAN_s_axi_ctrl_WVALID = WVALID_reg;
assign TRAN_s_axi_ctrl_WDATA = WDATA_reg;
assign TRAN_s_axi_ctrl_WSTRB = WSTRB_reg;
assign TRAN_s_axi_ctrl_ARADDR = ARADDR_reg;
assign TRAN_s_axi_ctrl_ARVALID = ARVALID_reg;
assign TRAN_s_axi_ctrl_RREADY = RREADY_reg;
assign TRAN_s_axi_ctrl_BREADY = BREADY_reg;
assign TRAN_ctrl_write_start_finish = AESL_write_start_finish;
assign TRAN_ctrl_done_out = AESL_done_index_reg;
assign TRAN_ctrl_ready_out = AESL_ready_out_index_reg;
assign TRAN_ctrl_idle_out = AESL_idle_index_reg;
assign TRAN_ctrl_write_data_finish = 1 & Q_write_data_finish & K_write_data_finish & V_write_data_finish & O_write_data_finish & N_write_data_finish & d_write_data_finish & causal_write_data_finish;
always @(TRAN_ctrl_ready_in or ready_initial) 
begin
    AESL_ready_reg <= TRAN_ctrl_ready_in | ready_initial;
end

always @(reset or process_0_finish or process_1_finish or process_2_finish or process_3_finish or process_4_finish or process_5_finish or process_6_finish or process_7_finish or process_8_finish ) begin
    if (reset == 0) begin
        ongoing_process_number <= 0;
    end
    else if (ongoing_process_number == 0 && process_0_finish == 1) begin
            ongoing_process_number <= ongoing_process_number + 1;
    end
    else if (ongoing_process_number == 1 && process_1_finish == 1) begin
            ongoing_process_number <= ongoing_process_number + 1;
    end
    else if (ongoing_process_number == 2 && process_2_finish == 1) begin
            ongoing_process_number <= ongoing_process_number + 1;
    end
    else if (ongoing_process_number == 3 && process_3_finish == 1) begin
            ongoing_process_number <= ongoing_process_number + 1;
    end
    else if (ongoing_process_number == 4 && process_4_finish == 1) begin
            ongoing_process_number <= ongoing_process_number + 1;
    end
    else if (ongoing_process_number == 5 && process_5_finish == 1) begin
            ongoing_process_number <= ongoing_process_number + 1;
    end
    else if (ongoing_process_number == 6 && process_6_finish == 1) begin
            ongoing_process_number <= ongoing_process_number + 1;
    end
    else if (ongoing_process_number == 7 && process_7_finish == 1) begin
            ongoing_process_number <= ongoing_process_number + 1;
    end
    else if (ongoing_process_number == 8 && process_8_finish == 1) begin
            ongoing_process_number <= 0;
    end
end

task count_c_data_four_byte_num_by_bitwidth;
input  integer bitwidth;
output integer num;
integer factor;
integer i;
begin
    factor = 32;
    for (i = 1; i <= 1024; i = i + 1) begin
        if (bitwidth <= factor && bitwidth > factor - 32) begin
            num = i;
        end
        factor = factor + 32;
    end
end    
endtask

function integer ceil_align_to_pow_of_two;
input integer a;
begin
    ceil_align_to_pow_of_two = $pow(2,$clog2(a));
end
endfunction

task count_seperate_factor_by_bitwidth;
input  integer bitwidth;
output integer factor;
begin
    if (bitwidth <= 8) begin
        factor=4;
    end
    if (bitwidth <= 16 & bitwidth > 8 ) begin
        factor=2;
    end
    if (bitwidth <= 32 & bitwidth > 16 ) begin
        factor=1;
    end
    if (bitwidth > 32 ) begin
        factor=1;
    end
end    
endtask

task count_operate_depth_by_bitwidth_and_depth;
input  integer bitwidth;
input  integer depth;
output integer operate_depth;
integer factor;
integer remain;
begin
    count_seperate_factor_by_bitwidth (bitwidth , factor);
    operate_depth = depth / factor;
    remain = depth % factor;
    if (remain > 0) begin
        operate_depth = operate_depth + 1;
    end
end    
endtask

task write; /*{{{*/
    input  reg [ADDR_WIDTH - 1:0] waddr;   // write address
    input  reg [DATA_WIDTH - 1:0] wdata;   // write data
    output reg wresp;
    reg aw_flag;
    reg w_flag;
    reg [DATA_WIDTH/8 - 1:0] wstrb_reg;
    integer i;
begin 
    wresp = 0;
    aw_flag = 0;
    w_flag = 0;
//=======================one single write operate======================
    AWADDR_reg <= waddr;
    AWVALID_reg <= 1;
    WDATA_reg <= wdata;
    WVALID_reg <= 1;
    for (i = 0; i < DATA_WIDTH/8; i = i + 1) begin
        wstrb_reg [i] = 1;
    end    
    WSTRB_reg <= wstrb_reg;
    while (!(aw_flag && w_flag)) begin
        @(posedge clk);
        if (aw_flag != 1)
            aw_flag = TRAN_s_axi_ctrl_AWREADY & AWVALID_reg;
        if (w_flag != 1)
            w_flag = TRAN_s_axi_ctrl_WREADY & WVALID_reg;
        AWVALID_reg <= !aw_flag;
        WVALID_reg <= !w_flag;
    end

    BREADY_reg <= 1;
    while (TRAN_s_axi_ctrl_BVALID != 1) begin
        //wait for response 
        @(posedge clk);
    end
    @(posedge clk);
    BREADY_reg <= 0;
    if (TRAN_s_axi_ctrl_BRESP === 2'b00) begin
        wresp = 1;
        //input success. in fact BRESP is always 2'b00
    end   
//=======================one single write operate======================

end
endtask/*}}}*/

task read (/*{{{*/
    input  [ADDR_WIDTH - 1:0] raddr ,   // write address
    output [DATA_WIDTH - 1:0] RDATA_result ,
    output rresp
);
begin 
    rresp = 0;
//=======================one single read operate======================
    ARADDR_reg <= raddr;
    ARVALID_reg <= 1;
    while (TRAN_s_axi_ctrl_ARREADY !== 1) begin
        @(posedge clk);
    end
    @(posedge clk);
    ARVALID_reg <= 0;
    RREADY_reg <= 1;
    while (TRAN_s_axi_ctrl_RVALID !== 1) begin
        //wait for response 
        @(posedge clk);
    end
    @(posedge clk);
    RDATA_result  <= TRAN_s_axi_ctrl_RDATA;
    RREADY_reg <= 0;
    if (TRAN_s_axi_ctrl_RRESP === 2'b00 ) begin
        rresp <= 1;
        //output success. in fact RRESP is always 2'b00
    end  
    @(posedge clk);

//=======================one single read operate end======================

end
endtask/*}}}*/

initial begin : ready_initial_process
    ready_initial = 0;
    wait(reset === 1);
    @(posedge clk);
    ready_initial = 1;
    @(posedge clk);
    ready_initial = 0;
end

initial begin : update_status
    integer process_num ;
    integer read_status_resp;
    wait(reset === 1);
    @(posedge clk);
    process_num = 0;
    while (1) begin
        process_0_finish = 0;
        AESL_done_index_reg         <= 0;
        AESL_ready_out_index_reg        <= 0;
        if (ongoing_process_number === process_num && process_busy === 0) begin
            process_busy = 1;
            read (STATUS_ADDR, RDATA_reg, read_status_resp);
                AESL_done_index_reg         <= RDATA_reg[1 : 1];
                AESL_ready_out_index_reg    <= RDATA_reg[1 : 1];
                AESL_idle_index_reg         <= RDATA_reg[2 : 2];
            process_0_finish = 1;
            process_busy = 0;
        end 
        @(posedge clk);
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_Q_run_flag <= 0; 
        count_operate_depth_by_bitwidth_and_depth (Q_c_bitwidth, Q_DEPTH, Q_OPERATE_DEPTH);
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_Q_run_flag <= 1; 
        end
        else if ((write_one_Q_data_done == 1 && write_Q_count == Q_diff_count - 1) || Q_diff_count == 0) begin
            write_Q_run_flag <= 0; 
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_Q_count = 0;
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_Q_count = 0;
        end
        if (write_one_Q_data_done === 1) begin
            write_Q_count = write_Q_count + 1;
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        Q_write_data_finish <= 0;
    end
    else begin
        if (TRAN_ctrl_start_in === 1) begin
            Q_write_data_finish <= 0;
        end
        if (write_Q_run_flag == 1 && write_Q_count == Q_diff_count) begin
            Q_write_data_finish <= 1;
        end
    end
end

initial begin : initial_diff_counter_Q
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer k;
    reg [31 : 0] Q_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = Q_c_bitwidth;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        wait (AESL_ready_reg === 1);
        Q_diff_count = 0;

        for (k = 0; k < Q_OPERATE_DEPTH; k = k + 1) begin
            for (i = 0; i < four_byte_num; i = i + 1) begin
                if (Q_c_bitwidth < 32) begin
                    Q_data_tmp_reg = mem_Q[k];
                end
                else begin
                    for (j = 0; j < 32; j = j + 1) begin
                        if (i*32 + j < Q_c_bitwidth) begin
                            Q_data_tmp_reg[j] = mem_Q[k][i*32 + j];
                        end
                        else begin
                            Q_data_tmp_reg[j] = 0;
                        end
                    end
                end
                if(image_mem_Q[k * four_byte_num  + i]!==Q_data_tmp_reg) begin
                Q_diff_count = Q_diff_count + 1;
                end
            end
        end

        @(posedge clk);
    end
end

initial begin : write_Q
    integer write_Q_resp;
    integer process_num ;
    integer get_ack;
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer check_Q_count;
    reg [31 : 0] Q_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = Q_c_bitwidth;
    process_num = 1;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        process_1_finish <= 0;

        for (check_Q_count = 0; check_Q_count < Q_OPERATE_DEPTH; check_Q_count = check_Q_count + 1) begin
            if (ongoing_process_number === process_num && process_busy === 0 ) begin
                get_ack = 1;
                if (write_Q_run_flag === 1 && get_ack === 1) begin
                    process_busy = 1;
                    //write Q data 
                    for (i = 0; i < four_byte_num; i = i + 1) begin
                        if (Q_c_bitwidth < 32) begin
                            Q_data_tmp_reg = mem_Q[check_Q_count];
                        end
                        else begin
                            for (j = 0; j < 32; j = j + 1) begin
                                if (i*32 + j < Q_c_bitwidth) begin
                                    Q_data_tmp_reg[j] = mem_Q[check_Q_count][i*32 + j];
                                end
                                else begin
                                    Q_data_tmp_reg[j] = 0;
                                end
                            end
                        end
                        if(image_mem_Q[check_Q_count * four_byte_num  + i]!==Q_data_tmp_reg) begin
                        write (Q_data_in_addr + check_Q_count * ceil_align_to_pow_of_two_four_byte_num * 4 + i * 4, Q_data_tmp_reg, write_Q_resp);
                        write_one_Q_data_done <= 1;
                        @(posedge clk);
                        write_one_Q_data_done <= 0;
                        image_mem_Q[check_Q_count * four_byte_num + i]=Q_data_tmp_reg;
                        end
                    end
                    process_busy = 0;
                end   
                process_1_finish <= 1;
            end
        end

        @(posedge clk);
    end    
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_K_run_flag <= 0; 
        count_operate_depth_by_bitwidth_and_depth (K_c_bitwidth, K_DEPTH, K_OPERATE_DEPTH);
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_K_run_flag <= 1; 
        end
        else if ((write_one_K_data_done == 1 && write_K_count == K_diff_count - 1) || K_diff_count == 0) begin
            write_K_run_flag <= 0; 
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_K_count = 0;
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_K_count = 0;
        end
        if (write_one_K_data_done === 1) begin
            write_K_count = write_K_count + 1;
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        K_write_data_finish <= 0;
    end
    else begin
        if (TRAN_ctrl_start_in === 1) begin
            K_write_data_finish <= 0;
        end
        if (write_K_run_flag == 1 && write_K_count == K_diff_count) begin
            K_write_data_finish <= 1;
        end
    end
end

initial begin : initial_diff_counter_K
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer k;
    reg [31 : 0] K_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = K_c_bitwidth;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        wait (AESL_ready_reg === 1);
        K_diff_count = 0;

        for (k = 0; k < K_OPERATE_DEPTH; k = k + 1) begin
            for (i = 0; i < four_byte_num; i = i + 1) begin
                if (K_c_bitwidth < 32) begin
                    K_data_tmp_reg = mem_K[k];
                end
                else begin
                    for (j = 0; j < 32; j = j + 1) begin
                        if (i*32 + j < K_c_bitwidth) begin
                            K_data_tmp_reg[j] = mem_K[k][i*32 + j];
                        end
                        else begin
                            K_data_tmp_reg[j] = 0;
                        end
                    end
                end
                if(image_mem_K[k * four_byte_num  + i]!==K_data_tmp_reg) begin
                K_diff_count = K_diff_count + 1;
                end
            end
        end

        @(posedge clk);
    end
end

initial begin : write_K
    integer write_K_resp;
    integer process_num ;
    integer get_ack;
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer check_K_count;
    reg [31 : 0] K_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = K_c_bitwidth;
    process_num = 2;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        process_2_finish <= 0;

        for (check_K_count = 0; check_K_count < K_OPERATE_DEPTH; check_K_count = check_K_count + 1) begin
            if (ongoing_process_number === process_num && process_busy === 0 ) begin
                get_ack = 1;
                if (write_K_run_flag === 1 && get_ack === 1) begin
                    process_busy = 1;
                    //write K data 
                    for (i = 0; i < four_byte_num; i = i + 1) begin
                        if (K_c_bitwidth < 32) begin
                            K_data_tmp_reg = mem_K[check_K_count];
                        end
                        else begin
                            for (j = 0; j < 32; j = j + 1) begin
                                if (i*32 + j < K_c_bitwidth) begin
                                    K_data_tmp_reg[j] = mem_K[check_K_count][i*32 + j];
                                end
                                else begin
                                    K_data_tmp_reg[j] = 0;
                                end
                            end
                        end
                        if(image_mem_K[check_K_count * four_byte_num  + i]!==K_data_tmp_reg) begin
                        write (K_data_in_addr + check_K_count * ceil_align_to_pow_of_two_four_byte_num * 4 + i * 4, K_data_tmp_reg, write_K_resp);
                        write_one_K_data_done <= 1;
                        @(posedge clk);
                        write_one_K_data_done <= 0;
                        image_mem_K[check_K_count * four_byte_num + i]=K_data_tmp_reg;
                        end
                    end
                    process_busy = 0;
                end   
                process_2_finish <= 1;
            end
        end

        @(posedge clk);
    end    
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_V_run_flag <= 0; 
        count_operate_depth_by_bitwidth_and_depth (V_c_bitwidth, V_DEPTH, V_OPERATE_DEPTH);
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_V_run_flag <= 1; 
        end
        else if ((write_one_V_data_done == 1 && write_V_count == V_diff_count - 1) || V_diff_count == 0) begin
            write_V_run_flag <= 0; 
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_V_count = 0;
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_V_count = 0;
        end
        if (write_one_V_data_done === 1) begin
            write_V_count = write_V_count + 1;
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        V_write_data_finish <= 0;
    end
    else begin
        if (TRAN_ctrl_start_in === 1) begin
            V_write_data_finish <= 0;
        end
        if (write_V_run_flag == 1 && write_V_count == V_diff_count) begin
            V_write_data_finish <= 1;
        end
    end
end

initial begin : initial_diff_counter_V
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer k;
    reg [31 : 0] V_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = V_c_bitwidth;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        wait (AESL_ready_reg === 1);
        V_diff_count = 0;

        for (k = 0; k < V_OPERATE_DEPTH; k = k + 1) begin
            for (i = 0; i < four_byte_num; i = i + 1) begin
                if (V_c_bitwidth < 32) begin
                    V_data_tmp_reg = mem_V[k];
                end
                else begin
                    for (j = 0; j < 32; j = j + 1) begin
                        if (i*32 + j < V_c_bitwidth) begin
                            V_data_tmp_reg[j] = mem_V[k][i*32 + j];
                        end
                        else begin
                            V_data_tmp_reg[j] = 0;
                        end
                    end
                end
                if(image_mem_V[k * four_byte_num  + i]!==V_data_tmp_reg) begin
                V_diff_count = V_diff_count + 1;
                end
            end
        end

        @(posedge clk);
    end
end

initial begin : write_V
    integer write_V_resp;
    integer process_num ;
    integer get_ack;
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer check_V_count;
    reg [31 : 0] V_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = V_c_bitwidth;
    process_num = 3;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        process_3_finish <= 0;

        for (check_V_count = 0; check_V_count < V_OPERATE_DEPTH; check_V_count = check_V_count + 1) begin
            if (ongoing_process_number === process_num && process_busy === 0 ) begin
                get_ack = 1;
                if (write_V_run_flag === 1 && get_ack === 1) begin
                    process_busy = 1;
                    //write V data 
                    for (i = 0; i < four_byte_num; i = i + 1) begin
                        if (V_c_bitwidth < 32) begin
                            V_data_tmp_reg = mem_V[check_V_count];
                        end
                        else begin
                            for (j = 0; j < 32; j = j + 1) begin
                                if (i*32 + j < V_c_bitwidth) begin
                                    V_data_tmp_reg[j] = mem_V[check_V_count][i*32 + j];
                                end
                                else begin
                                    V_data_tmp_reg[j] = 0;
                                end
                            end
                        end
                        if(image_mem_V[check_V_count * four_byte_num  + i]!==V_data_tmp_reg) begin
                        write (V_data_in_addr + check_V_count * ceil_align_to_pow_of_two_four_byte_num * 4 + i * 4, V_data_tmp_reg, write_V_resp);
                        write_one_V_data_done <= 1;
                        @(posedge clk);
                        write_one_V_data_done <= 0;
                        image_mem_V[check_V_count * four_byte_num + i]=V_data_tmp_reg;
                        end
                    end
                    process_busy = 0;
                end   
                process_3_finish <= 1;
            end
        end

        @(posedge clk);
    end    
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_O_run_flag <= 0; 
        count_operate_depth_by_bitwidth_and_depth (O_c_bitwidth, O_DEPTH, O_OPERATE_DEPTH);
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_O_run_flag <= 1; 
        end
        else if ((write_one_O_data_done == 1 && write_O_count == O_diff_count - 1) || O_diff_count == 0) begin
            write_O_run_flag <= 0; 
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_O_count = 0;
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_O_count = 0;
        end
        if (write_one_O_data_done === 1) begin
            write_O_count = write_O_count + 1;
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        O_write_data_finish <= 0;
    end
    else begin
        if (TRAN_ctrl_start_in === 1) begin
            O_write_data_finish <= 0;
        end
        if (write_O_run_flag == 1 && write_O_count == O_diff_count) begin
            O_write_data_finish <= 1;
        end
    end
end

initial begin : initial_diff_counter_O
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer k;
    reg [31 : 0] O_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = O_c_bitwidth;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        wait (AESL_ready_reg === 1);
        O_diff_count = 0;

        for (k = 0; k < O_OPERATE_DEPTH; k = k + 1) begin
            for (i = 0; i < four_byte_num; i = i + 1) begin
                if (O_c_bitwidth < 32) begin
                    O_data_tmp_reg = mem_O[k];
                end
                else begin
                    for (j = 0; j < 32; j = j + 1) begin
                        if (i*32 + j < O_c_bitwidth) begin
                            O_data_tmp_reg[j] = mem_O[k][i*32 + j];
                        end
                        else begin
                            O_data_tmp_reg[j] = 0;
                        end
                    end
                end
                if(image_mem_O[k * four_byte_num  + i]!==O_data_tmp_reg) begin
                O_diff_count = O_diff_count + 1;
                end
            end
        end

        @(posedge clk);
    end
end

initial begin : write_O
    integer write_O_resp;
    integer process_num ;
    integer get_ack;
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer check_O_count;
    reg [31 : 0] O_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = O_c_bitwidth;
    process_num = 4;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        process_4_finish <= 0;

        for (check_O_count = 0; check_O_count < O_OPERATE_DEPTH; check_O_count = check_O_count + 1) begin
            if (ongoing_process_number === process_num && process_busy === 0 ) begin
                get_ack = 1;
                if (write_O_run_flag === 1 && get_ack === 1) begin
                    process_busy = 1;
                    //write O data 
                    for (i = 0; i < four_byte_num; i = i + 1) begin
                        if (O_c_bitwidth < 32) begin
                            O_data_tmp_reg = mem_O[check_O_count];
                        end
                        else begin
                            for (j = 0; j < 32; j = j + 1) begin
                                if (i*32 + j < O_c_bitwidth) begin
                                    O_data_tmp_reg[j] = mem_O[check_O_count][i*32 + j];
                                end
                                else begin
                                    O_data_tmp_reg[j] = 0;
                                end
                            end
                        end
                        if(image_mem_O[check_O_count * four_byte_num  + i]!==O_data_tmp_reg) begin
                        write (O_data_in_addr + check_O_count * ceil_align_to_pow_of_two_four_byte_num * 4 + i * 4, O_data_tmp_reg, write_O_resp);
                        write_one_O_data_done <= 1;
                        @(posedge clk);
                        write_one_O_data_done <= 0;
                        image_mem_O[check_O_count * four_byte_num + i]=O_data_tmp_reg;
                        end
                    end
                    process_busy = 0;
                end   
                process_4_finish <= 1;
            end
        end

        @(posedge clk);
    end    
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_N_run_flag <= 0; 
        count_operate_depth_by_bitwidth_and_depth (N_c_bitwidth, N_DEPTH, N_OPERATE_DEPTH);
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_N_run_flag <= 1; 
        end
        else if ((write_one_N_data_done == 1 && write_N_count == N_diff_count - 1) || N_diff_count == 0) begin
            write_N_run_flag <= 0; 
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_N_count = 0;
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_N_count = 0;
        end
        if (write_one_N_data_done === 1) begin
            write_N_count = write_N_count + 1;
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        N_write_data_finish <= 0;
    end
    else begin
        if (TRAN_ctrl_start_in === 1) begin
            N_write_data_finish <= 0;
        end
        if (write_N_run_flag == 1 && write_N_count == N_diff_count) begin
            N_write_data_finish <= 1;
        end
    end
end

initial begin : initial_diff_counter_N
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer k;
    reg [31 : 0] N_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = N_c_bitwidth;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        wait (AESL_ready_reg === 1);
        N_diff_count = 0;

        for (k = 0; k < N_OPERATE_DEPTH; k = k + 1) begin
            for (i = 0; i < four_byte_num; i = i + 1) begin
                if (N_c_bitwidth < 32) begin
                    N_data_tmp_reg = mem_N[k];
                end
                else begin
                    for (j = 0; j < 32; j = j + 1) begin
                        if (i*32 + j < N_c_bitwidth) begin
                            N_data_tmp_reg[j] = mem_N[k][i*32 + j];
                        end
                        else begin
                            N_data_tmp_reg[j] = 0;
                        end
                    end
                end
                if(image_mem_N[k * four_byte_num  + i]!==N_data_tmp_reg) begin
                N_diff_count = N_diff_count + 1;
                end
            end
        end

        @(posedge clk);
    end
end

initial begin : write_N
    integer write_N_resp;
    integer process_num ;
    integer get_ack;
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer check_N_count;
    reg [31 : 0] N_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = N_c_bitwidth;
    process_num = 5;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        process_5_finish <= 0;

        for (check_N_count = 0; check_N_count < N_OPERATE_DEPTH; check_N_count = check_N_count + 1) begin
            if (ongoing_process_number === process_num && process_busy === 0 ) begin
                get_ack = 1;
                if (write_N_run_flag === 1 && get_ack === 1) begin
                    process_busy = 1;
                    //write N data 
                    for (i = 0; i < four_byte_num; i = i + 1) begin
                        if (N_c_bitwidth < 32) begin
                            N_data_tmp_reg = mem_N[check_N_count];
                        end
                        else begin
                            for (j = 0; j < 32; j = j + 1) begin
                                if (i*32 + j < N_c_bitwidth) begin
                                    N_data_tmp_reg[j] = mem_N[check_N_count][i*32 + j];
                                end
                                else begin
                                    N_data_tmp_reg[j] = 0;
                                end
                            end
                        end
                        if(image_mem_N[check_N_count * four_byte_num  + i]!==N_data_tmp_reg) begin
                        write (N_data_in_addr + check_N_count * ceil_align_to_pow_of_two_four_byte_num * 4 + i * 4, N_data_tmp_reg, write_N_resp);
                        write_one_N_data_done <= 1;
                        @(posedge clk);
                        write_one_N_data_done <= 0;
                        image_mem_N[check_N_count * four_byte_num + i]=N_data_tmp_reg;
                        end
                    end
                    process_busy = 0;
                end   
                process_5_finish <= 1;
            end
        end

        @(posedge clk);
    end    
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_d_run_flag <= 0; 
        count_operate_depth_by_bitwidth_and_depth (d_c_bitwidth, d_DEPTH, d_OPERATE_DEPTH);
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_d_run_flag <= 1; 
        end
        else if ((write_one_d_data_done == 1 && write_d_count == d_diff_count - 1) || d_diff_count == 0) begin
            write_d_run_flag <= 0; 
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_d_count = 0;
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_d_count = 0;
        end
        if (write_one_d_data_done === 1) begin
            write_d_count = write_d_count + 1;
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        d_write_data_finish <= 0;
    end
    else begin
        if (TRAN_ctrl_start_in === 1) begin
            d_write_data_finish <= 0;
        end
        if (write_d_run_flag == 1 && write_d_count == d_diff_count) begin
            d_write_data_finish <= 1;
        end
    end
end

initial begin : initial_diff_counter_d
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer k;
    reg [31 : 0] d_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = d_c_bitwidth;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        wait (AESL_ready_reg === 1);
        d_diff_count = 0;

        for (k = 0; k < d_OPERATE_DEPTH; k = k + 1) begin
            for (i = 0; i < four_byte_num; i = i + 1) begin
                if (d_c_bitwidth < 32) begin
                    d_data_tmp_reg = mem_d[k];
                end
                else begin
                    for (j = 0; j < 32; j = j + 1) begin
                        if (i*32 + j < d_c_bitwidth) begin
                            d_data_tmp_reg[j] = mem_d[k][i*32 + j];
                        end
                        else begin
                            d_data_tmp_reg[j] = 0;
                        end
                    end
                end
                if(image_mem_d[k * four_byte_num  + i]!==d_data_tmp_reg) begin
                d_diff_count = d_diff_count + 1;
                end
            end
        end

        @(posedge clk);
    end
end

initial begin : write_d
    integer write_d_resp;
    integer process_num ;
    integer get_ack;
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer check_d_count;
    reg [31 : 0] d_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = d_c_bitwidth;
    process_num = 6;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        process_6_finish <= 0;

        for (check_d_count = 0; check_d_count < d_OPERATE_DEPTH; check_d_count = check_d_count + 1) begin
            if (ongoing_process_number === process_num && process_busy === 0 ) begin
                get_ack = 1;
                if (write_d_run_flag === 1 && get_ack === 1) begin
                    process_busy = 1;
                    //write d data 
                    for (i = 0; i < four_byte_num; i = i + 1) begin
                        if (d_c_bitwidth < 32) begin
                            d_data_tmp_reg = mem_d[check_d_count];
                        end
                        else begin
                            for (j = 0; j < 32; j = j + 1) begin
                                if (i*32 + j < d_c_bitwidth) begin
                                    d_data_tmp_reg[j] = mem_d[check_d_count][i*32 + j];
                                end
                                else begin
                                    d_data_tmp_reg[j] = 0;
                                end
                            end
                        end
                        if(image_mem_d[check_d_count * four_byte_num  + i]!==d_data_tmp_reg) begin
                        write (d_data_in_addr + check_d_count * ceil_align_to_pow_of_two_four_byte_num * 4 + i * 4, d_data_tmp_reg, write_d_resp);
                        write_one_d_data_done <= 1;
                        @(posedge clk);
                        write_one_d_data_done <= 0;
                        image_mem_d[check_d_count * four_byte_num + i]=d_data_tmp_reg;
                        end
                    end
                    process_busy = 0;
                end   
                process_6_finish <= 1;
            end
        end

        @(posedge clk);
    end    
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_causal_run_flag <= 0; 
        count_operate_depth_by_bitwidth_and_depth (causal_c_bitwidth, causal_DEPTH, causal_OPERATE_DEPTH);
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_causal_run_flag <= 1; 
        end
        else if ((write_one_causal_data_done == 1 && write_causal_count == causal_diff_count - 1) || causal_diff_count == 0) begin
            write_causal_run_flag <= 0; 
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_causal_count = 0;
    end
    else begin
        if (AESL_ready_reg === 1) begin
            write_causal_count = 0;
        end
        if (write_one_causal_data_done === 1) begin
            write_causal_count = write_causal_count + 1;
        end
    end
end

always @(reset or posedge clk) begin
    if (reset == 0) begin
        causal_write_data_finish <= 0;
    end
    else begin
        if (TRAN_ctrl_start_in === 1) begin
            causal_write_data_finish <= 0;
        end
        if (write_causal_run_flag == 1 && write_causal_count == causal_diff_count) begin
            causal_write_data_finish <= 1;
        end
    end
end

initial begin : initial_diff_counter_causal
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer k;
    reg [31 : 0] causal_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = causal_c_bitwidth;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        wait (AESL_ready_reg === 1);
        causal_diff_count = 0;

        for (k = 0; k < causal_OPERATE_DEPTH; k = k + 1) begin
            for (i = 0; i < four_byte_num; i = i + 1) begin
                if (causal_c_bitwidth < 32) begin
                    causal_data_tmp_reg = mem_causal[k];
                end
                else begin
                    for (j = 0; j < 32; j = j + 1) begin
                        if (i*32 + j < causal_c_bitwidth) begin
                            causal_data_tmp_reg[j] = mem_causal[k][i*32 + j];
                        end
                        else begin
                            causal_data_tmp_reg[j] = 0;
                        end
                    end
                end
                if(image_mem_causal[k * four_byte_num  + i]!==causal_data_tmp_reg) begin
                causal_diff_count = causal_diff_count + 1;
                end
            end
        end

        @(posedge clk);
    end
end

initial begin : write_causal
    integer write_causal_resp;
    integer process_num ;
    integer get_ack;
    integer four_byte_num;
    integer ceil_align_to_pow_of_two_four_byte_num;
    integer c_bitwidth;
    integer i;
    integer j;
    integer check_causal_count;
    reg [31 : 0] causal_data_tmp_reg;
    wait(reset === 1);
    @(posedge clk);
    c_bitwidth = causal_c_bitwidth;
    process_num = 7;
    count_c_data_four_byte_num_by_bitwidth (c_bitwidth , four_byte_num);
    ceil_align_to_pow_of_two_four_byte_num = ceil_align_to_pow_of_two(four_byte_num);
    while (1) begin
        process_7_finish <= 0;

        for (check_causal_count = 0; check_causal_count < causal_OPERATE_DEPTH; check_causal_count = check_causal_count + 1) begin
            if (ongoing_process_number === process_num && process_busy === 0 ) begin
                get_ack = 1;
                if (write_causal_run_flag === 1 && get_ack === 1) begin
                    process_busy = 1;
                    //write causal data 
                    for (i = 0; i < four_byte_num; i = i + 1) begin
                        if (causal_c_bitwidth < 32) begin
                            causal_data_tmp_reg = mem_causal[check_causal_count];
                        end
                        else begin
                            for (j = 0; j < 32; j = j + 1) begin
                                if (i*32 + j < causal_c_bitwidth) begin
                                    causal_data_tmp_reg[j] = mem_causal[check_causal_count][i*32 + j];
                                end
                                else begin
                                    causal_data_tmp_reg[j] = 0;
                                end
                            end
                        end
                        if(image_mem_causal[check_causal_count * four_byte_num  + i]!==causal_data_tmp_reg) begin
                        write (causal_data_in_addr + check_causal_count * ceil_align_to_pow_of_two_four_byte_num * 4 + i * 4, causal_data_tmp_reg, write_causal_resp);
                        write_one_causal_data_done <= 1;
                        @(posedge clk);
                        write_one_causal_data_done <= 0;
                        image_mem_causal[check_causal_count * four_byte_num + i]=causal_data_tmp_reg;
                        end
                    end
                    process_busy = 0;
                end   
                process_7_finish <= 1;
            end
        end

        @(posedge clk);
    end    
end


always @(reset or posedge clk) begin
    if (reset == 0) begin
        write_start_run_flag <= 0; 
        write_start_count <= 0;
    end
    else begin
        if (write_start_count >= 5) begin
            write_start_run_flag <= 0; 
        end
        else if (TRAN_ctrl_write_start_in === 1) begin
            write_start_run_flag <= 1; 
        end
        if (AESL_write_start_finish === 1) begin
            write_start_count <= write_start_count + 1;
            write_start_run_flag <= 0; 
        end
    end
end

initial begin : write_start
    reg [DATA_WIDTH - 1 : 0] write_start_tmp;
    integer process_num;
    integer write_start_resp;
    wait(reset === 1);
    @(posedge clk);
    process_num = 8;
    while (1) begin
        process_8_finish = 0;
        if (ongoing_process_number === process_num && process_busy === 0 ) begin
            if (write_start_run_flag === 1) begin
                process_busy = 1;
                write_start_tmp=0;
                write_start_tmp[0 : 0] = 1;
                write (START_ADDR, write_start_tmp, write_start_resp);
                process_busy = 0;
                AESL_write_start_finish <= 1;
                @(posedge clk);
                AESL_write_start_finish <= 0;
            end
            process_8_finish <= 1;
        end 
        @(posedge clk);
    end
end

//------------------------Task and function-------------- 
task read_token; 
    input integer fp; 
    output reg [151 : 0] token;
    integer ret;
    begin
        token = "";
        ret = 0;
        ret = $fscanf(fp,"%s",token);
    end 
endtask 
 
//------------------------Read file------------------------ 
 
// Read data from file 
initial begin : read_Q_file_process 
  integer fp; 
  integer ret; 
  integer factor; 
  reg [151 : 0] token; 
  reg [151 : 0] token_tmp; 
  //reg [Q_c_bitwidth - 1 : 0] token_tmp; 
  reg [DATA_WIDTH - 1 : 0] tmp_cache_mem; 
  reg [ 8*5 : 1] str;
    reg [63:0] trans_depth;
  integer transaction_idx; 
  integer i; 
  transaction_idx = 0; 
  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
  count_seperate_factor_by_bitwidth (Q_c_bitwidth , factor);
  fp = $fopen(`TV_IN_Q ,"r"); 
  if(fp == 0) begin                               // Failed to open file 
      $display("Failed to open file \"%s\"!", `TV_IN_Q); 
      $finish; 
  end 
  read_token(fp, token); 
  if (token != "[[[runtime]]]") begin             // Illegal format 
      $display("ERROR: Simulation using HLS TB failed.");
      $finish; 
  end 
  read_token(fp, token); 
  while (token != "[[[/runtime]]]") begin 
      if (token != "[[transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token);                        // skip transaction number 
      @(posedge clk);
      # 0.2;
      while(AESL_ready_reg !== 1) begin
          @(posedge clk); 
          # 0.2;
      end
      for(i = 0; i < Q_DEPTH; i = i + 1) begin 
          read_token(fp, token); 
          ret = $sscanf(token, "0x%x", token_tmp); 
          if (factor == 4) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [7 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [15 : 8] = token_tmp;
              end
              if (i%factor == 2) begin
                  tmp_cache_mem [23 : 16] = token_tmp;
              end
              if (i%factor == 3) begin
                  tmp_cache_mem [31 : 24] = token_tmp;
                  mem_Q [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
              end
          end
          if (factor == 2) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [15 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [31 : 16] = token_tmp;
                  mem_Q [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1: 0] = 0;
              end
          end
          if (factor == 1) begin
              mem_Q [i] = token_tmp;
          end
      end 
      if (factor == 4) begin
          if (i%factor != 0) begin
              mem_Q [i/factor] = tmp_cache_mem;
          end
      end
      if (factor == 2) begin
          if (i%factor != 0) begin
              mem_Q [i/factor] = tmp_cache_mem;
          end
      end 
      read_token(fp, token); 
      if(token != "[[/transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token); 
      transaction_idx = transaction_idx + 1; 
  end 
  $fclose(fp); 
end 
 
task write_binary_Q;
    input integer fp;
    input reg[64-1:0] in;
    input integer in_bw;
    reg [63:0] tmp_long;
    reg[64-1:0] local_in;
    integer char_num;
    integer long_num;
    integer i;
    integer j;
    begin
        long_num = (in_bw + 63) / 64;
        char_num = ((in_bw - 1) % 64 + 7) / 8;
        for(i=long_num;i>0;i=i-1) begin
             local_in = in;
             tmp_long = local_in >> ((i-1)*64);
             for(j=0;j<64;j=j+1)
                 if (tmp_long[j] === 1'bx)
                     tmp_long[j] = 1'b0;
             if (i == long_num) begin
                 case(char_num)
                     1: $fwrite(fp,"%c",tmp_long[7:0]);
                     2: $fwrite(fp,"%c%c",tmp_long[15:8],tmp_long[7:0]);
                     3: $fwrite(fp,"%c%c%c",tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     4: $fwrite(fp,"%c%c%c%c",tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     5: $fwrite(fp,"%c%c%c%c%c",tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     6: $fwrite(fp,"%c%c%c%c%c%c",tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     7: $fwrite(fp,"%c%c%c%c%c%c%c",tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     8: $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     default: ;
                 endcase
             end
             else begin
                 $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
             end
        end
    end
endtask;
//------------------------Read file------------------------ 
 
// Read data from file 
initial begin : read_K_file_process 
  integer fp; 
  integer ret; 
  integer factor; 
  reg [151 : 0] token; 
  reg [151 : 0] token_tmp; 
  //reg [K_c_bitwidth - 1 : 0] token_tmp; 
  reg [DATA_WIDTH - 1 : 0] tmp_cache_mem; 
  reg [ 8*5 : 1] str;
    reg [63:0] trans_depth;
  integer transaction_idx; 
  integer i; 
  transaction_idx = 0; 
  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
  count_seperate_factor_by_bitwidth (K_c_bitwidth , factor);
  fp = $fopen(`TV_IN_K ,"r"); 
  if(fp == 0) begin                               // Failed to open file 
      $display("Failed to open file \"%s\"!", `TV_IN_K); 
      $finish; 
  end 
  read_token(fp, token); 
  if (token != "[[[runtime]]]") begin             // Illegal format 
      $display("ERROR: Simulation using HLS TB failed.");
      $finish; 
  end 
  read_token(fp, token); 
  while (token != "[[[/runtime]]]") begin 
      if (token != "[[transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token);                        // skip transaction number 
      @(posedge clk);
      # 0.2;
      while(AESL_ready_reg !== 1) begin
          @(posedge clk); 
          # 0.2;
      end
      for(i = 0; i < K_DEPTH; i = i + 1) begin 
          read_token(fp, token); 
          ret = $sscanf(token, "0x%x", token_tmp); 
          if (factor == 4) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [7 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [15 : 8] = token_tmp;
              end
              if (i%factor == 2) begin
                  tmp_cache_mem [23 : 16] = token_tmp;
              end
              if (i%factor == 3) begin
                  tmp_cache_mem [31 : 24] = token_tmp;
                  mem_K [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
              end
          end
          if (factor == 2) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [15 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [31 : 16] = token_tmp;
                  mem_K [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1: 0] = 0;
              end
          end
          if (factor == 1) begin
              mem_K [i] = token_tmp;
          end
      end 
      if (factor == 4) begin
          if (i%factor != 0) begin
              mem_K [i/factor] = tmp_cache_mem;
          end
      end
      if (factor == 2) begin
          if (i%factor != 0) begin
              mem_K [i/factor] = tmp_cache_mem;
          end
      end 
      read_token(fp, token); 
      if(token != "[[/transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token); 
      transaction_idx = transaction_idx + 1; 
  end 
  $fclose(fp); 
end 
 
task write_binary_K;
    input integer fp;
    input reg[64-1:0] in;
    input integer in_bw;
    reg [63:0] tmp_long;
    reg[64-1:0] local_in;
    integer char_num;
    integer long_num;
    integer i;
    integer j;
    begin
        long_num = (in_bw + 63) / 64;
        char_num = ((in_bw - 1) % 64 + 7) / 8;
        for(i=long_num;i>0;i=i-1) begin
             local_in = in;
             tmp_long = local_in >> ((i-1)*64);
             for(j=0;j<64;j=j+1)
                 if (tmp_long[j] === 1'bx)
                     tmp_long[j] = 1'b0;
             if (i == long_num) begin
                 case(char_num)
                     1: $fwrite(fp,"%c",tmp_long[7:0]);
                     2: $fwrite(fp,"%c%c",tmp_long[15:8],tmp_long[7:0]);
                     3: $fwrite(fp,"%c%c%c",tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     4: $fwrite(fp,"%c%c%c%c",tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     5: $fwrite(fp,"%c%c%c%c%c",tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     6: $fwrite(fp,"%c%c%c%c%c%c",tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     7: $fwrite(fp,"%c%c%c%c%c%c%c",tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     8: $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     default: ;
                 endcase
             end
             else begin
                 $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
             end
        end
    end
endtask;
//------------------------Read file------------------------ 
 
// Read data from file 
initial begin : read_V_file_process 
  integer fp; 
  integer ret; 
  integer factor; 
  reg [151 : 0] token; 
  reg [151 : 0] token_tmp; 
  //reg [V_c_bitwidth - 1 : 0] token_tmp; 
  reg [DATA_WIDTH - 1 : 0] tmp_cache_mem; 
  reg [ 8*5 : 1] str;
    reg [63:0] trans_depth;
  integer transaction_idx; 
  integer i; 
  transaction_idx = 0; 
  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
  count_seperate_factor_by_bitwidth (V_c_bitwidth , factor);
  fp = $fopen(`TV_IN_V ,"r"); 
  if(fp == 0) begin                               // Failed to open file 
      $display("Failed to open file \"%s\"!", `TV_IN_V); 
      $finish; 
  end 
  read_token(fp, token); 
  if (token != "[[[runtime]]]") begin             // Illegal format 
      $display("ERROR: Simulation using HLS TB failed.");
      $finish; 
  end 
  read_token(fp, token); 
  while (token != "[[[/runtime]]]") begin 
      if (token != "[[transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token);                        // skip transaction number 
      @(posedge clk);
      # 0.2;
      while(AESL_ready_reg !== 1) begin
          @(posedge clk); 
          # 0.2;
      end
      for(i = 0; i < V_DEPTH; i = i + 1) begin 
          read_token(fp, token); 
          ret = $sscanf(token, "0x%x", token_tmp); 
          if (factor == 4) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [7 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [15 : 8] = token_tmp;
              end
              if (i%factor == 2) begin
                  tmp_cache_mem [23 : 16] = token_tmp;
              end
              if (i%factor == 3) begin
                  tmp_cache_mem [31 : 24] = token_tmp;
                  mem_V [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
              end
          end
          if (factor == 2) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [15 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [31 : 16] = token_tmp;
                  mem_V [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1: 0] = 0;
              end
          end
          if (factor == 1) begin
              mem_V [i] = token_tmp;
          end
      end 
      if (factor == 4) begin
          if (i%factor != 0) begin
              mem_V [i/factor] = tmp_cache_mem;
          end
      end
      if (factor == 2) begin
          if (i%factor != 0) begin
              mem_V [i/factor] = tmp_cache_mem;
          end
      end 
      read_token(fp, token); 
      if(token != "[[/transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token); 
      transaction_idx = transaction_idx + 1; 
  end 
  $fclose(fp); 
end 
 
task write_binary_V;
    input integer fp;
    input reg[64-1:0] in;
    input integer in_bw;
    reg [63:0] tmp_long;
    reg[64-1:0] local_in;
    integer char_num;
    integer long_num;
    integer i;
    integer j;
    begin
        long_num = (in_bw + 63) / 64;
        char_num = ((in_bw - 1) % 64 + 7) / 8;
        for(i=long_num;i>0;i=i-1) begin
             local_in = in;
             tmp_long = local_in >> ((i-1)*64);
             for(j=0;j<64;j=j+1)
                 if (tmp_long[j] === 1'bx)
                     tmp_long[j] = 1'b0;
             if (i == long_num) begin
                 case(char_num)
                     1: $fwrite(fp,"%c",tmp_long[7:0]);
                     2: $fwrite(fp,"%c%c",tmp_long[15:8],tmp_long[7:0]);
                     3: $fwrite(fp,"%c%c%c",tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     4: $fwrite(fp,"%c%c%c%c",tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     5: $fwrite(fp,"%c%c%c%c%c",tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     6: $fwrite(fp,"%c%c%c%c%c%c",tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     7: $fwrite(fp,"%c%c%c%c%c%c%c",tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     8: $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     default: ;
                 endcase
             end
             else begin
                 $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
             end
        end
    end
endtask;
//------------------------Read file------------------------ 
 
// Read data from file 
initial begin : read_O_file_process 
  integer fp; 
  integer ret; 
  integer factor; 
  reg [151 : 0] token; 
  reg [151 : 0] token_tmp; 
  //reg [O_c_bitwidth - 1 : 0] token_tmp; 
  reg [DATA_WIDTH - 1 : 0] tmp_cache_mem; 
  reg [ 8*5 : 1] str;
    reg [63:0] trans_depth;
  integer transaction_idx; 
  integer i; 
  transaction_idx = 0; 
  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
  count_seperate_factor_by_bitwidth (O_c_bitwidth , factor);
  fp = $fopen(`TV_IN_O ,"r"); 
  if(fp == 0) begin                               // Failed to open file 
      $display("Failed to open file \"%s\"!", `TV_IN_O); 
      $finish; 
  end 
  read_token(fp, token); 
  if (token != "[[[runtime]]]") begin             // Illegal format 
      $display("ERROR: Simulation using HLS TB failed.");
      $finish; 
  end 
  read_token(fp, token); 
  while (token != "[[[/runtime]]]") begin 
      if (token != "[[transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token);                        // skip transaction number 
      @(posedge clk);
      # 0.2;
      while(AESL_ready_reg !== 1) begin
          @(posedge clk); 
          # 0.2;
      end
      for(i = 0; i < O_DEPTH; i = i + 1) begin 
          read_token(fp, token); 
          ret = $sscanf(token, "0x%x", token_tmp); 
          if (factor == 4) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [7 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [15 : 8] = token_tmp;
              end
              if (i%factor == 2) begin
                  tmp_cache_mem [23 : 16] = token_tmp;
              end
              if (i%factor == 3) begin
                  tmp_cache_mem [31 : 24] = token_tmp;
                  mem_O [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
              end
          end
          if (factor == 2) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [15 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [31 : 16] = token_tmp;
                  mem_O [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1: 0] = 0;
              end
          end
          if (factor == 1) begin
              mem_O [i] = token_tmp;
          end
      end 
      if (factor == 4) begin
          if (i%factor != 0) begin
              mem_O [i/factor] = tmp_cache_mem;
          end
      end
      if (factor == 2) begin
          if (i%factor != 0) begin
              mem_O [i/factor] = tmp_cache_mem;
          end
      end 
      read_token(fp, token); 
      if(token != "[[/transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token); 
      transaction_idx = transaction_idx + 1; 
  end 
  $fclose(fp); 
end 
 
task write_binary_O;
    input integer fp;
    input reg[64-1:0] in;
    input integer in_bw;
    reg [63:0] tmp_long;
    reg[64-1:0] local_in;
    integer char_num;
    integer long_num;
    integer i;
    integer j;
    begin
        long_num = (in_bw + 63) / 64;
        char_num = ((in_bw - 1) % 64 + 7) / 8;
        for(i=long_num;i>0;i=i-1) begin
             local_in = in;
             tmp_long = local_in >> ((i-1)*64);
             for(j=0;j<64;j=j+1)
                 if (tmp_long[j] === 1'bx)
                     tmp_long[j] = 1'b0;
             if (i == long_num) begin
                 case(char_num)
                     1: $fwrite(fp,"%c",tmp_long[7:0]);
                     2: $fwrite(fp,"%c%c",tmp_long[15:8],tmp_long[7:0]);
                     3: $fwrite(fp,"%c%c%c",tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     4: $fwrite(fp,"%c%c%c%c",tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     5: $fwrite(fp,"%c%c%c%c%c",tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     6: $fwrite(fp,"%c%c%c%c%c%c",tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     7: $fwrite(fp,"%c%c%c%c%c%c%c",tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     8: $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     default: ;
                 endcase
             end
             else begin
                 $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
             end
        end
    end
endtask;
//------------------------Read file------------------------ 
 
// Read data from file 
initial begin : read_N_file_process 
  integer fp; 
  integer ret; 
  integer factor; 
  reg [127 : 0] token; 
  reg [127 : 0] token_tmp; 
  //reg [N_c_bitwidth - 1 : 0] token_tmp; 
  reg [DATA_WIDTH - 1 : 0] tmp_cache_mem; 
  reg [ 8*5 : 1] str;
    reg [63:0] trans_depth;
  integer transaction_idx; 
  integer i; 
  transaction_idx = 0; 
  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
  count_seperate_factor_by_bitwidth (N_c_bitwidth , factor);
  fp = $fopen(`TV_IN_N ,"r"); 
  if(fp == 0) begin                               // Failed to open file 
      $display("Failed to open file \"%s\"!", `TV_IN_N); 
      $finish; 
  end 
  read_token(fp, token); 
  if (token != "[[[runtime]]]") begin             // Illegal format 
      $display("ERROR: Simulation using HLS TB failed.");
      $finish; 
  end 
  read_token(fp, token); 
  while (token != "[[[/runtime]]]") begin 
      if (token != "[[transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token);                        // skip transaction number 
      @(posedge clk);
      # 0.2;
      while(AESL_ready_reg !== 1) begin
          @(posedge clk); 
          # 0.2;
      end
      for(i = 0; i < N_DEPTH; i = i + 1) begin 
          read_token(fp, token); 
          ret = $sscanf(token, "0x%x", token_tmp); 
          if (factor == 4) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [7 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [15 : 8] = token_tmp;
              end
              if (i%factor == 2) begin
                  tmp_cache_mem [23 : 16] = token_tmp;
              end
              if (i%factor == 3) begin
                  tmp_cache_mem [31 : 24] = token_tmp;
                  mem_N [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
              end
          end
          if (factor == 2) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [15 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [31 : 16] = token_tmp;
                  mem_N [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1: 0] = 0;
              end
          end
          if (factor == 1) begin
              mem_N [i] = token_tmp;
          end
      end 
      if (factor == 4) begin
          if (i%factor != 0) begin
              mem_N [i/factor] = tmp_cache_mem;
          end
      end
      if (factor == 2) begin
          if (i%factor != 0) begin
              mem_N [i/factor] = tmp_cache_mem;
          end
      end 
      read_token(fp, token); 
      if(token != "[[/transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token); 
      transaction_idx = transaction_idx + 1; 
  end 
  $fclose(fp); 
end 
 
task write_binary_N;
    input integer fp;
    input reg[64-1:0] in;
    input integer in_bw;
    reg [63:0] tmp_long;
    reg[64-1:0] local_in;
    integer char_num;
    integer long_num;
    integer i;
    integer j;
    begin
        long_num = (in_bw + 63) / 64;
        char_num = ((in_bw - 1) % 64 + 7) / 8;
        for(i=long_num;i>0;i=i-1) begin
             local_in = in;
             tmp_long = local_in >> ((i-1)*64);
             for(j=0;j<64;j=j+1)
                 if (tmp_long[j] === 1'bx)
                     tmp_long[j] = 1'b0;
             if (i == long_num) begin
                 case(char_num)
                     1: $fwrite(fp,"%c",tmp_long[7:0]);
                     2: $fwrite(fp,"%c%c",tmp_long[15:8],tmp_long[7:0]);
                     3: $fwrite(fp,"%c%c%c",tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     4: $fwrite(fp,"%c%c%c%c",tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     5: $fwrite(fp,"%c%c%c%c%c",tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     6: $fwrite(fp,"%c%c%c%c%c%c",tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     7: $fwrite(fp,"%c%c%c%c%c%c%c",tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     8: $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     default: ;
                 endcase
             end
             else begin
                 $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
             end
        end
    end
endtask;
//------------------------Read file------------------------ 
 
// Read data from file 
initial begin : read_d_file_process 
  integer fp; 
  integer ret; 
  integer factor; 
  reg [127 : 0] token; 
  reg [127 : 0] token_tmp; 
  //reg [d_c_bitwidth - 1 : 0] token_tmp; 
  reg [DATA_WIDTH - 1 : 0] tmp_cache_mem; 
  reg [ 8*5 : 1] str;
    reg [63:0] trans_depth;
  integer transaction_idx; 
  integer i; 
  transaction_idx = 0; 
  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
  count_seperate_factor_by_bitwidth (d_c_bitwidth , factor);
  fp = $fopen(`TV_IN_d ,"r"); 
  if(fp == 0) begin                               // Failed to open file 
      $display("Failed to open file \"%s\"!", `TV_IN_d); 
      $finish; 
  end 
  read_token(fp, token); 
  if (token != "[[[runtime]]]") begin             // Illegal format 
      $display("ERROR: Simulation using HLS TB failed.");
      $finish; 
  end 
  read_token(fp, token); 
  while (token != "[[[/runtime]]]") begin 
      if (token != "[[transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token);                        // skip transaction number 
      @(posedge clk);
      # 0.2;
      while(AESL_ready_reg !== 1) begin
          @(posedge clk); 
          # 0.2;
      end
      for(i = 0; i < d_DEPTH; i = i + 1) begin 
          read_token(fp, token); 
          ret = $sscanf(token, "0x%x", token_tmp); 
          if (factor == 4) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [7 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [15 : 8] = token_tmp;
              end
              if (i%factor == 2) begin
                  tmp_cache_mem [23 : 16] = token_tmp;
              end
              if (i%factor == 3) begin
                  tmp_cache_mem [31 : 24] = token_tmp;
                  mem_d [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
              end
          end
          if (factor == 2) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [15 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [31 : 16] = token_tmp;
                  mem_d [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1: 0] = 0;
              end
          end
          if (factor == 1) begin
              mem_d [i] = token_tmp;
          end
      end 
      if (factor == 4) begin
          if (i%factor != 0) begin
              mem_d [i/factor] = tmp_cache_mem;
          end
      end
      if (factor == 2) begin
          if (i%factor != 0) begin
              mem_d [i/factor] = tmp_cache_mem;
          end
      end 
      read_token(fp, token); 
      if(token != "[[/transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token); 
      transaction_idx = transaction_idx + 1; 
  end 
  $fclose(fp); 
end 
 
task write_binary_d;
    input integer fp;
    input reg[64-1:0] in;
    input integer in_bw;
    reg [63:0] tmp_long;
    reg[64-1:0] local_in;
    integer char_num;
    integer long_num;
    integer i;
    integer j;
    begin
        long_num = (in_bw + 63) / 64;
        char_num = ((in_bw - 1) % 64 + 7) / 8;
        for(i=long_num;i>0;i=i-1) begin
             local_in = in;
             tmp_long = local_in >> ((i-1)*64);
             for(j=0;j<64;j=j+1)
                 if (tmp_long[j] === 1'bx)
                     tmp_long[j] = 1'b0;
             if (i == long_num) begin
                 case(char_num)
                     1: $fwrite(fp,"%c",tmp_long[7:0]);
                     2: $fwrite(fp,"%c%c",tmp_long[15:8],tmp_long[7:0]);
                     3: $fwrite(fp,"%c%c%c",tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     4: $fwrite(fp,"%c%c%c%c",tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     5: $fwrite(fp,"%c%c%c%c%c",tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     6: $fwrite(fp,"%c%c%c%c%c%c",tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     7: $fwrite(fp,"%c%c%c%c%c%c%c",tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     8: $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     default: ;
                 endcase
             end
             else begin
                 $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
             end
        end
    end
endtask;
//------------------------Read file------------------------ 
 
// Read data from file 
initial begin : read_causal_file_process 
  integer fp; 
  integer ret; 
  integer factor; 
  reg [127 : 0] token; 
  reg [127 : 0] token_tmp; 
  //reg [causal_c_bitwidth - 1 : 0] token_tmp; 
  reg [DATA_WIDTH - 1 : 0] tmp_cache_mem; 
  reg [ 8*5 : 1] str;
    reg [63:0] trans_depth;
  integer transaction_idx; 
  integer i; 
  transaction_idx = 0; 
  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
  count_seperate_factor_by_bitwidth (causal_c_bitwidth , factor);
  fp = $fopen(`TV_IN_causal ,"r"); 
  if(fp == 0) begin                               // Failed to open file 
      $display("Failed to open file \"%s\"!", `TV_IN_causal); 
      $finish; 
  end 
  read_token(fp, token); 
  if (token != "[[[runtime]]]") begin             // Illegal format 
      $display("ERROR: Simulation using HLS TB failed.");
      $finish; 
  end 
  read_token(fp, token); 
  while (token != "[[[/runtime]]]") begin 
      if (token != "[[transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token);                        // skip transaction number 
      @(posedge clk);
      # 0.2;
      while(AESL_ready_reg !== 1) begin
          @(posedge clk); 
          # 0.2;
      end
      for(i = 0; i < causal_DEPTH; i = i + 1) begin 
          read_token(fp, token); 
          ret = $sscanf(token, "0x%x", token_tmp); 
          if (factor == 4) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [7 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [15 : 8] = token_tmp;
              end
              if (i%factor == 2) begin
                  tmp_cache_mem [23 : 16] = token_tmp;
              end
              if (i%factor == 3) begin
                  tmp_cache_mem [31 : 24] = token_tmp;
                  mem_causal [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1 : 0] = 0;
              end
          end
          if (factor == 2) begin
              if (i%factor == 0) begin
                  tmp_cache_mem [15 : 0] = token_tmp;
              end
              if (i%factor == 1) begin
                  tmp_cache_mem [31 : 16] = token_tmp;
                  mem_causal [i/factor] = tmp_cache_mem;
                  tmp_cache_mem [DATA_WIDTH - 1: 0] = 0;
              end
          end
          if (factor == 1) begin
              mem_causal [i] = token_tmp;
          end
      end 
      if (factor == 4) begin
          if (i%factor != 0) begin
              mem_causal [i/factor] = tmp_cache_mem;
          end
      end
      if (factor == 2) begin
          if (i%factor != 0) begin
              mem_causal [i/factor] = tmp_cache_mem;
          end
      end 
      read_token(fp, token); 
      if(token != "[[/transaction]]") begin 
          $display("ERROR: Simulation using HLS TB failed.");
          $finish; 
      end 
      read_token(fp, token); 
      transaction_idx = transaction_idx + 1; 
  end 
  $fclose(fp); 
end 
 
task write_binary_causal;
    input integer fp;
    input reg[64-1:0] in;
    input integer in_bw;
    reg [63:0] tmp_long;
    reg[64-1:0] local_in;
    integer char_num;
    integer long_num;
    integer i;
    integer j;
    begin
        long_num = (in_bw + 63) / 64;
        char_num = ((in_bw - 1) % 64 + 7) / 8;
        for(i=long_num;i>0;i=i-1) begin
             local_in = in;
             tmp_long = local_in >> ((i-1)*64);
             for(j=0;j<64;j=j+1)
                 if (tmp_long[j] === 1'bx)
                     tmp_long[j] = 1'b0;
             if (i == long_num) begin
                 case(char_num)
                     1: $fwrite(fp,"%c",tmp_long[7:0]);
                     2: $fwrite(fp,"%c%c",tmp_long[15:8],tmp_long[7:0]);
                     3: $fwrite(fp,"%c%c%c",tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     4: $fwrite(fp,"%c%c%c%c",tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     5: $fwrite(fp,"%c%c%c%c%c",tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     6: $fwrite(fp,"%c%c%c%c%c%c",tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     7: $fwrite(fp,"%c%c%c%c%c%c%c",tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     8: $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
                     default: ;
                 endcase
             end
             else begin
                 $fwrite(fp,"%c%c%c%c%c%c%c%c",tmp_long[63:56],tmp_long[55:48],tmp_long[47:40],tmp_long[39:32],tmp_long[31:24],tmp_long[23:16],tmp_long[15:8],tmp_long[7:0]);
             end
        end
    end
endtask;
endmodule
