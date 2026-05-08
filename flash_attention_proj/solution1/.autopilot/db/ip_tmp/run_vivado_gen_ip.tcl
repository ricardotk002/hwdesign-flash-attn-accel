create_project prj -part xc7z020-clg400-1 -force
set_property target_language verilog [current_project]
set vivado_ver [version -short]
set COE_DIR "../../syn/verilog"
source "/home/aa13231/Downloads/code/flash_attention_proj/solution1/syn/verilog/flash_attention_hls_fdiv_32ns_32ns_32_16_no_dsp_1_ip.tcl"
source "/home/aa13231/Downloads/code/flash_attention_proj/solution1/syn/verilog/flash_attention_hls_fcmp_32ns_32ns_1_2_no_dsp_1_ip.tcl"
source "/home/aa13231/Downloads/code/flash_attention_proj/solution1/syn/verilog/flash_attention_hls_fadd_32ns_32ns_32_5_full_dsp_1_ip.tcl"
source "/home/aa13231/Downloads/code/flash_attention_proj/solution1/syn/verilog/flash_attention_hls_sitofp_32ns_32_6_no_dsp_1_ip.tcl"
source "/home/aa13231/Downloads/code/flash_attention_proj/solution1/syn/verilog/flash_attention_hls_fsqrt_32ns_32ns_32_16_no_dsp_1_ip.tcl"
source "/home/aa13231/Downloads/code/flash_attention_proj/solution1/syn/verilog/flash_attention_hls_fexp_32ns_32ns_32_10_full_dsp_1_ip.tcl"
source "/home/aa13231/Downloads/code/flash_attention_proj/solution1/syn/verilog/flash_attention_hls_fmul_32ns_32ns_32_4_max_dsp_1_ip.tcl"
source "/home/aa13231/Downloads/code/flash_attention_proj/solution1/syn/verilog/flash_attention_hls_faddfsub_32ns_32ns_32_5_full_dsp_1_ip.tcl"
