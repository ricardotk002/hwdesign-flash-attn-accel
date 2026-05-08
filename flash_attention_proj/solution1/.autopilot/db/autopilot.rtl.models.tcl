set SynModuleInfo {
  {SRCNAME flash_attention_hls_Pipeline_LOAD_Q_VITIS_LOOP_69_1 MODELNAME flash_attention_hls_Pipeline_LOAD_Q_VITIS_LOOP_69_1 RTLNAME flash_attention_hls_flash_attention_hls_Pipeline_LOAD_Q_VITIS_LOOP_69_1
    SUBMODULES {
      {MODELNAME flash_attention_hls_flow_control_loop_pipe_sequential_init RTLNAME flash_attention_hls_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME flash_attention_hls_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME flash_attention_hls_Pipeline_VITIS_LOOP_80_2 MODELNAME flash_attention_hls_Pipeline_VITIS_LOOP_80_2 RTLNAME flash_attention_hls_flash_attention_hls_Pipeline_VITIS_LOOP_80_2}
  {SRCNAME flash_attention_hls_Pipeline_LOAD_KV_VITIS_LOOP_93_3 MODELNAME flash_attention_hls_Pipeline_LOAD_KV_VITIS_LOOP_93_3 RTLNAME flash_attention_hls_flash_attention_hls_Pipeline_LOAD_KV_VITIS_LOOP_93_3}
  {SRCNAME flash_attention_hls_Pipeline_SCORE_LOOP_I_SCORE_LOOP_J MODELNAME flash_attention_hls_Pipeline_SCORE_LOOP_I_SCORE_LOOP_J RTLNAME flash_attention_hls_flash_attention_hls_Pipeline_SCORE_LOOP_I_SCORE_LOOP_J
    SUBMODULES {
      {MODELNAME flash_attention_hls_fadd_32ns_32ns_32_5_full_dsp_1 RTLNAME flash_attention_hls_fadd_32ns_32ns_32_5_full_dsp_1 BINDTYPE op TYPE fadd IMPL fulldsp LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_fmul_32ns_32ns_32_4_max_dsp_1 RTLNAME flash_attention_hls_fmul_32ns_32ns_32_4_max_dsp_1 BINDTYPE op TYPE fmul IMPL maxdsp LATENCY 3 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME flash_attention_hls_Pipeline_FIND_MAX MODELNAME flash_attention_hls_Pipeline_FIND_MAX RTLNAME flash_attention_hls_flash_attention_hls_Pipeline_FIND_MAX}
  {SRCNAME flash_attention_hls_Pipeline_VITIS_LOOP_160_4 MODELNAME flash_attention_hls_Pipeline_VITIS_LOOP_160_4 RTLNAME flash_attention_hls_flash_attention_hls_Pipeline_VITIS_LOOP_160_4}
  {SRCNAME flash_attention_hls_Pipeline_UPDATE_ACC MODELNAME flash_attention_hls_Pipeline_UPDATE_ACC RTLNAME flash_attention_hls_flash_attention_hls_Pipeline_UPDATE_ACC
    SUBMODULES {
      {MODELNAME flash_attention_hls_fdiv_32ns_32ns_32_16_no_dsp_1 RTLNAME flash_attention_hls_fdiv_32ns_32ns_32_16_no_dsp_1 BINDTYPE op TYPE fdiv IMPL fabric LATENCY 15 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME flash_attention_hls_Pipeline_STORE_O_VITIS_LOOP_188_5 MODELNAME flash_attention_hls_Pipeline_STORE_O_VITIS_LOOP_188_5 RTLNAME flash_attention_hls_flash_attention_hls_Pipeline_STORE_O_VITIS_LOOP_188_5
    SUBMODULES {
      {MODELNAME flash_attention_hls_sparsemux_9_2_32_1_1 RTLNAME flash_attention_hls_sparsemux_9_2_32_1_1 BINDTYPE op TYPE sparsemux IMPL auto}
    }
  }
  {SRCNAME flash_attention_hls MODELNAME flash_attention_hls RTLNAME flash_attention_hls IS_TOP 1
    SUBMODULES {
      {MODELNAME flash_attention_hls_faddfsub_32ns_32ns_32_5_full_dsp_1 RTLNAME flash_attention_hls_faddfsub_32ns_32ns_32_5_full_dsp_1 BINDTYPE op TYPE fsub IMPL fulldsp LATENCY 4 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_sitofp_32ns_32_6_no_dsp_1 RTLNAME flash_attention_hls_sitofp_32ns_32_6_no_dsp_1 BINDTYPE op TYPE sitofp IMPL auto LATENCY 5 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_fcmp_32ns_32ns_1_2_no_dsp_1 RTLNAME flash_attention_hls_fcmp_32ns_32ns_1_2_no_dsp_1 BINDTYPE op TYPE fcmp IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_fsqrt_32ns_32ns_32_16_no_dsp_1 RTLNAME flash_attention_hls_fsqrt_32ns_32ns_32_16_no_dsp_1 BINDTYPE op TYPE fsqrt IMPL fabric LATENCY 15 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_fexp_32ns_32ns_32_10_full_dsp_1 RTLNAME flash_attention_hls_fexp_32ns_32ns_32_10_full_dsp_1 BINDTYPE op TYPE fexp IMPL fulldsp LATENCY 9 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_mul_32ns_32ns_63_2_1 RTLNAME flash_attention_hls_mul_32ns_32ns_63_2_1 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_mul_32ns_32ns_64_2_1 RTLNAME flash_attention_hls_mul_32ns_32ns_64_2_1 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_Qbuf_RAM_1WNR_AUTO_1R1W RTLNAME flash_attention_hls_Qbuf_RAM_1WNR_AUTO_1R1W BINDTYPE storage TYPE ram_1wnr IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_Vbuf_RAM_AUTO_1R1W RTLNAME flash_attention_hls_Vbuf_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_acc_RAM_AUTO_1R1W RTLNAME flash_attention_hls_acc_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_m_RAM_AUTO_1R1W RTLNAME flash_attention_hls_m_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_weighted_RAM_AUTO_1R1W RTLNAME flash_attention_hls_weighted_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME flash_attention_hls_gmem0_m_axi RTLNAME flash_attention_hls_gmem0_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME flash_attention_hls_gmem1_m_axi RTLNAME flash_attention_hls_gmem1_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME flash_attention_hls_gmem2_m_axi RTLNAME flash_attention_hls_gmem2_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME flash_attention_hls_gmem3_m_axi RTLNAME flash_attention_hls_gmem3_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME flash_attention_hls_ctrl_s_axi RTLNAME flash_attention_hls_ctrl_s_axi BINDTYPE interface TYPE interface_s_axilite}
    }
  }
}
