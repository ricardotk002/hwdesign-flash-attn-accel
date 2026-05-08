# This script segment is generated automatically by AutoPilot

set name flash_attention_hls_fadd_32ns_32ns_32_5_full_dsp_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {fadd} IMPL {fulldsp} LATENCY 4 ALLOW_PRAGMA 1
}


set name flash_attention_hls_fmul_32ns_32ns_32_4_max_dsp_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {fmul} IMPL {maxdsp} LATENCY 3 ALLOW_PRAGMA 1
}


# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 102 \
    name Qbuf \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename Qbuf \
    op interface \
    ports { Qbuf_address0 { O 6 vector } Qbuf_ce0 { O 1 bit } Qbuf_q0 { I 32 vector } Qbuf_address1 { O 6 vector } Qbuf_ce1 { O 1 bit } Qbuf_q1 { I 32 vector } Qbuf_address2 { O 6 vector } Qbuf_ce2 { O 1 bit } Qbuf_q2 { I 32 vector } Qbuf_address3 { O 6 vector } Qbuf_ce3 { O 1 bit } Qbuf_q3 { I 32 vector } Qbuf_address4 { O 6 vector } Qbuf_ce4 { O 1 bit } Qbuf_q4 { I 32 vector } Qbuf_address5 { O 6 vector } Qbuf_ce5 { O 1 bit } Qbuf_q5 { I 32 vector } Qbuf_address6 { O 6 vector } Qbuf_ce6 { O 1 bit } Qbuf_q6 { I 32 vector } Qbuf_address7 { O 6 vector } Qbuf_ce7 { O 1 bit } Qbuf_q7 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'Qbuf'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 103 \
    name Qbuf_1 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename Qbuf_1 \
    op interface \
    ports { Qbuf_1_address0 { O 6 vector } Qbuf_1_ce0 { O 1 bit } Qbuf_1_q0 { I 32 vector } Qbuf_1_address1 { O 6 vector } Qbuf_1_ce1 { O 1 bit } Qbuf_1_q1 { I 32 vector } Qbuf_1_address2 { O 6 vector } Qbuf_1_ce2 { O 1 bit } Qbuf_1_q2 { I 32 vector } Qbuf_1_address3 { O 6 vector } Qbuf_1_ce3 { O 1 bit } Qbuf_1_q3 { I 32 vector } Qbuf_1_address4 { O 6 vector } Qbuf_1_ce4 { O 1 bit } Qbuf_1_q4 { I 32 vector } Qbuf_1_address5 { O 6 vector } Qbuf_1_ce5 { O 1 bit } Qbuf_1_q5 { I 32 vector } Qbuf_1_address6 { O 6 vector } Qbuf_1_ce6 { O 1 bit } Qbuf_1_q6 { I 32 vector } Qbuf_1_address7 { O 6 vector } Qbuf_1_ce7 { O 1 bit } Qbuf_1_q7 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'Qbuf_1'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 104 \
    name Qbuf_2 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename Qbuf_2 \
    op interface \
    ports { Qbuf_2_address0 { O 6 vector } Qbuf_2_ce0 { O 1 bit } Qbuf_2_q0 { I 32 vector } Qbuf_2_address1 { O 6 vector } Qbuf_2_ce1 { O 1 bit } Qbuf_2_q1 { I 32 vector } Qbuf_2_address2 { O 6 vector } Qbuf_2_ce2 { O 1 bit } Qbuf_2_q2 { I 32 vector } Qbuf_2_address3 { O 6 vector } Qbuf_2_ce3 { O 1 bit } Qbuf_2_q3 { I 32 vector } Qbuf_2_address4 { O 6 vector } Qbuf_2_ce4 { O 1 bit } Qbuf_2_q4 { I 32 vector } Qbuf_2_address5 { O 6 vector } Qbuf_2_ce5 { O 1 bit } Qbuf_2_q5 { I 32 vector } Qbuf_2_address6 { O 6 vector } Qbuf_2_ce6 { O 1 bit } Qbuf_2_q6 { I 32 vector } Qbuf_2_address7 { O 6 vector } Qbuf_2_ce7 { O 1 bit } Qbuf_2_q7 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'Qbuf_2'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 105 \
    name Qbuf_3 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename Qbuf_3 \
    op interface \
    ports { Qbuf_3_address0 { O 6 vector } Qbuf_3_ce0 { O 1 bit } Qbuf_3_q0 { I 32 vector } Qbuf_3_address1 { O 6 vector } Qbuf_3_ce1 { O 1 bit } Qbuf_3_q1 { I 32 vector } Qbuf_3_address2 { O 6 vector } Qbuf_3_ce2 { O 1 bit } Qbuf_3_q2 { I 32 vector } Qbuf_3_address3 { O 6 vector } Qbuf_3_ce3 { O 1 bit } Qbuf_3_q3 { I 32 vector } Qbuf_3_address4 { O 6 vector } Qbuf_3_ce4 { O 1 bit } Qbuf_3_q4 { I 32 vector } Qbuf_3_address5 { O 6 vector } Qbuf_3_ce5 { O 1 bit } Qbuf_3_q5 { I 32 vector } Qbuf_3_address6 { O 6 vector } Qbuf_3_ce6 { O 1 bit } Qbuf_3_q6 { I 32 vector } Qbuf_3_address7 { O 6 vector } Qbuf_3_ce7 { O 1 bit } Qbuf_3_q7 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'Qbuf_3'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 107 \
    name Kbuf \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename Kbuf \
    op interface \
    ports { Kbuf_address0 { O 6 vector } Kbuf_ce0 { O 1 bit } Kbuf_q0 { I 32 vector } Kbuf_address1 { O 6 vector } Kbuf_ce1 { O 1 bit } Kbuf_q1 { I 32 vector } Kbuf_address2 { O 6 vector } Kbuf_ce2 { O 1 bit } Kbuf_q2 { I 32 vector } Kbuf_address3 { O 6 vector } Kbuf_ce3 { O 1 bit } Kbuf_q3 { I 32 vector } Kbuf_address4 { O 6 vector } Kbuf_ce4 { O 1 bit } Kbuf_q4 { I 32 vector } Kbuf_address5 { O 6 vector } Kbuf_ce5 { O 1 bit } Kbuf_q5 { I 32 vector } Kbuf_address6 { O 6 vector } Kbuf_ce6 { O 1 bit } Kbuf_q6 { I 32 vector } Kbuf_address7 { O 6 vector } Kbuf_ce7 { O 1 bit } Kbuf_q7 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'Kbuf'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 108 \
    name Kbuf_1 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename Kbuf_1 \
    op interface \
    ports { Kbuf_1_address0 { O 6 vector } Kbuf_1_ce0 { O 1 bit } Kbuf_1_q0 { I 32 vector } Kbuf_1_address1 { O 6 vector } Kbuf_1_ce1 { O 1 bit } Kbuf_1_q1 { I 32 vector } Kbuf_1_address2 { O 6 vector } Kbuf_1_ce2 { O 1 bit } Kbuf_1_q2 { I 32 vector } Kbuf_1_address3 { O 6 vector } Kbuf_1_ce3 { O 1 bit } Kbuf_1_q3 { I 32 vector } Kbuf_1_address4 { O 6 vector } Kbuf_1_ce4 { O 1 bit } Kbuf_1_q4 { I 32 vector } Kbuf_1_address5 { O 6 vector } Kbuf_1_ce5 { O 1 bit } Kbuf_1_q5 { I 32 vector } Kbuf_1_address6 { O 6 vector } Kbuf_1_ce6 { O 1 bit } Kbuf_1_q6 { I 32 vector } Kbuf_1_address7 { O 6 vector } Kbuf_1_ce7 { O 1 bit } Kbuf_1_q7 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'Kbuf_1'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 109 \
    name Kbuf_2 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename Kbuf_2 \
    op interface \
    ports { Kbuf_2_address0 { O 6 vector } Kbuf_2_ce0 { O 1 bit } Kbuf_2_q0 { I 32 vector } Kbuf_2_address1 { O 6 vector } Kbuf_2_ce1 { O 1 bit } Kbuf_2_q1 { I 32 vector } Kbuf_2_address2 { O 6 vector } Kbuf_2_ce2 { O 1 bit } Kbuf_2_q2 { I 32 vector } Kbuf_2_address3 { O 6 vector } Kbuf_2_ce3 { O 1 bit } Kbuf_2_q3 { I 32 vector } Kbuf_2_address4 { O 6 vector } Kbuf_2_ce4 { O 1 bit } Kbuf_2_q4 { I 32 vector } Kbuf_2_address5 { O 6 vector } Kbuf_2_ce5 { O 1 bit } Kbuf_2_q5 { I 32 vector } Kbuf_2_address6 { O 6 vector } Kbuf_2_ce6 { O 1 bit } Kbuf_2_q6 { I 32 vector } Kbuf_2_address7 { O 6 vector } Kbuf_2_ce7 { O 1 bit } Kbuf_2_q7 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'Kbuf_2'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 110 \
    name Kbuf_3 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename Kbuf_3 \
    op interface \
    ports { Kbuf_3_address0 { O 6 vector } Kbuf_3_ce0 { O 1 bit } Kbuf_3_q0 { I 32 vector } Kbuf_3_address1 { O 6 vector } Kbuf_3_ce1 { O 1 bit } Kbuf_3_q1 { I 32 vector } Kbuf_3_address2 { O 6 vector } Kbuf_3_ce2 { O 1 bit } Kbuf_3_q2 { I 32 vector } Kbuf_3_address3 { O 6 vector } Kbuf_3_ce3 { O 1 bit } Kbuf_3_q3 { I 32 vector } Kbuf_3_address4 { O 6 vector } Kbuf_3_ce4 { O 1 bit } Kbuf_3_q4 { I 32 vector } Kbuf_3_address5 { O 6 vector } Kbuf_3_ce5 { O 1 bit } Kbuf_3_q5 { I 32 vector } Kbuf_3_address6 { O 6 vector } Kbuf_3_ce6 { O 1 bit } Kbuf_3_q6 { I 32 vector } Kbuf_3_address7 { O 6 vector } Kbuf_3_ce7 { O 1 bit } Kbuf_3_q7 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'Kbuf_3'"
}
}


# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 111 \
    name scores \
    reset_level 1 \
    sync_rst true \
    dir O \
    corename scores \
    op interface \
    ports { scores_address0 { O 6 vector } scores_ce0 { O 1 bit } scores_we0 { O 1 bit } scores_d0 { O 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'scores'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 100 \
    name k_lim \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_k_lim \
    op interface \
    ports { k_lim { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 101 \
    name mul_ln104 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mul_ln104 \
    op interface \
    ports { mul_ln104 { I 63 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 106 \
    name q0 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_q0 \
    op interface \
    ports { q0 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 112 \
    name cmp715 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp715 \
    op interface \
    ports { cmp715 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 113 \
    name icmp \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_icmp \
    op interface \
    ports { icmp { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 114 \
    name cmp102_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_2 \
    op interface \
    ports { cmp102_2 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 115 \
    name icmp49 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_icmp49 \
    op interface \
    ports { icmp49 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 116 \
    name cmp102_4 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_4 \
    op interface \
    ports { cmp102_4 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 117 \
    name cmp102_5 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_5 \
    op interface \
    ports { cmp102_5 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 118 \
    name cmp102_6 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_6 \
    op interface \
    ports { cmp102_6 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 119 \
    name icmp52 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_icmp52 \
    op interface \
    ports { icmp52 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 120 \
    name cmp102_8 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_8 \
    op interface \
    ports { cmp102_8 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 121 \
    name cmp102_9 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_9 \
    op interface \
    ports { cmp102_9 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 122 \
    name cmp102_10 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_10 \
    op interface \
    ports { cmp102_10 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 123 \
    name cmp102_11 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_11 \
    op interface \
    ports { cmp102_11 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 124 \
    name cmp102_12 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_12 \
    op interface \
    ports { cmp102_12 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 125 \
    name cmp102_13 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_13 \
    op interface \
    ports { cmp102_13 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 126 \
    name cmp102_14 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_14 \
    op interface \
    ports { cmp102_14 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 127 \
    name icmp55 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_icmp55 \
    op interface \
    ports { icmp55 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 128 \
    name cmp102_16 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_16 \
    op interface \
    ports { cmp102_16 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 129 \
    name cmp102_17 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_17 \
    op interface \
    ports { cmp102_17 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 130 \
    name cmp102_18 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_18 \
    op interface \
    ports { cmp102_18 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 131 \
    name cmp102_19 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_19 \
    op interface \
    ports { cmp102_19 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 132 \
    name cmp102_20 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_20 \
    op interface \
    ports { cmp102_20 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 133 \
    name cmp102_21 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_21 \
    op interface \
    ports { cmp102_21 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 134 \
    name cmp102_22 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_22 \
    op interface \
    ports { cmp102_22 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 135 \
    name cmp102_23 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_23 \
    op interface \
    ports { cmp102_23 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 136 \
    name cmp102_24 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_24 \
    op interface \
    ports { cmp102_24 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 137 \
    name cmp102_25 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_25 \
    op interface \
    ports { cmp102_25 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 138 \
    name cmp102_26 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_26 \
    op interface \
    ports { cmp102_26 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 139 \
    name cmp102_27 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_27 \
    op interface \
    ports { cmp102_27 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 140 \
    name cmp102_28 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_28 \
    op interface \
    ports { cmp102_28 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 141 \
    name cmp102_29 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_29 \
    op interface \
    ports { cmp102_29 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 142 \
    name cmp102_30 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_cmp102_30 \
    op interface \
    ports { cmp102_30 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 143 \
    name icmp58 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_icmp58 \
    op interface \
    ports { icmp58 { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 144 \
    name inv_sqrt_d \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inv_sqrt_d \
    op interface \
    ports { inv_sqrt_d { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 145 \
    name k0 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_k0 \
    op interface \
    ports { k0 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 146 \
    name tobool \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_tobool \
    op interface \
    ports { tobool { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


# flow_control definition:
set InstName flash_attention_hls_flow_control_loop_pipe_sequential_init_U
set CompName flash_attention_hls_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix flash_attention_hls_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


