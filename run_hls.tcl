# ============================================================
# Vitis HLS Tcl script for FlashAttention
# ============================================================
# Usage from a shell:
#   vitis_hls -f run_hls.tcl
#
# Or from the Vitis HLS Tcl console (in the GUI):
#   source run_hls.tcl
#
# To run only specific stages, comment out the ones you don't want
# at the bottom of this file.
# ============================================================

# ---- Project / solution ----
open_project flash_attention_proj
set_top flash_attention_hls

# ---- Source files (DUT) ----
add_files flash_attention.cpp

# ---- Testbench files ----
# The testbench reads a CSV from ./test_outputs relative to its own
# directory, so we add the CSV as a TB file too — Vitis copies it into
# the csim working directory.
add_files -tb tb_flash_attention.cpp
add_files -tb test_outputs/tv_attention_python.csv

# ---- Solution / target device ----
open_solution "solution1" -flow_target vivado

# PYNQ-Z2 board uses the Zynq-7020:  xc7z020clg400-1
# (Change to your part if you target something else.)
set_part {xc7z020clg400-1}

# Target clock (PYNQ-Z2 is typically run at 100 MHz on the PS)
create_clock -period 10 -name default

# ---- Stages ----
# Comment any stage out if you only want to re-run later ones.
puts ">>> C SIMULATION"
csim_design

puts ">>> C SYNTHESIS"
csynth_design

puts ">>> RTL CO-SIMULATION"
# Pass "rtl" to the testbench so it writes to tv_attention_rtl.csv
cosim_design -argv "rtl"

puts ">>> EXPORT IP (Vivado IP packager)"
export_design -format ip_catalog -rtl verilog

exit
