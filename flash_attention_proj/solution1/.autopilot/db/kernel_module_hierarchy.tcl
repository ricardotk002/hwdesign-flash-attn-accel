set ModuleHierarchy {[{
"Name" : "flash_attention_hls","ID" : "0","Type" : "sequential",
"SubLoops" : [
	{"Name" : "Q_TILE_LOOP","ID" : "1","Type" : "no",
	"SubInsts" : [
	{"Name" : "grp_flash_attention_hls_Pipeline_LOAD_Q_VITIS_LOOP_69_1_fu_777","ID" : "2","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "LOAD_Q_VITIS_LOOP_69_1","ID" : "3","Type" : "pipeline"},]},
	{"Name" : "grp_flash_attention_hls_Pipeline_STORE_O_VITIS_LOOP_188_5_fu_915","ID" : "4","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "STORE_O_VITIS_LOOP_188_5","ID" : "5","Type" : "pipeline"},]},],
	"SubLoops" : [
	{"Name" : "INIT_STATE","ID" : "6","Type" : "no",
		"SubInsts" : [
		{"Name" : "grp_flash_attention_hls_Pipeline_VITIS_LOOP_80_2_fu_792","ID" : "7","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "VITIS_LOOP_80_2","ID" : "8","Type" : "pipeline"},]},]},
	{"Name" : "K_TILE_LOOP","ID" : "9","Type" : "no",
		"SubInsts" : [
		{"Name" : "grp_flash_attention_hls_Pipeline_LOAD_KV_VITIS_LOOP_93_3_fu_802","ID" : "10","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "LOAD_KV_VITIS_LOOP_93_3","ID" : "11","Type" : "pipeline"},]},
		{"Name" : "grp_flash_attention_hls_Pipeline_SCORE_LOOP_I_SCORE_LOOP_J_fu_824","ID" : "12","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "SCORE_LOOP_I_SCORE_LOOP_J","ID" : "13","Type" : "pipeline"},]},],
		"SubLoops" : [
		{"Name" : "UPDATE_LOOP_I","ID" : "14","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_flash_attention_hls_Pipeline_FIND_MAX_fu_876","ID" : "15","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "FIND_MAX","ID" : "16","Type" : "pipeline"},]},
			{"Name" : "grp_flash_attention_hls_Pipeline_UPDATE_ACC_fu_899","ID" : "17","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "UPDATE_ACC","ID" : "18","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "EXP_ACCUM","ID" : "19","Type" : "no",
				"SubInsts" : [
				{"Name" : "grp_flash_attention_hls_Pipeline_VITIS_LOOP_160_4_fu_884","ID" : "20","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "VITIS_LOOP_160_4","ID" : "21","Type" : "pipeline"},]},]},]},]},]},]
}]}