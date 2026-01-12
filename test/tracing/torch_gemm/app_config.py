kernel_1 = {

	"kernel_name"			: "_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c104HalfEEESt5arrayIPcLm1EEEEviT0_T1_",
	"num_registers"			: 16,
	"shared_mem_bytes"		: 0,
	"grid_size"			: 7,
	"block_size"			: 128,
	"cuda_stream_id"		: 0
}

kernel_2 = {

	"kernel_name"			: "_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_11FillFunctorIN3c104HalfEEESt5arrayIPcLm1EEEEviT0_T1_",
	"num_registers"			: 16,
	"shared_mem_bytes"		: 0,
	"grid_size"			: 3584,
	"block_size"			: 128,
	"cuda_stream_id"		: 0
}

kernel_3 = {

	"kernel_name"			: "_ZN7cutlass7Kernel2I61cutlass_80_wmma_tensorop_s161616gemm_f16_16x16_64x2_nn_align8EEvNT_6ParamsE",
	"num_registers"			: 80,
	"shared_mem_bytes"		: 10496,
	"grid_size"			: 896,
	"block_size"			: 32,
	"cuda_stream_id"		: 0
}

kernel_4 = {

	"kernel_name"			: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Eif6__halffLb0ES1_S1_S1_Lb1ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT7_PT6_PT3_PKS3_SG_PKT8_S7_PSH_PvlPS3_PiSM_SG_SG_SG_SG_",
	"num_registers"			: 44,
	"shared_mem_bytes"		: 0,
	"grid_size"			: 16,
	"block_size"			: 512,
	"cuda_stream_id"		: 0
}

app_kernels_id = [1, 2, 3, 4]