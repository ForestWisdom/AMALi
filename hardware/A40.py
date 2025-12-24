# hardware/A40.py
# NVIDIA A40 (GA102, compute capability 8.6)

uarch = {
    "gpu_name"           : "A40",

    # 这里要和 hardware/ISA 目录下的文件名一致：
    # 你现在 A100 用的是 Tesla_Ampere，所以 A40 也可以继续用 Tesla_Ampere
    "gpu_arch"           : "Tesla_Ampere",

    # A40 = sm_86
    "compute_capabilty"  : 86,

    # 核心频率：建议你用 nvidia-smi 实机读出来填（见下方第4节）
    # 先给一个占位值，不填会报错
    "clockspeed"         : 1305 * 10**6,

    "num_sub_cores"      : 4,
    "num_L1_cache_banks" : 4,

    # GA102 full GPU 84 SM（A40 数据表/GA102 白皮书都能对上）
    "num_SMs"            : 84,

    # GA10x：每 SM 128 FP32 CUDA cores；INT32 仍是 64/SM（常见建模写法）
    "num_INT_units_per_SM" : 64,
    "num_SP_units_per_SM"  : 128,

    # GA102 FP64 单元非常少：白皮书提到 full GA102 有 168 FP64 units (=2 per SM)
    "num_DP_units_per_SM"  : 2,

    "num_SF_units_per_SM"  : 16,
    "num_TC_units_per_SM"  : 4,

    # LD/ST 单元：如果你不确定，先沿用 A100 的 16（后续可用微基准校准）
    "num_LDS_units_per_SM" : 16,

    # GA102 每 SM 4 个 texture units
    "num_TEX_units_per_SM" : 4,

    "num_BRA_units_per_SM" : 4,
    "num_warp_schedulers_per_SM": 4,

    "l1_cache_bypassed" : False,

    # GA10x: 每 SM L1/Shared 总计 128KB，可按 workload 配置
    # AMALi 的写法是把 “total_l1_cache_size” 设成这个值，然后根据 shared 配置扣减得到 L1
    "l1_cache_size"     : 128 * 1024,

    "shared_mem_size"   : 0,

    # GA10x compute mode 的配置组合（白皮书给的列表）
    # 注意：AMALi 的 update_shared_mem 是找 “第一个 > shared_mem_bytes 的档位”
    # 所以这里放的是“档位上限”，并且按从小到大排序
    "shared_mem_config_list": [
        8 * 1024,
        16 * 1024,
        32 * 1024,
        48 * 1024,   # 可选：很多 kernel 会用到 48KB
        64 * 1024,
        100 * 1024,
    ],

    "l1_cache_line_size"      : 32,
    "l1_cache_associativity"  : 64,

    # GA102 full L2 = 6144 KB
    "l2_cache_size"           : 6 * 1024 * 1024,
    "l2_cache_line_size"      : 32,
    "l2_cache_associativity"  : 16,

    # L1/Shared 带宽：白皮书提到 GA10x shared 带宽 128 bytes/clock/SM（可先用这个量级）
    "l1_cache_bandwidth"      : 128.0,   # byte/clk/SM (先给近似值)

    # GA102：12 个 32-bit memory controllers（= 384-bit）
    "num_l2_partitions" : 64,
    "num_dram_channels" : 96,

    # A40 数据表：memory bandwidth 696 GB/s
    "dram_th_bandwidth" : 696 * 10**9,

    # dram clockspeed：建议你用 deviceQuery / nvidia-smi 读出来填（见下方）
    "dram_clockspeed"   : 7251 * 10**6,

    # NoC：你现在 A100/H100 都用 1200 GB/s，这里先保持一致（后续可校准）
    "noc_th_bandwidth"  : 1200 * 10**9,

    "warp_scheduling"   : "GTO",
}
