import torch


import triton
import triton.language as tl

# import os
# os.environ["TRITON_INTERPRET"] = "1"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]




def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=0),
    ]




def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()




# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
# `triton.jit` 函数可以通过使用 `triton.autotune` 装饰器进行自动调优，该装饰器接受以下内容：
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - 一组 `triton.Config` 对象的列表，这些对象定义了不同的元参数配置（例如 `BLOCK_SIZE_M`）和编译选项（例如 `num_warps`）以供尝试。
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
# - 一个自动调优的 key，其值的变化将触发对所有提供的配置进行评估。


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        # 矩阵指针
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        # 矩阵维度
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        # 这些步幅变量表示在特定维度移动 1 个元素时，`ptr` 应该增加多少。例如，`stride_am` 指示了为了访问下一行的元素（假设 `A` 有 `M` 行），需要增加多少 `a_ptr`。
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        # 元参数
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    """计算矩阵乘法 C = A x B 的核心算法。
    其中，A 的形状为 (M, K)，B 的形状为 (K, N)，C 的形状为 (M, N)。
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # 将程序 ID `pid` 映射到它应计算的 C 块。
    # This is done in a grouped ordering to promote L2 data reuse.
    # 这是按组顺序进行的，以促进 L2 数据重用。
    # See above `L2 Cache Optimizations` section for details.
    # 详细信息请参见上述的 `L2 缓存优化` 部分。

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    #在C矩阵的内部，按照列主序存储，所以我们需要按照列主序计算pid_m和pid_n
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m


    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # 创建 A 和 B 第一个块的指针
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # 在沿着 K 方向移动时，我们将推进这个指针并累加
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `a_ptrs` 是一个 [BLOCK_SIZE_M, BLOCK_SIZE_K] 大小的指针块
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # `b_ptrs` 是一个 [BLOCK_SIZE_K, BLOCK_SIZE_N] 大小的指针块


    # See above `Pointer Arithmetic` section for details
    # 详细信息请参见上述的 `指针算术` 部分。
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)


    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # 迭代计算 C 矩阵的一个块。
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # 我们累加到一个 `[BLOCK_SIZE_M, BLOCK_SIZE_N]` 大小的 fp32 值块，以提高精度。
    # `accumulator` will be converted back to fp16 after the loop.
    # `accumulator` 在循环结束后将转换回 fp16。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # 加载 A 和 B 的下一个块，通过检查 K 维度生成一个掩码。
        # If it is out of bounds, set it to 0.
        # 如果超出边界设为 0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        # 通过着 K 维度进行累加。
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        # 指针前进到下一个 K 块。
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    # 在累加器仍然是 FP32 的情况下，您可以在这里融合任意激活函数！
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)


    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    # 写回带有掩码的输出矩阵 C 的块。
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)




# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
# 我们可以通过在 `matmul_kernel` 中将 `leaky_relu` 作为 `ACTIVATION` 元参数来融合 `leaky_relu`。
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, activation=""):
    # Check constraints.
    # 检查约束
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    # 分配输出
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    # 1 维启动核心，其中每个块都有自己的程序。
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    # Bigger tolerance for AMD MI200 devices.
    # 对于 AMD MI200 设备，使用更大的容差。
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    # MI200 设备使用降低精度的 FP16 和 BF16，并将输入和输出的非规格化值清零。详细信息在以下链接：https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol = 1e-2 if is_hip_mi200() else 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


    TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
    if TORCH_HAS_FP8 and is_cuda():
        torch.manual_seed(0)
        a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
        b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
        a = a.to(torch.float8_e5m2)
        # pre-transpose b for efficiency.
        # 提前转置 b 提高效率
        b = b.T
        b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        print(f"triton_output_with_fp8_inputs={triton_output}")
        print(f"torch_output_with_fp8_inputs={torch_output}")
        if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")