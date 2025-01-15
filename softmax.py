import torch


import triton
import triton.language as tl
from triton.runtime import driver

# import os
# os.environ["TRITON_INTERPRET"] = "1"


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    使用原生 PyTorch 计算 X 的逐行 softmax


    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    我们减去最大元素以避免溢出。Softmax 对于这种偏移是不变的。
    """
    # read  MN elements ; write M  elements
    # 读取 MN 个元素；写入 M 个元素
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    # 读取 MN + M 个元素；写入 MN 个元素
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    # 读取 MN 个元素；写入 MN 个元素
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    # 读取 MN 个元素；写入 M 个元素
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    # 读取 MN + M 个元素；写入 MN 个元素
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    # 总计：读取 5MN + 2M 个元素；写入 3MN + 2M 个元素
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    # 程序起始行
    row_start = tl.program_id(0) #[0, 1, 2...863]
    row_step = tl.num_programs(0) # 每个线程之间的步长，即总线程块数，864
    # tl.device_print("row_start", row_start)
    # tl.device_print("row_step", row_step)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        # 步长表示我们需要对指针增加多少以推进 1 行
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # 块大小是大于 n_cols 的下一个二的幂，因此我们可以适配
        # row in a single block
        # 单个块中的行
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        # 将行加载到 SRAM 中，使用掩码，因为 BLOCK_SIZE 可能大于 n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        # 为了数值稳定性而减去最大值
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        # 请注意，Triton 中的指数运算速度很快，但是是近似的（例如，类似于 CUDA 中的 __expf）。
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        # 将输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape


    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    # 每次循环迭代的块大小是大于 `x` 列数的最小二的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)


    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # 另一个技巧是通过增加每行分配的线程数来要求编译器使用更多的线程块 (`num_warps`)
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    # 将在下一个教程中看到如何以更自然的方式自动调整此值，以免自己进行手动启发式处理。
    num_warps = 8


    # Number of software piepling stages.
    # 软件流水线阶段的数量
    num_stages = 4 if SIZE_SMEM > 200000 else 2


    # Allocate output
    # 分配输出空间
    y = torch.empty_like(x)


    # pre-compile kernel to get register usage and compute thread occupancy.
    # 预编译内核以获取寄存器使用情况并计算线程占用情况。
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs #看起来这个kernel更像是cuda里面的thread的概念, n_regs对应每个thread的寄存器数
        size_smem = kernel.metadata.shared #size_smem对应每个block的共享内存大小
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps) # 每个sm 65536 // (28 * 32 * 8) = 9/8
        occupancy = min(occupancy, SIZE_SMEM // size_smem) # 每个sm 166912 / 4128 = 40, 根据硬件限制，计算每个sm的block数
        num_programs = NUM_SM * occupancy #类似cuda计算grid的划分，即block的shape, 108 * 9/8 = 972/864，occupancy对应每个sm的block数
        kernels[BLOCK_SIZE] = (kernel, num_programs)


    num_programs = min(num_programs, n_rows)
    print (f"num_programs: {num_programs}")


    # Create a number of persistent programs.
    # 创建一些持久化程序。
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0), #781
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name` `x_name` 的不同可能值
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot 参数名，其值对应于图表中不同线条
        line_vals=['triton', 'torch'],  # possible values for `line_arg`` `line_arg` 的可能值
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # line styles 线条的样式
        ylabel="GB/s",  # label name for the y-axis y 轴的标签名称
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot. 图表的名称，也用作保存图表的文件名
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name` `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == '__main__':
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    print(f"Device {device}: {properties}")
    NUM_SM = properties["multiprocessor_count"] # 单卡gpu的sm数量，108，a100为例
    NUM_REGS = properties["max_num_regs"] # 每个sm最大寄存器数量，65536=16384*4, 32bit寄存器
    SIZE_SMEM = properties["max_shared_mem"] # 每个sm的共享内存大小, 166912字节，约163kb。此外每个sm还有192kb的L1缓存？
    WARP_SIZE = properties["warpSize"] #warp大小，32 threads
    target = triton.runtime.driver.active.get_current_target()
    kernels = {}

    torch.manual_seed(0)
    x = torch.randn(1823, 1781, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(y_torch - y_triton))}')

    benchmark.run(print_data=True, show_plots=True, save_path='.')