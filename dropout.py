import tabulate
import torch


import triton
import triton.language as tl


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    # 计算由此实例处理的元素的内存偏移量
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    # 从 x 读取数据
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    # 随机修剪
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # tl.device_print("x_keep", x_keep)
    # tl.device_print("random", random)
    # write-back
    # 写回
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


@triton.jit
def _sparse_jl_transform(x_ptr, output_ptr, n_rows, n_cols, seed_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    row_id = offsets // n_cols
    col_id = offsets % n_cols
    mask = offsets < n_rows * n_cols

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Generate sparse projection matrix
    seed = tl.load(seed_ptr + row_id, mask=row_id < n_rows)
    random = tl.rand(seed, col_id)
    projection = tl.where(random < 1/3, -1.0, tl.where(random < 2/3, 1.0, 0.0))

    # Apply the projection
    projected = x * projection

    # Write-back
    tl.store(output_ptr + offsets, projected, mask=mask)

def sparse_jl_transform(x, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    seed_tensor = torch.tensor(seed, device=x.device, dtype=torch.int32).expand(n_rows)
    _sparse_jl_transform[grid](x, output, n_rows, n_cols, seed_tensor, BLOCK_SIZE=1024)
    return output

if __name__ == "__main__":
    x = torch.randn(size=(10, )).cuda()
    # Compare this to the baseline - dropout mask is never instantiated!
    # 与基线相比 - dropout 掩码从未被实例化！
    output = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=512)


    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))

    # 实现稀疏 Johnson-Lindenstrauss 变换的内核，每次使用种子动态生成投影矩阵。
    # Example usage
    x = torch.randn(size=(10, 10)).cuda()
    seed = [123 + i for i in range(x.shape[0])]
    output = sparse_jl_transform(x, seed)

    print("Input:\n", x)
    print("Output:\n", output)