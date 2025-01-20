import torch

SHM_BLOCK_SIZE = 1024 * 16  # 假定 shm 的大小为 16k

# 标准的 Attention
def standard_softmax_attention(Q, K, V):
    """
    执行标准的PyTorch softmax和attention计算。
    """
    expected_softmax = torch.softmax(Q @ K.T, dim=1)
    expected_attention = expected_softmax @ V
    return expected_softmax, expected_attention

# falsh-attention-v2 的流程
def flash_attn_v2(Q, K, V):
    seq_length, q_head_dim = Q.shape[0], Q.shape[1]
    k_seq_length, k_head_dim = K.shape[0], K.shape[1]
    v_seq_length, v_head_dim = K.shape[0], K.shape[1]
    assert q_head_dim == k_head_dim
    assert k_seq_length == v_seq_length
    Br = min(int(SHM_BLOCK_SIZE / 4 / q_head_dim), q_head_dim)
    Bc = int(SHM_BLOCK_SIZE / 4 / q_head_dim)
    M = torch.zeros(seq_length, 1)
    O = torch.zeros(seq_length, v_head_dim)
    # output = []
    for i in range(0, seq_length, Br):
        Qi = Q[i:i+Br, :]
        # Mi = torch.zeros(Br, 1)
        Mi = M[i:i+Br, :]
        # Li = torch.ones(Br, 1)
        Li = torch.zeros(Br, 1) #自行修改
        oi = O[i:i+Br, :]

        for j in range(0, k_seq_length, Bc):
            Kj = K[j:j+Bc, :]
            Vj = V[j:j+Bc, :]
            Sij = Qi @ Kj.T
            
            mij_hat = torch.max(Sij, dim=1).values[:, None]
            # pij_hat = torch.exp(Sij - mij_hat)
            # lij_hat = torch.sum(pij_hat, dim=1)[:, None]
            
            # Mi_new = torch.max(torch.column_stack([Mi, torch.max(Sij, dim=1).values[:, None]]), dim=1).values[:, None]
            Mi_new = torch.max(torch.column_stack([Mi, mij_hat]), dim=1).values[:, None] 
            #当外循环i时，内循环j不断更新行块的m。Mi_new表示截止j块的最新max，Mi表示截止j-1块的max, Mi初始化为0
            Pij = torch.exp(Sij - Mi_new)

            # Li = torch.exp(Mi - Mi_new) * Li + torch.sum(Pij, dim=-1)[:, None]
            # Li_new表示截止j块的最新softmax分母，Li表示截止j-1块的softmax分母，Li初始化为1
            Li_new = torch.exp(Mi - Mi_new) * Li + torch.sum(Pij, dim=-1)[:, None]
            
            #oi初始化为0
            oi = oi * torch.exp(Mi - Mi_new) + Pij @ Vj
            # Mi = Mi_new
            # import ipdb; ipdb.set_trace()
            Mi = Mi_new
            Li = Li_new
            # print (i,j, seq_length, k_seq_length)
        
        # Oi = Oi / Li
        O[i:i+Br, :] = oi / Li
        # output.append(Oi)
    # res = torch.row_stack(output)
    # return res
    return O

def test_flash_v2():
    N, d = 1024, 128  # 更新N和d的值 (seqlen, head_dim)

    Q_mat = torch.rand((N, d))
    K_mat = torch.rand((N, d))
    V_mat = torch.rand((N, d))

    # 执行flash attention计算
    flash_attention_v2_output = flash_attn_v2(Q_mat, K_mat, V_mat)

    # 执行标准的PyTorch softmax和attention计算
    _, expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)
    print(flash_attention_v2_output)
    print(expected_attention)
    print ('max diff', (flash_attention_v2_output - expected_attention).max())
    # 断言flash attention计算的结果与标准计算结果是否接近
    assert torch.allclose(flash_attention_v2_output, expected_attention), "Error in flash attention calculation"


if __name__ == '__main__':
    test_flash_v2()