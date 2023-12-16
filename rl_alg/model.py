import torch
import torch.nn as nn
import math


class attention_prop(torch.nn.Module):

    def __init__(self, q_size, k_size, feat_size):
        super(attention_prop, self).__init__()
        self.feat_size = feat_size
        self.q1 = nn.Sequential(
            nn.Linear(q_size, feat_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size, feat_size, bias=True),
            nn.LeakyReLU(),
        )
        self.bnq = nn.BatchNorm1d(feat_size)
        self.k1 = nn.Sequential(
            nn.Linear(k_size, feat_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size, feat_size, bias=True),
            nn.LeakyReLU(),
        )
        self.bnk = nn.BatchNorm1d(feat_size)
        self.v1 = nn.Sequential(
            nn.Linear(k_size, feat_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(feat_size, feat_size, bias=True),
            nn.LeakyReLU(),
        )
        self.bnv = nn.BatchNorm1d(feat_size)
        self.sfm = nn.Softmax(dim=1)

    def sparse_dense_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def forward(self, A, q, k):
        qe1 = self.bnq(self.q1(q))
        ke1 = self.bnk(self.k1(k)).T
        sqdk = torch.sqrt(torch.tensor(self.feat_size)).to(q.device)
        prod = qe1.matmul(ke1) / sqdk
        A = A.to_dense()
        masked_prod = torch.mul(A, prod)
        masked_prod = self.sfm(masked_prod)

        ve1 = self.bnv(self.v1(k))
        vres = masked_prod.matmul(ve1)

        return vres


class base_emb(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super(base_emb, self).__init__()
        self.emb = nn.Sequential(
            nn.Linear(in_size, out_size, bias=True),
            nn.Linear(out_size, out_size, bias=True),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.emb(x)


class relu_layer(torch.nn.Module):

    def __init__(self, in_size, out_size, norm=False):
        super(relu_layer, self).__init__()
        self.mlp = nn.Linear(in_size, out_size, bias=True)
        self.act = nn.LeakyReLU()
        self.norm = None
        self.in_size = in_size
        if norm:
            self.norm = nn.LayerNorm(out_size)
        self.dout = nn.Dropout(p=0.1)

    def forward(self, x, training=False):
        # print(self.in_size,x.shape)
        x = self.mlp(x)
        if training:
            x = self.dout(x)
        res = self.act(x)
        if self.norm is not None:
            res = self.norm(res)
        return res


class att_block(torch.nn.Module):

    def __init__(self, v_size, c_size, first_emb_size, multi_head=4, is_first=False):
        super(att_block, self).__init__()
        self.multi_head = multi_head

        if is_first:
            self.embv = base_emb(v_size, first_emb_size)
            self.embc = base_emb(c_size, first_emb_size)
        else:
            self.embv = base_emb(v_size * multi_head, first_emb_size)
            self.embc = base_emb(c_size * multi_head, first_emb_size)

        self.att1 = attention_prop(first_emb_size * multi_head, first_emb_size * multi_head,
                                   first_emb_size * multi_head)
        self.att_for1 = relu_layer(first_emb_size * multi_head, first_emb_size * multi_head, norm=False)
        self.norm11 = nn.LayerNorm(first_emb_size * multi_head)
        self.norm12 = nn.LayerNorm(first_emb_size * multi_head)

        self.att2 = attention_prop(first_emb_size * multi_head, first_emb_size * multi_head,
                                   first_emb_size * multi_head)
        self.att_for2 = relu_layer(first_emb_size * multi_head, first_emb_size * multi_head, norm=False)
        self.norm21 = nn.LayerNorm(first_emb_size * multi_head)
        self.norm22 = nn.LayerNorm(first_emb_size * multi_head)

    def forward(self, A, AT, v, c, training=False):
        # an attention block
        c1 = self.embc(c)
        v1 = self.embv(v)

        v1t = v1.repeat(1, self.multi_head)
        c1t = c1.repeat(1, self.multi_head)

        # V to C
        c2 = self.norm11(self.att1(A, c1t, v1t) + c1t)
        c2 = self.norm12(c2 + self.att_for1(c2))

        # C to V
        v2 = self.norm21(self.att2(AT, v1t, c2) + v1t)
        v2 = self.norm22(v2 + self.att_for1(v2))
        # end of attention block
        return c2, v2


class var_sorter(torch.nn.Module):

    def __init__(self, v_size, c_size, sample_sizes, multi_head=4, natt=2):
        super(var_sorter, self).__init__()
        first_emb_size = sample_sizes[0]
        self.embv = base_emb(v_size, first_emb_size)  # 嵌入层（Embedding Layer）是一种用于将离散的整数或类别型数据映射到连续的低维向量空间的神经网络层。
        self.embc = base_emb(c_size, first_emb_size)
        self.multi_head = multi_head

        self.vsz = v_size

        self.att_lays = nn.ModuleList()
        self.att_lays.append(
            att_block(first_emb_size, first_emb_size, first_emb_size, multi_head=multi_head, is_first=True))
        for i in range(natt):
            self.att_lays.append(att_block(first_emb_size, first_emb_size, first_emb_size, multi_head=multi_head))

        self.final = nn.ModuleList()
        self.final.append(relu_layer(first_emb_size * multi_head, first_emb_size))
        for i in range(len(sample_sizes) - 1):
            self.final.append(relu_layer(sample_sizes[i], sample_sizes[i + 1]))
        self.final.append(relu_layer(sample_sizes[-1], 1))

    def forward(self, A, v, c, training=False, bvs=None):

        v = self.embv(v)
        c = self.embc(c)

        AT = torch.transpose(A, 0, 1)

        for indx, m in enumerate(self.att_lays):
            c, v = m(A, AT, v, c)

        # v=torch.transpose(v,1,0)
        v = torch.unsqueeze(v, 0)

        for indx, m in enumerate(self.final):
            # print(v,v.shape)
            v = m(v)

        v = torch.squeeze(v, 0)
        v = torch.squeeze(v, -1)
        # print(v,v.shape)

        return v
