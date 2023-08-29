import torch.nn as nn
import torch
from loguru import logger

d_model = 1024  # 字 Embedding 的维度
d_ff = 256  # 前向传播隐藏层维度
n_heads = 8


class SelfAttention_1(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(SelfAttention_1, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        # self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to('cuda')

    def forward(self, query, key, value, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        # logger.error(Q.device)
        # logger.error(K.device)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = self.fc(x)
        return x


class SelfAttention_2(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(SelfAttention_2, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        # self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to('cuda')

    def forward(self, query, key, value, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        # logger.error(Q.device)
        # logger.error(K.device)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = self.fc(x)
        return x


class CrossAttention_1(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(CrossAttention_1, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        # self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to('cuda')

    def forward(self, query, key, value, mask=None):
        q_value = query.shape[0]
        if query.shape[0] > key.shape[0]:
            ex = query.shape[0] - key.shape[0]
            zero = torch.zeros(ex, 1, 1024).float().to('cuda')
            key = torch.cat([key, zero], 0)
            value = torch.cat([value, zero], 0)
        elif query.shape[0] < key.shape[0]:
            ex = key.shape[0] - query.shape[0]
            zero = torch.zeros(ex, 1, 1024).float().to('cuda')
            query = torch.cat([query, zero], 0)
        else:
            pass

        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = self.fc(x)
        if q_value < bsz:
            x = x.split(q_value, 0)[0]
        return x


class CrossAttention_2(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(CrossAttention_2, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        # self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to('cuda')

    def forward(self, query, key, value, mask=None):
        q_value = query.shape[0]
        if query.shape[0] > key.shape[0]:
            ex = query.shape[0] - key.shape[0]
            zero = torch.zeros(ex, 1, 1024).float().to('cuda')
            key = torch.cat([key, zero], 0)
            value = torch.cat([value, zero], 0)
        elif query.shape[0] < key.shape[0]:
            ex = key.shape[0] - query.shape[0]
            zero = torch.zeros(ex, 1, 1024).float().to('cuda')
            query = torch.cat([query, zero], 0)
        else:
            pass

        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = self.fc(x)
        if q_value < bsz:
            x = x.split(q_value, 0)[0]
        return x


class PoswiseFeedForwardNet_1(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet_1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        # residual = inputs
        output = self.fc(inputs)
        # output = self.ln(output)
        return output  # [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet_2(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet_2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        # residual = inputs
        output = self.fc(inputs)
        # output = self.ln(output)
        return output  # [batch_size, seq_len, d_model]


class SCABlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SCABlock, self).__init__()
        # self.img_self_attn = SelfAttention_1(hid_dim=d_model, n_heads=n_heads, dropout=0.1)  # 多头注意力机制
        # self.text_self_attn = SelfAttention_2(hid_dim=d_model, n_heads=n_heads, dropout=0.1)  # 多头注意力机制
        # self.img_cross_attn = CrossAttention_1(hid_dim=d_model, n_heads=n_heads, dropout=0.1)  # 多头注意力机制
        # self.text_cross_attn = CrossAttention_2(hid_dim=d_model, n_heads=n_heads, dropout=0.1)  # 多头注意力机制
        self.img_pos_ffn = PoswiseFeedForwardNet_1()
        self.text_pos_ffn = PoswiseFeedForwardNet_2()

    def forward(self, img_feature, text_feature):
        img_feature = img_feature.float()
        text_feature = text_feature.float()
        img_feature = img_feature.unsqueeze(1)
        text_feature = text_feature.unsqueeze(1)

        # self_img_feature = self.img_self_attn(img_feature, img_feature, img_feature)
        # self_text_feature = self.text_self_attn(text_feature, text_feature, text_feature)
        #
        # cross_img_feature = self.img_cross_attn(self_img_feature, self_text_feature, self_text_feature)
        # cross_text_feature = self.text_cross_attn(self_text_feature, self_img_feature, self_img_feature)

        self_img_feature = self.img_pos_ffn(img_feature)
        self_text_feature = self.text_pos_ffn(text_feature)
        self_img_feature = img_feature + self_img_feature
        self_text_feature = text_feature + self_text_feature

        img_feature_output = self_img_feature.squeeze(1)
        text_feature_output = self_text_feature.squeeze(1)

        img_feature_output = img_feature_output.half()
        text_feature_output = text_feature_output.half()

        return img_feature_output, text_feature_output


if __name__ == '__main__':
    # a = torch.tensor(((1, 2), (3, 4), (5, 6))).float().to('cuda')
    # b = torch.tensor(((6, 5), (4, 3))).float().to('cuda')
    # print(a)
    # print(b)
    # c = torch.zeros(a.shape[0] - b.shape[0], 2).float().to('cuda')
    # b = torch.cat([b, c], 0)
    # print(b)
    # model = SCABlock(d_model, n_heads).to('cuda')
    # c, d = model(a, b)
    # print(c)
    # print(d)
    # d = d.split(2, 0)[0]
    # print(d)
    # exit()

    model = SCABlock(d_model, n_heads).to('cuda')
    a = torch.rand(32, 1024).to('cuda')  # img
    b = torch.rand(37, 1024).to('cuda')  # text
    # c = torch.zeros(b.shape[0] - a.shape[0], 1024).to('cuda')
    # print(c.shape)
    # a = torch.cat([a, c], 0)
    # print(a.shape)
    c, d = model(a, b)
    print(c.shape)
    print(d.shape)
    exit()
