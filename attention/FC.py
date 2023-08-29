import torch
import torch.nn as nn


class CLS_Head(nn.Module):
    def __init__(self, vis_dim, n_cls):
        super(CLS_Head, self).__init__()
        self.fc = nn.Linear(vis_dim, n_cls, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        # x = self.relu(x)
        x = self.sm(x)
        return x


class PoswiseFeedForwardNet_1(nn.Module):
    def __init__(self, vis_dim):
        super(PoswiseFeedForwardNet_1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // 4, vis_dim, bias=False))
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        # residual = inputs
        output = self.fc(inputs)
        # output = self.ln(output)
        return output  # [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet_2(nn.Module):
    def __init__(self, vis_dim):
        super(PoswiseFeedForwardNet_2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // 4, vis_dim, bias=False))
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        # residual = inputs
        output = self.fc(inputs)
        # output = self.ln(output)
        return output  # [batch_size, seq_len, d_model]


# double fc
class FCBD(nn.Module):
    def __init__(self, vis_dim):
        super(FCBD, self).__init__()
        self.img_pos_ffn = PoswiseFeedForwardNet_1(vis_dim)
        self.text_pos_ffn = PoswiseFeedForwardNet_2(vis_dim)

    def forward(self, img_feature, text_feature):
        self_img_feature = self.img_pos_ffn(img_feature)
        self_text_feature = self.text_pos_ffn(text_feature)
        img_feature = img_feature + self_img_feature
        text_feature = text_feature + self_text_feature
        return img_feature, text_feature


# single fc
class FCBS(nn.Module):
    def __init__(self, vis_dim):
        super(FCBS, self).__init__()
        self.pos_ffn = PoswiseFeedForwardNet_1(vis_dim)

    def forward(self, img_feature, text_feature):
        img_l = img_feature.shape[0]
        text_l = text_feature.shape[0]

        fusion = torch.cat([img_feature, text_feature], 0)
        fusion = self.pos_ffn(fusion)

        self_img_feature, self_text_feature = fusion.split([img_l, text_l], 0)

        img_feature = img_feature + self_img_feature
        text_feature = text_feature + self_text_feature
        return img_feature, text_feature


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

    # model = FCBS(1024).to('cuda')
    model = FCBD(1024).to('cuda')
    a = torch.rand(37, 1024).to('cuda')  # img
    b = torch.rand(32, 1024).to('cuda')  # text
    c, d = model(a, b)
    print(c.shape)
    print(d.shape)
    exit()
