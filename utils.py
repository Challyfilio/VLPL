import torch
from thop import profile
from thop import clever_format
from torchsummary import summary
from torchstat import stat


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    all_size = param_count / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    # return param_count


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    # return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def model_summary(model):
    count_param(model)
    getModelSize(model)

    input = torch.randn(1, 3, 224, 224).float().cuda()

    input1 = torch.tensor(torch.randn(37, 77, 512), dtype=torch.float16).cuda()
    input2 = torch.tensor(torch.randn(37, 77), dtype=torch.int64).cuda()
    # input1 = torch.randn(37, 77, 512).half().cuda()
    # input2 = torch.randn(37, 77).int64().cuda()
    # flops, params = profile(model.cuda(), inputs=(input,))
    flops, params = profile(model.cuda(), inputs=(input1,input2,))
    flops, params = clever_format([flops, params], '%.3f')
    print('模型参数：', params)
    print('每一个样本浮点运算量：', flops)

    # print(summary(model, (3, 224, 224), device="cuda"))
    # print(summary(model, input_size=(3, 224, 224), batch_size=-1, device="cuda"))
