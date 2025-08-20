import torch


def softmax(x: torch.Tensor, dim=-1):
    x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    x = x / x.sum(dim=dim, keepdim=True)
    return x

def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


def cross_entropy(inputs, targets):
    x = -log_softmax(inputs)
    return torch.mean(torch.gather(x, -1, targets.unsqueeze(-1)))





