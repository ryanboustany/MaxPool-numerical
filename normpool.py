import torch
from torch import nn
from torch.autograd import Function

class NormPool1Custom(Function):

    @staticmethod
    def forward(ctx, input_tensor, kernel_size, stride):
        N, C, H, W = input_tensor.size()
        pool_height, pool_width = kernel_size, kernel_size
        ctx.save_for_backward(input_tensor)
        ctx.kernel_size = kernel_size
        ctx.stride = stride

        assert pool_height == pool_width == stride, "Invalid pool parameters"

        input_reshaped = input_tensor.reshape(N, C, H // pool_height, pool_height, W // pool_width, pool_width)
        output = torch.sqrt((input_reshaped ** 2).sum(dim=(3, 5)))

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input_tensor, = ctx.saved_tensors
        N, C, H, W = input_tensor.size()
        pool_height, pool_width = ctx.kernel_size, ctx.kernel_size

        input_reshaped = input_tensor.reshape(N, C, H // pool_height, pool_height, W // pool_width, pool_width)
        output = torch.sqrt((input_reshaped ** 2).sum(dim=(3, 5)))
        output_newaxis = output[:, :, :, None, :, None]
        norm_factor = torch.where(
            output_newaxis != 0,
            input_reshaped / output_newaxis,
            torch.tensor([0.0], dtype=input_tensor.dtype, device=input_tensor.device)
        )
        grad_reshaped = norm_factor * grad_outputs[:, :, :, None, :, None]
        grad_input = grad_reshaped.reshape(input_tensor.size())

        return grad_input, None, None


class NormPool2Custom(Function):

    @staticmethod
    def forward(ctx, input_tensor, kernel_size, stride):
        N, C, H, W = input_tensor.size()
        pool_height, pool_width = kernel_size, kernel_size
        ctx.save_for_backward(input_tensor)
        ctx.kernel_size = kernel_size
        ctx.stride = stride

        assert pool_height == pool_width == stride, "Invalid pool parameters"

        input_reshaped = input_tensor.reshape(N, C, H // pool_height, pool_height, W // pool_width, pool_width)
        output = torch.sqrt((input_reshaped ** 2).sum(dim=(3, 5)))

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input_tensor, = ctx.saved_tensors
        N, C, H, W = input_tensor.size()
        pool_height, pool_width = ctx.kernel_size, ctx.kernel_size

        input_reshaped = input_tensor.reshape(N, C, H // pool_height, pool_height, W // pool_width, pool_width)
        output = torch.sqrt((input_reshaped ** 2).sum(dim=(3, 5)))
        output_newaxis = output[:, :, :, None, :, None]
        norm_factor = torch.where(
            output_newaxis != 0,
            input_reshaped / output_newaxis,
            torch.tensor([1.0], dtype=input_tensor.dtype, device=input_tensor.device)
        )
        grad_reshaped = norm_factor * grad_outputs[:, :, :, None, :, None]
        grad_input = grad_reshaped.reshape(input_tensor.size())

        return grad_input, None, None


class NormPool1(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input_tensor):
        return NormPool1Custom.apply(input_tensor, self.kernel_size, self.stride)


class NormPool2(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input_tensor):
        return NormPool2Custom.apply(input_tensor, self.kernel_size, self.stride)
