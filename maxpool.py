import torch
import torch.nn as nn

class MaxPool2DMinimalFunction(torch.autograd.Function):
    """
    This autograd function implements a MaxPooling function.
    """

    @staticmethod
    def forward(ctx, input, kernel_size, stride):

        out = None
        pool_height, pool_width = kernel_size, kernel_size
        ctx.pool_height, ctx.pool_width = pool_height, pool_width
        ctx.stride = stride
        
        N, C, H, W = input.size()
        
        assert pool_height == pool_width == stride, "Invalid pool params"  
        x_reshaped = input.reshape(N, C, H // pool_height, pool_height, W // pool_width, pool_width)
        out = torch.amax(torch.amax(x_reshaped,3),4)
        ctx.save_for_backward(input, x_reshaped, out)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):

        input, x_reshaped, out = ctx.saved_tensors
        dx_reshaped = x_reshaped.new_zeros(x_reshaped.size())
        out_newaxis = out[:, :, :, None, :, None]
        mask = (x_reshaped == out_newaxis)
        dout_newaxis = grad_outputs[:, :, :, None, :, None]
        dout_broadcast = torch.broadcast_to(dout_newaxis, dx_reshaped.size())
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= torch.sum(mask, axis=(3, 5), keepdims=True)
        grad_x = dx_reshaped.reshape(input.size())
        return grad_x, None, None


class MaxPool2DMinimal(nn.Module):

    def __init__(self, kernel_size, stride):
        super(MaxPool2DMinimal, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        return MaxPool2DMinimalFunction.apply(input, self.kernel_size, self.stride)


class MaxPool2DBeta(nn.Module):

    def __init__(self, beta=1):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = MaxPool2DMinimal(kernel_size=2, stride=2)
        self.beta = beta

    def forward(self, input_tensor):
        return self.beta * self.maxpool2(input_tensor) + (1 - self.beta) * self.maxpool1(input_tensor)
