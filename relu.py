import torch
import torch.nn as nn

class ReLUAlphaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input == 0] = ctx.alpha * grad_output[input == 0]
        return grad_input, None


class ReLUAlpha(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        return ReLUAlphaFunction.apply(input, self.alpha)

# Usage example:
# relu_alpha = ReLUAlpha(alpha=0.1)
# output = relu_alpha(input_tensor)
