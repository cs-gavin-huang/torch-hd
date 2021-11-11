import torch
import torch.nn as nn
import numpy as np
import math

class RandomProjectionEncoder(nn.Module):
    def __init__(self, dim_in, D = 5000, p = 0.5, dist = 'normal', mean = 0.0,
            std = 1.0, quantize = True):
        super().__init__()
        self.D = D
        self.quantize = quantize
        self.dim_in = dim_in
        self.flat = nn.Flatten()

        if quantize:
            if dist == 'bernoulli':
                rand_proj = 2 * torch.rand(size = (dim_in, D))
                proj = torch.where(rand_proj > p, 1, -1).type(torch.float)
            elif dist == 'normal':
                means = torch.ones((dim_in, D)) * mean
                std = torch.ones((dim_in, D)) * std
                proj = torch.normal(means, std)
                proj = torch.where(proj > p, 1, -1).type(torch.float)
        else:
            means = torch.ones((dim_in, D)) * mean
            std = torch.ones((dim_in, D)) * std
            proj = torch.normal(means, std).type(torch.float)

        self.proj_matrix = nn.Parameter(proj, requires_grad = False)

    def forward(self, x, y = None):
        x = self.flat(x)
        out = torch.matmul(x, self.proj_matrix.detach())

        if self.quantize:
            out = torch.sign(out)

        return out

class IDLevelEncoder(nn.Module):
    def __init__(self, dim_in, D, qbins = 16, pact=False, nbits=3, max_val = None,
            min_val = None, sparsity = 0.5, quantize=False):
        super().__init__()
        self.dim_in = dim_in
        self.D = D
        self.pact = pact
        self.quant = quantize
        self.sparsity= sparsity

        if pact:
            self.nbits = nbits
            self.alpha = nn.Parameter(torch.tensor(2.0))
            self.maxval = 2 ** nbits
            self.minval = 0
            self.activn = pact_actvn.apply
        else:
            assert max_val is not None
            assert min_val is not None

            self.maxval = max_val
            self.minval = min_val


        self.bin_len = (self.maxval - self.minval) / qbins
        self.qbins = torch.tensor(qbins)
        intervals = torch.arange(self.minval, self.maxval, self.bin_len)
        self.intervals = nn.Parameter(intervals, requires_grad = False)
        
        #### Generate ID hypervectors
        temp = torch.ones(size=(dim_in, D)) * sparsity
        temp = 2 * torch.bernoulli(temp) - 1
        self.id_hvs = nn.Parameter(temp.type(torch.float), requires_grad = False)

        #### Generate Level hypervector
        lvl_hvs = []
        #temp = [-1]*int(D/2) + [1]*int(D/2)
        temp = [1] * int(D * sparsity) + [-1] * int(D * (1 - sparsity))
        np.random.shuffle(temp)
        lvl_hvs.append(temp)
        change_list = np.arange(0, D)
        np.random.shuffle(change_list)
        cnt_toChange = math.floor(D/2 / (qbins))
        for i in range(1, qbins + 1):
          temp = np.array(lvl_hvs[i-1])
          temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]] = -temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]]
          lvl_hvs.append(list(temp))
        lvl_hvs = torch.tensor(lvl_hvs).type(torch.float)
        self.lvl_hvs = nn.Parameter(lvl_hvs, requires_grad = False)
        self.flat = nn.Flatten()
    
    def forward(self, x):
        x = self.flat(x)
        x = x.clamp(self.minval, self.maxval)
        
        if self.pact:
            x = self.activn(x, self.alpha, self.nbits)

        #idx = torch.floor(x / self.bin_len).type(torch.long)
        idx = torch.searchsorted(self.intervals.detach(), x)
        encoded = (self.lvl_hvs.detach()[idx] * self.id_hvs.detach()).sum(dim=1)
        if self.quant:
            val = torch.tensor(2 ** self.nbits)
            encoded = torch.clamp(encoded, -val, val-1)
        else:
            encoded = torch.clamp(encoded, -1, 1)
            ones = torch.ones_like(encoded) * self.sparsity
            ones = 2 * torch.bernoulli(ones) - 1
            encoded[encoded == 0] = ones[encoded == 0]
        
        return encoded


class IDLevelDecoder(nn.Module):
    def __init__(self, id_hvs, lvl_hvs, bin_len, min_val, max_val):
        super().__init__()
        self.id_hvs = id_hvs
        self.lvl_hvs = lvl_hvs
        self.bin_len = bin_len
        self.minval = min_val
        self.maxval = max_val
      
    def forward(self, x):
        decoded = x.repeat(1, self.id_hvs.shape[0]).view(x.shape[0], self.id_hvs.shape[0], x.shape[1]) * self.id_hvs.detach()
        decoded = self.minval + torch.matmul(decoded, self.lvl_hvs.detach().transpose(0,1)).max(dim=2)[1] * self.bin_len
            
        return decoded

class IDLevelCodec(nn.Module):
    def __init__(self, dim_in, D, pact=True, nbits=3, qbins=8, max_val = None, min_val = None, quantize=False):
        super().__init__()
        self.encoder = IDLevelEncoder(dim_in, D, qbins, pact, nbits, max_val, min_val, quantize = quantize)
        self.decoder = IDLevelDecoder(
            self.encoder.id_hvs, self.encoder.lvl_hvs, self.encoder.bin_len,
            self.encoder.minval, self.encoder.maxval
        )
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)

        return out

class pact_actvn(torch.autograd.Function):
    '''
    Code for the pact activation was taken from
    https://github.com/KwangHoonAn/PACT
    '''
    @staticmethod
    def forward(ctx, x, alpha, k):
        ctx.save_for_backward(x, alpha)
        #y_1 = 0.5 * (torch.abs(x) - torch.abs(x - alpha) + alpha)
        y = torch.clamp(x, min = 0, max = alpha.item())
        scale = (2 ** k - 1) / alpha
        y_q = torch.round(y * scale) / scale

        return y_q
    
    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py 
        # We get dL / dy_q as a gradient
        x, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha] 
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
        lower_bound      = x < 0
        upper_bound      = x > alpha
        # x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)

        return dLdy_q * x_range.float(), grad_alpha, None



if __name__ == '__main__':
    testdata = torch.tensor([[0, 4, 1, 3, 0]]).type(torch.float).cuda()
    model = IDLevelCodec(dim_in=5, D=10000, pact=False, qbins = 8, max_val = 8, min_val = 0)
    model.cuda()
    out = model(testdata)
    print(testdata, out)

    model = IDLevelCodec(dim_in=5, D=10000, pact=True, qbins = 9)
    model.cuda()
    out = model(testdata)
    print(testdata, out)

    model = RandomProjectionEncoder(5, D = 10, quantize = True)
    model.cuda()
    out = model(testdata)
    print(testdata, out)

