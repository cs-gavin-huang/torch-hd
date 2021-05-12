import torch
import torch.nn as nn
import numpy as np
import math


class hd_rp_encoder(nn.Module):
    def __init__(self, size_in, D=5000, p=0.5):
        super().__init__()
        self.dim = D
        probs = torch.ones((size_in, D)) * p
        projection = 2 * torch.bernoulli(probs) - 1
        self.hdweights = nn.Parameter(projection, requires_grad=False)
        self.flat = nn.Flatten()
  
    def forward(self, x):
        x = self.flat(x)
        x = torch.matmul(x, self.hdweights)

        return x

class hdsign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return torch.sign(input_)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        ret = grad_input * (1 - torch.square(torch.tanh(input_)))

        return ret

class hd_rp_layer(nn.Module):
    def __init__(self, size_in, D=5000, p =0.5, quantize = True):
        super().__init__()
        self.encoder = hd_rp_encoder(size_in, D, p)
        self.quantize = quantize
    
    def forward(self, x):
        out = self.encoder(x)

        if self.quantize:
            out = torch.sign(out)
        else:
            if self.training:
                out = nn.Tanh(out)
            else:
                out = torch.sign(out)

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

class hd_id_lvl_encoder(nn.Module):
    def __init__(self, nfeats, D, qbins = 16, pact=True, k=3, max_val = None, min_val = None):
        super().__init__()
        self.nfeats = nfeats
        self.D = D
        self.pact = pact

        if pact:
            self.k = k
            self.alpha = nn.Parameter(torch.tensor(2.0))
            self.maxval = 2 ** k
            self.minval = 0
            self.activn = pact_actvn.apply
        else:
            assert max_val is not None
            assert min_val is not None

            self.maxval = max_val
            self.minval = min_val

        self.bin_len = (self.maxval - self.minval) / qbins
        self.qbins = torch.tensor(qbins)
        
        #### Generate ID hypervectors
        temp = torch.rand(size=(nfeats, D))
        temp = torch.where(temp > 0.5, 1, -1)
        self.id_hvs = nn.Parameter(temp.type(torch.float), requires_grad = False)

        #### Generate Level hypervector
        lvl_hvs = []
        temp = [-1]*int(D/2) + [1]*int(D/2)
        np.random.shuffle(temp)
        lvl_hvs.append(temp)
        change_list = np.arange(0, D)
        np.random.shuffle(change_list)
        cnt_toChange = math.floor(D/2 / (qbins-1))
        for i in range(1, qbins):
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
            x = self.activn(x, self.alpha, self.k)

        idx = (x // self.bin_len).type(torch.long)
        encoded = (self.lvl_hvs[idx] * self.id_hvs).sum(dim=1)
        encoded = torch.clamp(encoded, -1, 1)
        
        return encoded


class hd_id_lvl_decoder(nn.Module):
    def __init__(self, id_hvs, lvl_hvs, bin_len):
        super().__init__()
        self.id_hvs = id_hvs
        self.lvl_hvs = lvl_hvs
        self.bin_len = bin_len
      
    def forward(self, x):
        decoded = x.repeat(1, self.id_hvs.shape[0]).view(x.shape[0], self.id_hvs.shape[0], x.shape[1]) * self.id_hvs
        decoded = torch.matmul(decoded, self.lvl_hvs.transpose(0,1)).max(dim=2)[1] * self.bin_len
            
        return decoded

class hdcodec(nn.Module):
    def __init__(self, nfeats, D, pact=True, k=3, qbins=8, max_val = None, min_val = None):
        super().__init__()
        self.encoder = hd_id_lvl_encoder(nfeats, D, qbins, pact, k, max_val, min_val)
        self.decoder = hd_id_lvl_decoder(
            self.encoder.id_hvs, self.encoder.lvl_hvs, self.encoder.bin_len
        )
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)

        return out


class hd_classifier(nn.Module):
    def __init__(self, nclasses, D, alpha = 1.0):
        super().__init__()
        self.class_hvs = nn.Parameter(torch.zeros(size=(nclasses, D)), requires_grad = False)
        self.nclasses = nclasses
        self.alpha = alpha
    
    def forward(self, encoded, targets):
        scores = torch.matmul(encoded, self.class_hvs.transpose(0, 1))

        if self.training:
            _, preds = scores.max(dim=1)

            for label in range(self.nclasses):
                if label in targets:
                    incorrect = encoded[torch.bitwise_and(targets != preds, targets == label)]
                    incorrect = incorrect.sum(dim = 0, keepdim = True).squeeze() * self.alpha
                    self.class_hvs[label] += incorrect.clip(-1, 1)

                incorrect = encoded[torch.bitwise_and(targets != preds, preds == label)]
                incorrect = incorrect.sum(dim = 0, keepdim = True).squeeze()
                self.class_hvs[label] -= incorrect.clip(-1, 1) * self.alpha
        
        return scores


class hd_skc_layer(nn.Module):
    def __init__(self, nfeats, D, r, mean = 0.0, std = 1.0):
        super().__init__()
        
        self.nfeats = nfeats
        self.D = D
        self.r = nn.Parameter(torch.tensor(r), requires_grad = False)

        gauss_std = torch.ones(size=(D, nfeats)) * std
        self.prototypes = nn.Parameter(torch.normal(mean = mean, std = gauss_std), requires_grad = False)
        self.flat = nn.Flatten()
    
    def forward(self, x):
        x = self.flat(x)
        encoded = torch.cdist(x, self.prototypes, p = 2)
        encoded = torch.where(encoded > self.r, 1, -1).type(torch.float)

        return encoded

if __name__ == '__main__':
  testdata = torch.tensor([[0, 4, 1, 3, 0]]).cuda()
  model = hdcodec(nfeats=5, D=10000, qbins = 9)
  out = model(testdata)
  print(testdata, out)

  model = hdcodec(nfeats=5, D=10000, pact=False, qbins = 8, max_val = 8, min_val = 0)
  out = model(testdata)
  print(testdata, out)