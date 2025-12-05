import torch
from torch import nn
from asteroid.masknn import norms
from params import HParams
import torch.nn.functional as F
import sys
EPS = 1e-6
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

hparamas = HParams()

def add_loss(model, targets, post_out):
    post_loss = (F.binary_cross_entropy_with_logits(post_out, targets)).mean()
    regularization = sum([torch.norm(val, 2) for name, val in model.named_parameters() if
                          'weight' in name]).item() * hparamas.vad_reg_weight
    total_loss = post_loss + regularization
    return total_loss, post_loss

def add_loss_noregular(targets, post_out):
    post_loss = (F.binary_cross_entropy_with_logits(post_out, targets)).mean()

    total_loss = post_loss 
    return total_loss, post_loss

def add_loss_noregular_sigmoid(targets, post_out):
    post_out = post_out.float()
    targets = targets.float()
    post_loss = ( F.binary_cross_entropy(post_out, targets)).mean()

    total_loss = post_loss 

    return total_loss, post_loss

def prediction(targets, post_out, w=hparamas.w, u=hparamas.u):
    post_prediction = torch.round(F.sigmoid(post_out))
    post_acc = torch.mean(post_prediction.eq_(targets))

    return post_acc


def mvn(x, dim=-1) -> torch.Tensor:
    """
    Performs mean-variance normalization on a given tensor
    """
    x_norm = (x - torch.mean(x, dim=dim, keepdim=True)) / (
        torch.std(
            x,
            dim=dim,
            keepdim=True,
        )
        + EPS
    )
    return x_norm

class SelfAttention(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        att_dim,
        bias=True,
        ffn_out=False,
    ):
   

        super(SelfAttention, self).__init__()
        self.bias = bias
      
        self.query = torch.nn.Linear(
            in_features=input_dim,
            out_features=att_dim,
            bias=bias,
        ).apply(self._init_weights)
        self.key = torch.nn.Linear(
            in_features=input_dim,
            out_features=att_dim,
            bias=bias,
        ).apply(self._init_weights)
        self.value = torch.nn.Linear(
            in_features=input_dim,
            out_features=1,
            bias=bias,
        ).apply(self._init_weights)
        if not ffn_out:
            self.W0 = None
        else:
            self.W0 = torch.nn.Linear(
                in_features=1,
                out_features=1,
                bias=bias,
            ).apply(self._init_weights)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.att_dim = torch.Tensor([att_dim])
        self.scale_factor = att_dim**-0.5

    def forward(self, x, att_weights_fl=True):
        """
        Returns self attention weights as defined in [1]
        """
        x = mvn(x, -1) 

        Q = self.query(x)  # (B,T,C,D)
        K = self.key(x)  # (B,T,C,D)
        V = self.value(x)  # (B,T,C,1)

        w_att = self.softmax(
            torch.matmul(Q, K.transpose(-1, -2)) * self.scale_factor
        )  # (B,T,C,C)
        w = torch.matmul(w_att, V)

        if self.W0 is not None:
            w = self.W0(w)  # (B,T,C,H)

        if att_weights_fl:
            return w, w_att
        else:
            return w

    def _init_weights(self, m):
        """

        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if self.bias:
                m.bias.data.fill_(0.01)

class Time_SelfAttention(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        att_dim,
        bias=True,
        ffn_out=False,
    ):

        super(Time_SelfAttention, self).__init__()
        self.bias = bias
      
        self.query = torch.nn.Linear(
            in_features=input_dim,
            out_features=att_dim,
            bias=bias,
        ).apply(self._init_weights)
        self.key = torch.nn.Linear(
            in_features=input_dim,
            out_features=att_dim,
            bias=bias,
        ).apply(self._init_weights)
        self.value = torch.nn.Linear(
            in_features=input_dim,
            out_features=att_dim,
            bias=bias,
        ).apply(self._init_weights)


        self.softmax = torch.nn.Softmax(dim=-1)
        self.att_dim = torch.Tensor([att_dim])
        self.scale_factor = att_dim**-0.5

    def forward(self, x, att_weights_fl=True):
        x = mvn(x, -1) 
        Q = self.query(x).transpose(-2, -3)  # (B,C,T,D)
        K = self.key(x).transpose(-2, -3)  # (B,C,T,D)
        V = self.value(x).transpose(-2, -3)  # (B,C,T,D)
        w = fast_full_cross_channel_attention(Q,K,V)
        return w    


    def _init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if self.bias:
                m.bias.data.fill_(0.01)

def fast_full_cross_channel_attention(Q, K, V, window=3):

    B, C, T, D = Q.shape
    output = torch.zeros(B, C, T, D, device=Q.device)

    for c_q in range(C):
        q = Q[:, c_q]  # (B, T, D)
        output_cq = torch.zeros(B, T, D, device=Q.device)

        for c_k in range(C):
            if c_k == c_q:
                continue  

            k = K[:, c_k]  # (B, T, D)
            v = V[:, c_k]  # (B, T, D)

            outputs_ck = []

            for t in range(T):
                start = max(0, t - window + 1)
                k_slice = k[:, start:t + 1]  # (B, W, D)
                v_slice = v[:, start:t + 1]  # (B, W, D)
                q_t = q[:, t].unsqueeze(1)  # (B, 1, D)

                attn_scores = torch.bmm(q_t, k_slice.transpose(1, 2)).squeeze(1)  # (B, W)
                attn_weights = F.softmax(attn_scores / (D ** 0.5), dim=-1)  # (B, W)

                out_t = torch.sum(attn_weights.unsqueeze(-1) * v_slice, dim=1)  # (B, D)
                outputs_ck.append(out_t)


            attn_output_ck = torch.stack(outputs_ck, dim=1)  # (B, T, D)
            output_cq += attn_output_ck


        output[:, c_q] = output_cq

    return output  # (B, C, T, D) 

class TAttentionWithProjection(nn.Module):
    def __init__(self,  input_dim, att_dim, bias=True, ffn_out=False):
        super(TAttentionWithProjection,self).__init__()
        self.bias = bias
      
        self.query = torch.nn.Linear(
            in_features=input_dim,
            out_features=att_dim,
            bias=bias,
        ).apply(self._init_weights)
        self.key = torch.nn.Linear(
            in_features=input_dim,
            out_features=att_dim,
            bias=bias,
        ).apply(self._init_weights)
        self.value = torch.nn.Linear(
            in_features=input_dim,
            out_features=att_dim,
            bias=bias,
        ).apply(self._init_weights)
        self.d_model = att_dim

        self.PA = nn.Linear(att_dim, att_dim)     
        self.PC = nn.Linear(att_dim * 2, att_dim)  


    def cross_channel_attention(self, x):

        x = mvn(x, -1) 
        Q = self.query(x).transpose(-2, -3)  # (B, M, T, d)
        K = self.key(x).transpose(-2, -3)    # (B, M, T, d)
        V = self.value(x).transpose(-2, -3)  # (B, M, T, d)

        B, M, T, d = Q.shape
        A = torch.zeros_like(Q)
        sqrt_d = d ** 0.5

        causal_mask = torch.tril(torch.ones(T, T, device=x.device))  # (T, T)

        for b in range(B):
            for m in range(M):
                Qm = Q[b, m]  # (T, d)
                Am = 0
                for n in range(M):
                    Kn = K[b, n]  # (T, d)
                    Vn = V[b, n]  # (T, d)

     
                    scores = Qm @ Kn.transpose(0, 1) / sqrt_d
            
                    masked_scores = scores.masked_fill(causal_mask == 0, float('-inf'))
                    attn_weights = torch.softmax(masked_scores, dim=-1)  # (T, T)

               
                    Am += attn_weights @ Vn

                A[b, m] = Am

        return A  


    
    def cross_channel_window_attention(self, x, window_size=5):

        x = mvn(x, -1) 
        Q = self.query(x).transpose(-2, -3)  # (B, M, T, d)
        K = self.key(x).transpose(-2, -3)
        V = self.value(x).transpose(-2, -3)
        B, M, T, d = Q.shape
        sqrt_d = d ** 0.5
        A = torch.zeros_like(Q)

 
        mask = torch.full((T, T), float('-inf'), device=x.device)
        for i in range(T):
            start = max(0, i - window_size // 2)
            end = min(T, i + window_size // 2 + 1)
            mask[i, start:end] = 0 

   
        for b in range(B):
            for m in range(M):
                Qm = Q[b, m]  # (T, d)
                Am = 0
                for n in range(M):
                    Kn = K[b, n]  # (T, d)
                    Vn = V[b, n]  # (T, d)

                    score = (Qm @ Kn.T) / sqrt_d  # (T, T)
                    masked_score = score + mask  
                    attn_weights = torch.softmax(masked_score, dim=-1)  # (T, T)
                    Am += attn_weights @ Vn  # (T, d)

                A[b, m] = Am
        return A  # (B, M, T, d)

    def forward(self,x):

        A = self.cross_channel_attention(x)  
        concat = torch.cat([x.transpose(-2, -3), A_proj], dim=-1)   # (B, M, T, 2d)
        Z_hat = self.PC(concat)                    # (B, M, T, d)
        return Z_hat
    
    def _init_weights(self, m):
        """

        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if self.bias:
                m.bias.data.fill_(0.01)
    
class node_choice(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l1', after_relu=False):
        super(node_choice, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, num_channels, 1)).to(device)
        self.gamma = nn.Parameter(torch.ones(1, 1, num_channels, 1)).to(device)  # 替代 zeros
        self.beta = nn.Parameter(torch.zeros(1, 1, num_channels, 1)).to(device)
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum(3, keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(2, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            _x = torch.abs(x) if not self.after_relu else x
            embedding = _x.sum(3, keepdim=True) * self.alpha
            norm = self.gamma / (embedding.abs().mean(2, keepdim=True) + self.epsilon)

        elif self.mode == 'inf':
            _x = torch.abs(x) if not self.after_relu else x
            embedding = _x.max(3, keepdim=True).values * self.alpha
            norm = self.gamma / (embedding.max(2, keepdim=True).values + self.epsilon)

        else:
            print('Unknown mode!')
            sys.exit()

        y = embedding * norm + self.beta
        gate = torch.softmax(y, dim=-2)
        return gate

class node_fusion_v2(nn.Module):
    def __init__(self, input_dim, att_dim, temperature=0.05):
        super(node_fusion_v2, self).__init__()
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.temperature = temperature
        self.selfatt = SelfAttention(self.input_dim, self.att_dim)
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, x):
        w_comb, w_att = self.selfatt(x)
        w_comb = self.softmax(w_comb)
        return x, w_comb
    
class node_fusion_v3(nn.Module):
    def __init__(self, input_dim, att_dim, temperature=0.05):
        super(node_fusion_v3, self).__init__()
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.temperature = temperature
        self.selftimeatt = TAttentionWithProjection(self.input_dim, self.att_dim)
        self.selfatt = SelfAttention(self.input_dim, self.att_dim)
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, x):
        dq_x = self.selftimeatt(x).transpose(-2, -3)
        w_comb, w_att = self.selfatt(dq_x)
        w_comb = self.softmax(w_comb)
        return dq_x, w_comb
    
class node_fusion_v4(nn.Module):
    def __init__(self, input_dim, att_dim, temperature=0.05):
        super(node_fusion_v4, self).__init__()
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.temperature = temperature
        self.selftimeatt = TAttentionWithProjection(self.input_dim, self.att_dim)
        self.selfatt = SelfAttention(self.input_dim, self.att_dim)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.PC = nn.Linear(att_dim * 2, att_dim)  


    def forward(self, x):
        dq_x = self.selftimeatt(x).transpose(-2, -3)
        b,t,c,d = dq_x.shape
        half = int(c/2)
        upx = dq_x[:,:,0:half,:]
        downx = dq_x[:,:,half:,:]
        upw_comb, upw_att = self.selfatt(upx)
        upw_comb = self.softmax(upw_comb)


        downw_comb, downw_att = self.selfatt(downx)
        downw_comb = self.softmax(downw_comb)

        residual = (upx*upw_comb).sum(-2) - (downx*downw_comb).sum(-2)
        w_comb, w_att = self.selfatt(dq_x)
        w_comb = self.softmax(w_comb)
        wx = (dq_x*w_comb).sum(-2)
        z = torch.cat((wx,residual),dim=-1)
        z = self.PC(z)


        return z

class IPD_Weight(nn.Module):
    def __init__(self, input_dim, att_dim, temperature=0.05):
        super(IPD_Weight, self).__init__()
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.temperature = temperature
        self.selfatt = SelfAttention(self.input_dim, self.att_dim)
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, x):
        w_comb, w_att = self.selfatt(x)
        w_comb = self.softmax(w_comb)

        return x, w_comb
      
class Encoder(nn.Module):
    def __init__(self, in_dim = 40, out_dim=64, kernel_size=3, dilation=1):
        super(Encoder, self).__init__()
        self.pad = nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0)  
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation)
        self.gru = nn.GRU(input_size=64,hidden_size=64,batch_first = True)


    def forward(self, x):
        c,f,t = x.shape

        x = self.conv(self.pad(x))
        out,_ = self.gru(x.permute(0,2,1))
        return out.permute(0,2,1)
       
    
class Decoder(nn.Module):
    def __init__(self, dim = 64, out_chan=1):
        super(Decoder, self).__init__()
        out_conv = nn.Conv1d(dim, out_chan, 1).to(device)
        self.out =  nn.Sequential(nn.PReLU(), out_conv).to(device)

    def forward(self, x):
        logits = self.out(x)
        return logits
    
class Decoder_sigmoid(nn.Module):
    def __init__(self, dim = 64, out_chan=1):
        super(Decoder_sigmoid, self).__init__()
        out_conv = nn.Conv1d(dim, out_chan, 1).to(device)
        self.out =  nn.Sequential(nn.PReLU(), out_conv).to(device)

    def forward(self, x):
        logits = self.out(x)
        return F.sigmoid(logits)
    
class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation, norm_type="bN", delta=False):
        super(Conv1DBlock, self).__init__()
        conv_norm = norms.get(norm_type)
        self.delta = delta
        if delta:
            self.linear = nn.Linear(in_chan, in_chan)
            self.linear_norm = norms.get(norm_type)(in_chan*2)

        in_bottle = in_chan if not delta else in_chan*2
        in_conv1d = nn.Conv1d(in_bottle, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        if self.delta:
            delta = self.linear(x.transpose(1, -1)).transpose(1, -1)
            x = torch.cat((x, delta), 1)
            x = self.linear_norm(x)

        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out



class TCN(nn.Module):
    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3,
                 bn_chan=128, hid_chan=512,  kernel_size=3,
                 norm_type="gLN"):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        layer_norm = norms.get(norm_type)(in_chan).to(device)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1).to(device)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv).to(device)
        
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan,
                                            kernel_size, padding=padding,
                                            dilation=2**x, norm_type=norm_type).to(device))
       
        

    def forward(self, mixture_w):
 
        output = self.bottleneck(mixture_w.to(device))
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual
        
        return output

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config
    
class SelfAttentionImportance(nn.Module):
    def __init__(self, feature_dim=128, causal_mask=True):
        super(SelfAttentionImportance, self).__init__()
        self.causal_mask = causal_mask
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoded):

        x = encoded.permute(0, 2, 1)
        batch_size, seq_len, _ = x.size()
        
        q = self.query(x)
        k = self.key(x)  
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # [B=1, T, T]
        if self.causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device) * float('-inf'), diagonal=1)
            scores = scores + mask
        
        attn_weights = F.softmax(scores, dim=-1)  # [B=1, T, T]
        
        context = torch.matmul(attn_weights, x)  # [B=1, T, 128]

        importance = self.output(context).squeeze(-1)  # [B=1, T]
        
        importance = self.sigmoid(importance)
        if importance.size(0) == 1:
            importance = importance.squeeze(0)  # [T]
            
        return importance    

class TCNImpNet(nn.Module):
    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3,
                 bn_chan=128, hid_chan=512,  kernel_size=3,
                 norm_type="gLN"):
        super().__init__()
        self.encoder = TCN(in_chan, n_src=1, out_chan=in_chan,
                           n_blocks=n_blocks, n_repeats=n_repeats,
                           bn_chan=bn_chan, hid_chan=hid_chan, kernel_size=kernel_size)

        self.importance_head = nn.Sequential(
            nn.Conv1d(bn_chan, bn_chan, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(bn_chan, bn_chan // 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(bn_chan // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        


    def forward(self, x):
        # x: [C, D, T]
        C, D, T = x.shape
        encoded_feats = []
        importance_scores = []

        for c in range(C):
            x_c = x[c].unsqueeze(0)  # [1, D, T]
            encoded = self.encoder(x_c)  # [1, D, T]
          
            score = self.importance_head(encoded)


            encoded_feats.append(encoded.squeeze(0))               # [D, T]
            importance_scores.append(score.squeeze(0).squeeze(0))  # [T]

        encoded_feats = torch.stack(encoded_feats, dim=0)         # [C, D, T]
        importance_scores = torch.stack(importance_scores, dim=0) # [C, T]

        return encoded_feats, importance_scores
    

class CI2E(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.importance_head = nn.Sequential(
            nn.Conv1d(in_chan, in_chan, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_chan, in_chan // 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(in_chan // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: [C, D, T]
        C, D, T = x.shape
        importance_scores = []
        for c in range(C):  
            score = self.importance_head(x[c])
            importance_scores.append(score.squeeze(0).squeeze(0))  # [T]
        importance_scores = torch.stack(importance_scores, dim=0) 
        return importance_scores
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



