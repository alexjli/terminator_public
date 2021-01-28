# adapted from https://github.com/drorlab/gvp/blob/master/src/gvp.py

import torch
import torch.nn as nn

def norm_no_nan(x, dim = -1, keepdim = False, eps=1e-8, sqrt=True):
    local_dev = x.device
    inner = torch.sum(torch.square(x.clone()), dim = dim, keepdim = keepdim)
    eps_tensor = torch.tensor(eps).view([1 for _ in range(len(inner.shape))]).expand(inner.shape).to(local_dev)
    out = torch.where(inner > eps, inner, eps_tensor)
    return (torch.sqrt(out) if sqrt else out)

class GVP(nn.Module):
    def __init__(self, vi, vo, si, so,
                 nlv = torch.sigmoid, nls = nn.ReLU):
        '''[v/s][i/o] = number of [vector/scalar] channels [in/out]'''
        super(GVP, self).__init__()
        nh = max(vi, vo)
        if vi: self.wh = nn.Linear(vi, nh)
        if nls and vi:
            self.ws = nn.Sequential(
                        nn.Linear(nh + si, so),
                        nls()
                    )
        elif nls:
            self.ws = nn.Sequential(
                        nn.Linear(nh, so),
                        nls()
                    )
        elif vi:
            self.ws = nn.Linear(nh + si, so)
        else:
            self.ws = nn.Linear(nh, so)

        if vo: self.wv = nn.Linear(nh, vo)
        self.vi, self.vo, self.si, self.so, self.nlv = vi, vo, si, so, nlv

    def forward(self, x, return_split = False):
        # X: [..., 3*vi + si]
        # if split, returns: [..., 3, vo], [..., so]
        # if not split, returns [..., 3*vo + so]
        v, s = split(x, self.vi)
        if self.vi:
            vh = self.wh(v)
            vn = norm_no_nan(vh, dim=-2)
            out = self.ws(torch.cat([s, vn], dim=-1))

            if self.vo: 
                vo = self.wv(vh)
                if self.nlv: vo *= self.nlv(norm_no_nan(vo, dim=-2, keepdim=True))
                out = (vo, out) if return_split else merge(vo, out)

        else: out = self.ws(s)
        return out


# Dropout that drops vector and scalar channels separately
class GVPDropout(nn.Module):
    def __init__(self, rate, nv):
        super(GVPDropout, self).__init__()
        self.nv = nv
        self.rate = rate
        self.sdropout = nn.Dropout(rate)

    def forward(self, x):
        if not self.training: return x
        v, s = split(x, self.nv)
        v, s = self.vdropout(v), self.sdropout(s)
        return merge(v, s)

    # a form of dropout that either drops
    # the whole vector or none of it
    def vdropout(self, x):
        dev = x.device

        p = self.rate
        p_mask = torch.tensor(1-p).to(dev) # probability of a 1
        p_mask = p_mask.view([1 for _ in range(len(x.shape))]) # view so we can expand
        p_mask = p_mask.expand(list(x.shape[:-2]) + [1, self.nv]) # e x p a n d
        mask = torch.bernoulli(p_mask) # now we have dropout probs

        x = mask * x # apply dropout
        x *= 1/(1-p) if p != 1 else 0 # scale by prob, avoid div by 0
        return x


# Normal layer norm for scalars, nontrainable norm for vectors
class GVPLayerNorm(nn.Module):
    def __init__(self, nv, ns):
        super(GVPLayerNorm, self).__init__()
        self.nv = nv
        self.ns = ns
        self.snorm = nn.LayerNorm(ns)
    def forward(self, x):
        v, s = split(x, self.nv)
        vn = norm_no_nan(v, dim=-2, keepdim=True, sqrt=False) # [..,1, nv]
        vn = torch.sqrt(torch.mean(vn, dim=-1, keepdim=True))
        return merge(v/vn, self.snorm(s))

# [..., 3*nv + ns] -> [..., 3, nv], [..., ns]
# nv = number of vector channels
# ns = number of scalar channels
# vector channels are ALWAYS at the top!
def split(x, nv):
    v = torch.reshape(x[..., :3*nv], list(x.shape[:-1]) + [3, nv])
    s = x[..., 3*nv:]
    return v, s

# [..., 3, nv], [..., ns] -> [..., 3*nv + ns]
def merge(v, s):
    v = torch.reshape(v, list(v.shape[:-2]) + [3*v.shape[-1]])
    return torch.cat([v, s], dim=-1)

# Concat in a way that keeps vector channels at the top
def vs_concat(x1, x2, nv1, nv2):
    
    v1, s1 = split(x1, nv1)
    v2, s2 = split(x2, nv2)
    
    v = torch.cat([v1, v2], -1)
    s = torch.cat([s1, s2], -1)
    return merge(v, s)




# common layers using GVP

class GVPNodeLayer(nn.Module):
    def __init__(self, nv, ns, ev, es, dropout=0.1):
        super(GVPNodeLayer, self).__init__()
        self.nv, self.ns, self.ev, self.es = nv, ns, ev, es
        self.norm = nn.ModuleList([GVPLayerNorm(nv, ns) for _ in range(2)])
        self.dropout = GVPDropout(dropout, nv)

        # this receives the vec_in message AND the receiver node
        self.W_EV = nn.Sequential(
                            GVP(vi=ev, vo=ev, si=es, so=es),
                            GVP(vi=ev, vo=ev, si=es, so=es),
                            GVP(vi=ev, vo=nv, si=es, so=ns, nls=None, nlv=None)
                        )

        self.W_dh = nn.Sequential(
                            GVP(vi=nv, vo=2*nv, si=ns, so=4*ns),
                            GVP(vi=2*nv, vo=nv, si=4*ns, so=ns, nls=None, nlv=None)
                        )

    def forward(self, h_V, h_EV, mask_V=None, mask_attend=None):
        # Concatenate h_V_i to h_E_ij
        h_message = self.W_EV(h_EV)

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.mean(h_message, -2)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.W_dh(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

class GVPEdgeLayer(nn.Module):
    def __init__(self, nv, ns, ev, es, dropout=0.1):
        super(GVPEdgeLayer, self).__init__()
        self.nv, self.ns, self.ev, self.es = nv, ns, ev, es
        self.norm = nn.ModuleList([GVPLayerNorm(nv, ns) for _ in range(2)])
        self.dropout = GVPDropout(dropout, nv)

        # this receives the vec_in message AND the receiver node
        self.W_EV = nn.Sequential(
                            GVP(vi=ev, vo=ev, si=es, so=es),
                            GVP(vi=ev, vo=ev, si=es, so=es),
                            GVP(vi=ev, vo=nv, si=es, so=ns, nls=None, nlv=None)
                        )

        self.W_dh = nn.Sequential(
                            GVP(vi=nv, vo=2*nv, si=ns, so=4*ns),
                            GVP(vi=2*nv, vo=nv, si=4*ns, so=ns, nls=None, nlv=None)
                        )

    def forward(self, h_E, h_EV, mask_E=None, mask_attend=None):
        # Concatenate h_V_i to h_E_ij
        dh = self.W_EV(h_EV)
        if mask_attend is not None:
            dh = mask_attend.unsqueeze(-1) * dh
        h_E = self.norm[0](h_E + self.dropout(dh))

        # Position-wise feedforward
        dh = self.W_dh(h_E)
        h_E = self.norm[1](h_E + self.dropout(dh))

        if mask_E is not None:
            mask_E = mask_E.unsqueeze(-1)
            h_E = mask_E * h_E
        return h_E

def merge_duplicate_term_edges(h_E_update, E_idx):
    dev = h_E_update.device
    n_batch, n_terms, n_aa, n_neighbors, hidden_dim = h_E_update.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_terms, n_aa, n_aa, hidden_dim)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, -1, hidden_dim).to(dev)
    collection.scatter_(3, neighbor_idx, h_E_update)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(2,3)
    # gather reverse edges
    reverse_E_update = gather_term_edges(collection, E_idx)
    # average h_E_update and reverse_E_update at non-zero positions
    merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update)/2, h_E_update)
    return merged_E_updates
