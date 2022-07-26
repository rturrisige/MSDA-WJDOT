#!/usr/bin/env python3

import numpy as np
import torch
from torch.autograd import Function
import ot

def dist(x1, x2):
    """ Compute squared euclidean distance between samples (autograd)
    """
    x1p2 = torch.sum(x1**2, 1)
    x2p2 = torch.sum(x2**2, 1)
    return x1p2.reshape((-1, 1)) + x2p2.reshape((1, -1)) - 2 * torch.mm(x1, torch.transpose(x2,0, 1))


def proj_simplex(v, z=1):
    """w=proj_simplex(v) """
    device=v.device
    n_features = v.shape[0]
    if v.ndimension()==1:
        d1=1
        v=v[:,None]
    else:
        d1=0
    u,indices = torch.sort(v,dim=0,descending=True)
    cssv = torch.cumsum(u,dim=0) - z
    ind = torch.arange(n_features,device=device)[:,None].float() + 1
    cond = u - cssv / ind > 0
    rho=torch.max(torch.arange(n_features,device=device)[:,None]*cond.long(),dim=0)[0]
    theta=cssv[torch.arange(n_features,device=device)[:,None]==rho[None,:]]/(1+rho.float())
    w = torch.max(v - theta[None,:], torch.zeros_like(v))
    if d1:
        return w[:,0]
    else:
        return w
    
class EMDLossFunction(Function):
    """Return the EMD los emd(a,b,M)"""

    @staticmethod
    def forward(ctx, a, b, M):
        
        a2=a.detach().cpu().numpy().astype(np.float64)
        b2=b.detach().cpu().numpy().astype(np.float64)
        M2=M.detach().cpu().numpy().astype(np.float64)
        
        a2/=a2.sum()
        b2/=b2.sum()
        
        G,log=ot.emd(a2,b2,M2,log=True)
        
        G=torch.from_numpy(G).type_as(M)
        grad_a=torch.from_numpy(log['u']).type_as(a)
        grad_b=torch.from_numpy(log['v']).type_as(b)
        grad_M=G
            
        ctx.save_for_backward(grad_a, grad_b, grad_M)
        return torch.sum(G*M)

    @staticmethod
    def backward(ctx, grad_output):
        grad_a0, grad_b0, grad_M0 = ctx.saved_tensors
        grad_a= grad_b= grad_M = None
        
        if ctx.needs_input_grad[0]:
            grad_a=grad_a0
        if ctx.needs_input_grad[1]:
            grad_b=grad_b0
        if ctx.needs_input_grad[2]:
            grad_M=grad_M0            
        return grad_a, grad_b, grad_M


def emd_loss(a,b,M):
    """loss=emd2(a,b,M)"""
    return EMDLossFunction.apply(a,b,M)
    

def sqrtm(M):
    """matrix square root"""
    vp,Vp=torch.symeig(M,eigenvectors=True)
    vp2=torch.sqrt(vp)
    return Vp.mm(torch.diag(vp2).mm(Vp.transpose(0,1)))

def bures_wass_loss(xs,xt,ws=None,wt=None,reg=1e-8):
    """ computes the bures loss """
    ns=xs.shape[0]
    nt=xt.shape[0]
    
    if ws is None:
        ws=torch.ones(ns,device=xs.device)/ns
    if wt is None:
        wt=torch.ones(nt,device=xt.device)/nt
    
    ms=ws[None,:].mm(xs)
    mt=wt[None,:].mm(xt)
    
    xsc=xs-ms
    xtc=xt-mt

    Cs=(xsc.transpose(0,-1)*ws[None,:]).mm(xsc)
    Ct=(xtc.transpose(0,-1)*wt[None,:]).mm(xtc)
    
    Cs+=reg*torch.eye(Cs.shape[0],device=Cs.device)
    Ct+=reg*torch.eye(Ct.shape[0],device=Ct.device)
    
    Cs2=sqrtm(Cs)
    
    bures=torch.trace(Cs)+torch.trace(Ct)-2*torch.trace(sqrtm(Cs2.mm(Ct.mm(Cs2))))
    
    bures_wass=torch.sum((ms-mt)**2)+bures
    return bures_wass

def cost_emd_jdot(alpha, xy_all, x, fx, Tn, ws, target_reg=1):
    """ computes the exact ot on the product space """
    # joint dist
    xy = torch.cat((x, target_reg*fx), 1)
    wt = Tn.mv(alpha)
    M = dist(xy, xy_all)
    return emd_loss(ws, wt, M)


def cost_bures_jdot(alpha, xy_all, x, fx, Tn, ws, target_reg=1, bures_reg=0.3):
    """ computes the bures loss on the product space """
    # joint dist
    xy = torch.cat((x, target_reg*fx), 1)
    wt = Tn.mv(alpha)
    return bures_wass_loss(xy, xy_all, ws, wt, reg=bures_reg)



