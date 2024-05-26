import torch
from torch import nn
import math
import numpy as np
#from rdkit import DataStructs, Chem
import torch.nn.functional as F
#from rdkit.Chem import AllChem
from tqdm import tqdm
import copy

def _diffusion(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for diffusion operator
    '''
    coeff = coeff.unsqueeze(-1).unsqueeze(-1)
    dt = dt.unsqueeze(-1).unsqueeze(-1)
    dx = dx.unsqueeze(-1).unsqueeze(-1)

    diff_u = (torch.roll(u, shifts=(1), dims=(1)) + torch.roll(u, shifts=(-1), dims=(1)) - 2*u)
    diff_u = coeff*dt*diff_u/dx

    return diff_u
    

def _advection(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for nonlinear advection operator
    '''
    coeff = coeff.unsqueeze(-1).unsqueeze(-1)
    dt = dt.unsqueeze(-1).unsqueeze(-1)
    dx = dx.unsqueeze(-1).unsqueeze(-1)

    diff_u = (torch.roll(u, shifts=(1), dims=(1)) - torch.roll(u, shifts=(-1), dims=(1))) * u
    diff_u = coeff * dt * diff_u/(2 * dx)

    return diff_u
    
    
def _dispersion(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for third derivative operator
    '''
    numerator =  torch.roll(u, shifts=(3), dims=(1)) - \
               8*torch.roll(u, shifts=(2), dims=(1)) + \
              13*torch.roll(u, shifts=(1), dims=(1)) - \
              13*torch.roll(u, shifts=(-1), dims=(1)) + \
               8*torch.roll(u, shifts=(-2), dims=(1)) - \
                 torch.roll(u, shifts=(-2), dims=(1))
    diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) 
    return diff_u


def _linear_advection(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for third derivative operator
    '''
    coeff = coeff.unsqueeze(-1).unsqueeze(-1)
    dt = dt.unsqueeze(-1).unsqueeze(-1)
    dx = dx.unsqueeze(-1).unsqueeze(-1)

    numerator = u - torch.roll(u, shifts=(-1), dims=(1))
    adv_u = -coeff * dt * numerator / dx
    return adv_u


def _anchored_l2(xi, xj, ui, uj, dt, dx, coeff):

    # Do finite difference update on initial condition if that's the only input
    if(ui.shape[2] == 1):
        u_adv = _advection(uj, dt, dx, coeff[:,0])
        u_diff = _diffusion(uj, dt, dx, coeff[:,1])
        u_dis = _linear_advection(uj, dt, dx, coeff[:,2])
        uj += u_adv + u_diff + u_dis
        us = uj[...,-1].unsqueeze(0) - ui[...,-1].unsqueeze(1)
    else:
        us = (ui[...,-1].unsqueeze(0) - uj[...,-2].unsqueeze(1)) # Temporal evolution

    # Calculate physics-informed update
    adv = _advection(xi, dt, dx, coeff[:,0])
    diff = _diffusion(xi, dt, dx, coeff[:,1])
    dis = _linear_advection(xi, dt, dx, coeff[:,2])

    updates = (adv + diff + dis)[...,0]
    physics_distance = ((xi[...,0] + updates).unsqueeze(1) - xj[...,0].unsqueeze(0))

    norm = ((us - physics_distance).norm(dim=-1))**2
    if(norm.isnan().any()):
        raise ValueError("\n\n\nNAN IN DISTANCE FUNCTION\n\n\n")
    return norm


def _diffusion_2d(u, dt, dx, coeff):
    diff_ux = (torch.roll(u, shifts=(1), dims=(1)) + torch.roll(u, shifts=(-1), dims=(1)) - 2*u)
    diff_uy = (torch.roll(u, shifts=(1), dims=(2)) + torch.roll(u, shifts=(-1), dims=(2)) - 2*u)
    diff_u = coeff.view(coeff.shape[0], 1, 1, 1)*(diff_ux + diff_uy)/dx**2
    return diff_u
    
    #print(diff_u.shape)
    #raise
    #pass


def _advection_2d(u, dt, dx, coeff):
    # Calculate finite differences
    un = u.clone()
    vn = u.clone()

    # Calculate finite differences for nonlinear advection term
    # TODO: Vectorize this if possible
    adv_u = torch.zeros(u.shape)
    for idx, (cx, cy) in enumerate(coeff):
        cx = cx.view(1,1,1,1)
        cy = cy.view(1,1,1,1)
        if(cx <= 0 and cy >= 0):
            adv_u[idx] = -cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(1))) + \
                          cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(0)))
            #adv_v = cy*vn*(vn - torch.roll(vn, shifts=(1), dims=(0))) - cx*un*(vn - torch.roll(vn, shifts=(-1), dims=(1)))
        elif(cx >= 0 and cy >= 0):
            adv_u[idx] = cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(1))) + \
                         cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(0)))
            #adv_v = cy*vn*(vn - torch.roll(vn, shifts=(1), dims=(0))) + cx*un*(vn - torch.roll(vn, shifts=(1), dims=(1)))
        elif(cx <= 0 and cy <= 0):
            adv_u[idx] = -cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(1))) - \
                          cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(0)))
            #adv_v = -cy*vn*(vn - torch.roll(vn, shifts=(-1), dims=(0))) - cx*un*(vn - torch.roll(vn, shifts=(-1), dims=(1)))
        elif(cx >= 0 and cy <= 0):
            adv_u[idx] = cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(1))) - \
                         cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(0)))
            #adv_v = -cy*vn*(vn - torch.roll(vn, shifts=(-1), dims=(0))) + cx*un*(vn - torch.roll(vn, shifts=(1), dims=(1)))

    return -dt*adv_u/dx


def _anchored_l2_2d(xi, xj, ui, uj, dt, dx, coeff):

    # Temporal evolution
    us = (ui[...,-1].unsqueeze(0) - uj[...,-2].unsqueeze(1)).norm(dim=(-2,-1)) # Temporal evolution

    difference = ((xi[...,-1] - ui[...,-1]).unsqueeze(0) - \
                  (xj[...,-1] - uj[...,-1]).unsqueeze(1)).norm(dim=-1)

    # Calculate physics-informed update
    #print(coeff)
    diff = _diffusion_2d(xi, dt, dx, coeff[:,0])
    adv = _advection_2d(xi, dt, dx, coeff[:,1:])
    #advy = _advection(xi, dt, dx, coeff[:,0])
    #dis = _dispersion(xi, dt, dx, coeff[:,2])

    if(diff.isnan().any()):
        print("ISSUE WITH DIFF")
    if(adv.isnan().any()):
        print("ISSUE WITH ADV")
    updates = (adv + diff)[...,0]

    # Conservation of mass
    u_mass = ui[...,0].sum(dim=1)
    x_mass = xi.sum(dim=1)[...,0]

    # Homogeneous case
    #mass = (u_mass.unsqueeze(0) - x_mass.unsqueeze(1))*dx

    # Operator distance TODO: This is not what I want it to be...
    #physics_distance = (updates.unsqueeze(1) - updates.unsqueeze(0)).norm(dim=(-2,-1))
    #print(xi.shape)
    #print(updates.shape)
    physics_distance = ((xi[...,0] + updates).unsqueeze(1) - xj[...,0].unsqueeze(0)).norm(dim=(-2,-1))
    #print()
    #print()
    #print(updates.isnan().any())
    #print(physics_distance)
    #print(us.isnan().any())
    #print(us)
    #print(physics_distance.shape)
    #print()
    #print()

    #return (us - physics_distance)**2 + mass**2 + difference**2  # Full
    #return (us - physics_distance)**2 + difference**2            # No mass
    #return (us - physics_distance)**2 + mass**2                  # No pred
    #return mass**2 + difference**2                                # No Physics
    #print((0.1*us - physics_distance)**2)
    return (us - physics_distance)**2                            # Just Physics
    #return mass**2                                                # Just mass
    #return difference**2                                          # Just pred


class WeightedNTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature=1., similarity='cosine', lambda_1=0.9, **kwargs):
        super(WeightedNTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        #self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.similarity_function = self._get_similarity_function(similarity)
        self.lambda_1 = lambda_1
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, similarity):
        #return torch.cdist
        #f = lambda xi, xj, ui, uj: (frdist(xi, xj) - frdist(ui, uj)).abs()
        if(similarity == 'cosine'):
            print("\nUSING COSINE SIMILARITY\n")
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        elif(similarity == 'anchored_l2'):
            print("\nUSING ANCHORED L2 SIMILARITY\n")
            return _anchored_l2
        elif(similarity == 'dot'):
            print("\nUSING DOT SIMILARITY\n")
            return self._dot_simililarity
        else:
            raise ValueError("Invalid choice of similarity function.")
        #if use_cosine_similarity:
        #    self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        #    return self._cosine_simililarity
        #else:
        #    return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        #diag = np.eye(batch_size)
        #l1 = np.eye((batch_size), batch_size, k=-batch_size)
        #l2 = np.eye((batch_size), batch_size, k=batch_size)
        #mask = torch.from_numpy((diag + l1 + l2))
        #mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        print(v.shape)
        raise
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        #print(x.shape)
        #print(y.shape)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        #print(v.shape)
        return v


    def _anchored_l2(self, xi, xj, ui, uj, dt, dx, coeff):
        xs = (xi[...,0].unsqueeze(0) - xj[...,0].unsqueeze(1)).norm(dim=-1)#[...,0]
        us = (ui.unsqueeze(0) - uj.unsqueeze(1)).norm(dim=-1)#[...,0]

        adv = self._advection(xi, dt, dx, coeff[:,0])
        diff = self._diffusion(xi, dt, dx, coeff[:,1])
        dis = self._dispersion(xi, dt, dx, coeff[:,2])

        updates = (adv + diff + dis)[...,0]
        operator_distance = (updates.unsqueeze(1) - updates.unsqueeze(0)).norm(dim=-1)
        #print((xs - us).abs().shape, operator_distance.shape)
        #raise
        return -((xs - us).abs() + operator_distance)


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        diff_u = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 
        return diff_u
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) 
        return diff_u
        
        
    def _dispersion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) 
        return diff_u


    def _pde_similarity(self, u, dt, dx, target):
        
        u_adv = self._advection(u, dt, dx, target[:,0].unsqueeze(-1))
        u_diff = self._diffusion(u, dt, dx, target[:,1].unsqueeze(-1))
        u_third = self._third_order(u, dt, dx, target[:,2].unsqueeze(-1))

        xi = torch.cat((u_adv, u_diff, u_third), dim=1)
        xj = torch.cat((u_adv, u_diff, u_third), dim=1)

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat


    def _pde_similarity(self, target):
        xi = target.clone()
        xj = target.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]
        #print(prod_mat.shape)
        #print(norm_mat.shape)
        #print(prod_mat[0][0], norm_mat[0][0])
        #print(prod_mat)
        #print(norm_mat)
        #raise
        return prod_mat / norm_mat


    def forward(self, x1, x2, u, dt, dx, vs):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim = self._pde_similarity(vs)
        sim_score = 1 - self.lambda_1 * sim
        sim_score = torch.nan_to_num(sim_score, nan=0.)
        sim_score = sim_score.masked_select(~torch.eye(batch_size, dtype=bool)).view(batch_size, batch_size - 1)
        sim_score = sim_score.repeat(2,2)

        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #similarity_matrix = self.similarity_function(representations[...,0],
                                                     #representations[...,0])
        #similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
                                                     #representations.flatten(1,2))#[...,0])

        #similarity_matrix = self.similarity_function(x1.flatten(1,2), x2.flatten(1,2))
        #print()
        #print()
        #print(x1.shape, u[0].shape)
        #print()
        #print()
        if(len(x1.shape) == 4):
            x1 = x1[...,0]
        if(len(x2.shape) == 4):
            x2 = x2[...,0]

        u = torch.cat((u,u), dim=0)
        dt = torch.cat((dt,dt), dim=0)
        vs = torch.cat((vs,vs), dim=0)
        similarity_matrix = self.similarity_function(representations.flatten(1,2), representations.flatten(1,2),
                                                     u.clone(), u.clone(), dt, dx, vs)
                                                     #u[...,0].clone(), u[...,0].clone(), dt, dx, vs)

        #print(similarity_matrix.shape)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        #print(positives)
        
        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #print(mask_samples_from_same_repr.shape)
        #print(similarity_matrix.shape)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        #print(negatives.shape)
        #print(sim_score.shape)
        negatives *= sim_score
        
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        
        return loss / (2 * batch_size)
        #similarity_matrix = torch.zeros(batch_size, batch_size)
        #for i in range(len(x1)):
        #    for j in range(i, len(x2)):
        #        sim = self.similarity_function(x1[i], x1[j], u[i], u[j])
        #        similarity_matrix[i,j] = sim
        #        similarity_matrix[j,i] = sim

        #similarity_matrix = self.similarity_function(x1.flatten(1,2), x2.flatten(1,2),
        #                                             u.flatten(1,2).clone(), u.flatten(1,2).clone())
        #similarity_matrix = torch.Tensor(similarity_matrix)
        #print(similarity_matrix)
        #print(representations.shape)
        #print(x1.shape)
        #print(x2.shape)
        #print(similarity_matrix.shape)
        #print(l_pos)
        #print(similarity_matrix.shape)
        #raise
        #l_pos = torch.diag(similarity_matrix, batch_size)
        #r_pos = torch.diag(similarity_matrix, -batch_size)
        l_pos = torch.diag(similarity_matrix)#, batch_size)
        #r_pos = torch.diag(similarity_matrix)#, -batch_size)
        #print(l_pos)
        #print(r_pos)
        #print(l_pos.shape)
        #print(r_pos.shape)
        #positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        #positives = torch.cat([l_pos, r_pos]).unsqueeze(-1)#.view(batch_size, 1)
        positives = l_pos.clone().unsqueeze(-1)
        #print(x1)
        #print(x2)
        #print(similarity_matrix)
        #raise

        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(batch_size, -1)
        #print(negatives.shape)
        #print(sim_score)
        #print(negatives)
        #print(positives)
        #raise
        #negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)
        #print(sim_score.shape)
        #print(sim_score)
        negatives *= sim_score[mask_samples_from_same_repr].view(batch_size, -1)
        #raise

        #print(positives.shape)
        #print(negatives.shape)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        #print(logits)

        #labels = torch.zeros(2 * batch_size).to(self.device).long()
        labels = torch.zeros(batch_size).to(self.device).long()
        #print(labels)
        #print(logits)
        loss = self.criterion(logits, labels)
        #print(logits)
        #print(loss)
        return loss/batch_size
        #raise

        # filter out the scores from the positive samples
        #print(similarity_matrix)

        # filter out the scores from the positive samples
        #l_pos = torch.diag(similarity_matrix, batch_size)
        #r_pos = torch.diag(similarity_matrix, -batch_size)
        #positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        #negatives *= sim_score

        #logits = torch.cat((positives, negatives), dim=1)
        #logits /= self.temperature

        #labels = torch.zeros(2 * batch_size).to(self.device).long()
        #loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class PhysicsInformedWNTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature=0.1, use_cosine_similarity=True, lambda_1=0.9999, **kwargs):
        super(PhysicsInformedWNTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.lambda_1 = lambda_1
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        print("\nUSING DOT SIMILARITY\n")
        return self._dot_simililarity
        #return torch.cdist
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity


    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        diff_u = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        
        
    def _third_order(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        #numerator = torch.roll(u, shifts=(0,2), dims=(0,1)) - \
        #           2*torch.roll(u, shifts=(0,1), dims=(0,1)) + \
        #           2*torch.roll(u, shifts=(0,-1), dims=(0,1)) - \
        #             torch.roll(u, shifts=(0,-2), dims=(0,1))
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        #
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)


    def _pde_similarity(self, u, dt, dx, target):
        
        u_adv = self._advection(u, dt, dx, target[:,0].unsqueeze(-1))
        u_diff = self._diffusion(u, dt, dx, target[:,1].unsqueeze(-1))
        u_third = self._third_order(u, dt, dx, target[:,2].unsqueeze(-1))
        #print(target[:,2])
        #print()
        #print(u_third)
        #print()

        xi = torch.cat((u_adv, u_diff, u_third), dim=1)
        xj = torch.cat((u_adv, u_diff, u_third), dim=1)
        #print(xi)
        #print(target)
        #raise

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        #print()
        #print(prod_mat)
        #print(norm_mat)
        #print(target)
        #print(xi)
        #print(prod_mat/norm_mat)
        #print()
        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        #diag = np.eye(2 * batch_size)
        #l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        #l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        #mask = torch.from_numpy((diag + l1 + l2))
        #mask = (1 - mask).type(torch.bool)
        diag = np.eye(batch_size)
        l1 = np.eye((batch_size), batch_size, k=-batch_size)
        l2 = np.eye((batch_size), batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        #sim_score = np.zeros((batch_size, batch_size-1))
        #for i in range(len(vs)):
        #    for j in range(i+1, len(vs)):
        #        sim = self._similarity(vs[i], vs[j])
        #        sim_score[i,j-1] = sim
        #        sim_score[j,i] = sim

        #sim_score = 1 - self.lambda_1 * torch.tensor(sim_score, dtype=torch.float).to(x1.device)
        #sim_score = sim_score.repeat(2, 2)
        #sim_score = torch.tensor(sim_score, dtype=torch.float).repeat(2, 2)


        #print(vs[:4])
        #sim = self._pde_similarity(vs)
        sim = self._pde_similarity(u, dt, dx, vs)
        #print(sim)
        #print(sim.max())
        #print(sim.min())
        #raise
        #print(sim)
        sim_score = 1 - self.lambda_1 * sim
        #sim_score = sim.repeat(2, 2)
        sim_score = torch.nan_to_num(sim_score, nan=0.)
        #print(sim_score)
        #raise


        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #similarity_matrix = self.similarity_function(representations[...,0],
                                                     #representations[...,0])
        #similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
        #                                             representations.flatten(1,2))#[...,0])
        similarity_matrix = self.similarity_function(x1.flatten(1,2), x2.flatten(1,2))
        #print(representations.shape)
        #print(x1.shape)
        #print(x2.shape)
        #print(similarity_matrix.shape)
        #print(l_pos)
        #print(similarity_matrix.shape)
        #raise
        #l_pos = torch.diag(similarity_matrix, batch_size)
        #r_pos = torch.diag(similarity_matrix, -batch_size)
        l_pos = torch.diag(similarity_matrix)#, batch_size)
        #r_pos = torch.diag(similarity_matrix)#, -batch_size)
        #print(l_pos)
        #print(r_pos)
        #print(l_pos.shape)
        #print(r_pos.shape)
        #positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        #positives = torch.cat([l_pos, r_pos]).unsqueeze(-1)#.view(batch_size, 1)
        positives = l_pos.clone().unsqueeze(-1)
        #print(x1)
        #print(x2)
        #print(similarity_matrix)
        #raise

        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(batch_size, -1)
        #print(negatives.shape)
        #print(sim_score)
        #print(negatives)
        #raise
        #negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)
        #print(sim_score.shape)
        #print(sim_score)
        negatives *= sim_score[mask_samples_from_same_repr].view(batch_size, -1)
        #raise

        #print(positives.shape)
        #print(negatives.shape)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        #labels = torch.zeros(2 * batch_size).to(self.device).long()
        labels = torch.zeros(batch_size).to(self.device).long()
        #print(labels)
        #print(logits)
        loss = self.criterion(logits, labels)
        #print(logits)
        #print(loss)
        return loss/batch_size
        #raise

        # filter out the scores from the positive samples
        #print(similarity_matrix)

        # filter out the scores from the positive samples
        #l_pos = torch.diag(similarity_matrix, batch_size)
        #r_pos = torch.diag(similarity_matrix, -batch_size)
        #positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        #negatives *= sim_score

        #logits = torch.cat((positives, negatives), dim=1)
        #logits /= self.temperature

        #labels = torch.zeros(2 * batch_size).to(self.device).long()
        #loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class PhysicsInformedGCL(torch.nn.Module):
    def __init__(self, device, tau=0.5, use_cosine_similarity=True, **kwargs):
        super(PhysicsInformedGCL, self).__init__()
        #self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.tau = tau
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        print("\nUSING COSINE SIMILARITY\n")
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity


    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        diff_u = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(0.01*(u[...,1] - diff_u[...,0]).abs(), dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0])
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(0.01*(u[...,1] - coeff*diff_u[...,0]).abs(), dim=1)**2)))
        #print(torch.exp(-(torch.sum(0.01*(u[...,1] - coeff*diff_u[...,0]).abs(), dim=1)**2)))
        #raise
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(0.01*(u[...,1] - diff_u[...,0]).abs(), dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        
        
    def _third_order(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        #numerator = torch.roll(u, shifts=(0,2), dims=(0,1)) - \
        #           2*torch.roll(u, shifts=(0,1), dims=(0,1)) + \
        #           2*torch.roll(u, shifts=(0,-1), dims=(0,1)) - \
        #             torch.roll(u, shifts=(0,-2), dims=(0,1))
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        #
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) + u
        #print(u[...,1])
        #print(diff_u[...,0])
        #print(torch.sum((u[...,1] - diff_u[...,0]).abs(), dim=1))
        return (coeff[:,0]*torch.exp(-(torch.sum(0.01*(u[...,1] - diff_u[...,0]).abs(), dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)


    def _pde_similarity(self, u, dt, dx, target):
        
        u_adv = self._advection(u, dt, dx, target[:,0].unsqueeze(-1))
        u_diff = self._diffusion(u, dt, dx, target[:,1].unsqueeze(-1))
        u_third = self._third_order(u, dt, dx, target[:,2].unsqueeze(-1))
        #print(target[:,2])
        #print()
        #print(u_third)
        #print()

        xi = torch.cat((u_adv, u_diff, u_third), dim=1)
        xj = torch.cat((u_adv, u_diff, u_third), dim=1)
        #print(target[:,-1])
        #print(u_third)
        #raise
        #print(target)
        #print(xi)
        #print(xi)
        #print(target)

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        #print()
        #print(prod_mat)
        #print(norm_mat)
        #print(target)
        #print(xi)
        #print(prod_mat/norm_mat)
        #print()
        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = np.zeros((batch_size, batch_size-1))

        sim_score = self._pde_similarity(u, dt, dx, vs)

        representations = x1#torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #print(representations.shape)
        #similarity_matrix = self.similarity_function(representations[...,0],
        #                                             representations[...,0])
        similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
                                                     representations.flatten(1,2))#[...,0])
        #print(similarity_matrix)
        #print(similarity_matrix.shape)

        positive = 0.5 * sim_score * similarity_matrix**2
        #torch.cat(((self.tau-similarity_matrix**2).unsqueeze(0), torch.zeros(sim_score.shape).unsqueeze(0))).shape)
        negative = 0.5 * (1 - sim_score) * torch.max(torch.cat(((self.tau - similarity_matrix**2).unsqueeze(0),
                                                           torch.zeros(sim_score.shape).unsqueeze(0)), axis=0))
        return (positive + negative).mean()
        raise

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        if(loss.isnan()):
            print(similarity_matrix)
            print(sim_score)
            print(logits)

        return loss #/ (2 * batch_size)


class GCL(torch.nn.Module):
    def __init__(self, device, tau=10., use_cosine_similarity=True, **kwargs):
        super(GCL, self).__init__()
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.tau = tau
        self._tau = tau
        print("\nTAU: {}".format(self.tau))
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        #print("\nUSING COSINE SIMILARITY\n")
        #return torch.cdist
        print("\nUSING ANCHORED L2 SIMILARITY\n")
        return _anchored_l2
        #return self._dot_simililarity
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity



    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        diff_u = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 
        return diff_u
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) 
        return diff_u
        
        
    def _dispersion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) 
        return diff_u


    def _pde_similarity(self, u, dt, dx, target):
        
        xi = target.clone()
        xj = target.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0).cuda() * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1).cuda()
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = np.zeros((batch_size, batch_size-1))

        #sim_score = 1 - self._pde_similarity(u, dt, dx, vs)
        sim_score = self._pde_similarity(u, dt, dx, vs).cuda()
        sim_score = sim_score.repeat(2,2)

        #representations = x1#torch.cat([x2, x1], dim=0)
        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #print(representations.shape)
        #similarity_matrix = self.similarity_function(representations[...,0],
        #                                             representations[...,0])
        #similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
        #                                             representations.flatten(1,2))#[...,0])
        if(len(x1.shape) == 4):
            x1 = x1[...,0]
        if(len(x2.shape) == 4):
            x2 = x2[...,0]
        #similarity_matrix = self.similarity_function(x1, x2,
        #                                             u[...,0].clone(), u[...,0].clone(), dt, dx, vs)

        u = torch.cat((u,u), dim=0)
        dt = torch.cat((dt,dt), dim=0)
        vs = torch.cat((vs,vs), dim=0)
        similarity_matrix = self.similarity_function(representations.flatten(1,2).cuda(), representations.flatten(1,2).cuda(),
                                                     u.clone().cuda(), u.clone().cuda(), dt.cuda(), dx.cuda(), vs.cuda()).cuda()
        
        positive = 0.5 * sim_score * similarity_matrix**2

        if(self._tau == 'mean'):
            self.tau = similarity_matrix.mean().cuda()
        max_obj = torch.amax(torch.cat(((self.tau - similarity_matrix).unsqueeze(0),
                             torch.zeros(sim_score.shape).unsqueeze(0).cuda()), axis=0), dim=0)**2
        negative = 0.5 * (1 - sim_score) * max_obj
        #print()
        #print()
        #print(negative.shape)
        #print(negative.max())
        #print()
        #print()
        return (positive + negative).mean()/batch_size


class PassthroughGCL(torch.nn.Module):
    def __init__(self, device, tau=10., use_cosine_similarity=True, **kwargs):
        super(PassthroughGCL, self).__init__()
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.tau = tau
        self._tau = tau
        print("\nTAU: {}".format(self.tau))
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        #print("\nUSING COSINE SIMILARITY\n")
        #return torch.cdist
        print("\nUSING ANCHORED L2 Passthrough SIMILARITY\n")
        return None
        #return self._dot_simililarity
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        diff_u = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 
        return diff_u
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) 
        return diff_u
        
        
    def _dispersion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) 
        return diff_u


    def _pde_similarity(self, u, dt, dx, target):
        
        xi = target.clone()
        xj = target.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0).cuda() * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1).cuda()
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = np.zeros((batch_size, batch_size-1))

        #sim_score = 1 - self._pde_similarity(u, dt, dx, vs)
        sim_score = self._pde_similarity(u, dt, dx, vs).cuda()
        sim_score = sim_score.repeat(2,2) # Supports x1 and x2 being different, but we don't use it.

        #representations = x1#torch.cat([x2, x1], dim=0)
        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #print(representations.shape)
        #similarity_matrix = self.similarity_function(representations[...,0],
        #                                             representations[...,0])
        #similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
        #                                             representations.flatten(1,2))#[...,0])
        if(len(x1.shape) == 4):
            x1 = x1[...,0]
        if(len(x2.shape) == 4):
            x2 = x2[...,0]
        #similarity_matrix = self.similarity_function(x1, x2,
        #                                             u[...,0].clone(), u[...,0].clone(), dt, dx, vs)
        u = torch.cat((u,u), dim=0)
        dt = torch.cat((dt,dt), dim=0)
        vs = torch.cat((vs,vs), dim=0)
        #similarity_matrix = self.similarity_function(representations.flatten(1,2).cuda(), representations.flatten(1,2).cuda(),
        #                                             u.clone().cuda(), u.clone().cuda(), dt.cuda(), dx.cuda(), vs.cuda()).cuda()
        #print()
        #print()
        #print(x1.shape, x2.shape)
        #print(representations.shape)
        #similarity_matrix = x1.unsqueeze(0) * x2.unsqueeze(1)
        similarity_matrix = representations.unsqueeze(0) * representations.unsqueeze(1)
        similarity_matrix = similarity_matrix.sum(dim=2)[...,0,0]
        
        positive = 0.5 * sim_score * similarity_matrix**2

        if(self._tau == 'mean'):
            self.tau = similarity_matrix.mean().cuda()
        #negative = 0.5 * (1 - sim_score) * torch.max(torch.cat(((self.tau - similarity_matrix**2).unsqueeze(0),
        #                                                   torch.zeros(sim_score.shape).unsqueeze(0).cuda()), axis=0))
        #print(torch.max(torch.cat(((self.tau - similarity_matrix**2).unsqueeze(0), torch.zeros(sim_score.shape).unsqueeze(0).cuda()), axis=0)))
        positive = 0.5 * sim_score * similarity_matrix**2

        if(self._tau == 'mean'):
            self.tau = similarity_matrix.mean().cuda()
        max_obj = torch.amax(torch.cat(((self.tau - similarity_matrix).unsqueeze(0),
                             torch.zeros(sim_score.shape).unsqueeze(0).cuda()), axis=0), dim=0)**2
        negative = 0.5 * (1 - sim_score) * max_obj
        return (positive + negative).mean()/batch_size


class GCL2D(torch.nn.Module):
    def __init__(self, device, tau=100., use_cosine_similarity=True, **kwargs):
        super(GCL2D, self).__init__()
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.tau = tau
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        #print("\nUSING COSINE SIMILARITY\n")
        #return torch.cdist
        print("\nUSING ANCHORED L2 SIMILARITY\n")
        return _anchored_l2_2d
        #return self._dot_simililarity
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

        
    def _pde_similarity_2d(self, u, dt, dx, target):
        
        # Convert individual advections into total advection
        #print(target)
        #print(target[:,1:].norm(dim=1))
        true_target = torch.zeros((len(target), 2))
        true_target[:,0] = target[:,0]
        true_target[:,1] = target[:,1:].norm(dim=1)
        xi = true_target.clone()
        xj = true_target.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        #print(true_target)
        #raise
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = np.zeros((batch_size, batch_size-1))

        sim_score = self._pde_similarity_2d(u, dt, dx, vs)
        sim_score = sim_score.repeat(2,2)

        #representations = x1#torch.cat([x2, x1], dim=0)
        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #print(representations.shape)
        #similarity_matrix = self.similarity_function(representations[...,0],
        #                                             representations[...,0])
        #similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
        #                                             representations.flatten(1,2))#[...,0])
        if(len(x1.shape) == 4):
            x1 = x1[...,0]
        if(len(x2.shape) == 4):
            x2 = x2[...,0]
        #print()
        #print()
        #print(x1.shape, x2.shape, u.shape)
        #print(dt.shape, dx, vs.shape)
        #print()
        #print()
        #similarity_matrix = self.similarity_function(x1, x2,
        #                                             u[...,0].clone(), u[...,0].clone(), dt, dx, vs)

        u = torch.cat((u,u), dim=0)
        dt = torch.cat((dt,dt), dim=0)
        vs = torch.cat((vs,vs), dim=0)
        similarity_matrix = self.similarity_function(representations, representations,
                                                     u.clone(), u.clone(), dt, dx, vs)
        sim_diag = sim_score.diag()
        sim_mat_diag = similarity_matrix.diag()

        sim_off_diag = sim_score - sim_diag
        sim_mat_off_diag = similarity_matrix - sim_mat_diag
        positive = 0.5 * sim_score * similarity_matrix**2

        #torch.cat(((self.tau-similarity_matrix**2).unsqueeze(0), torch.zeros(sim_score.shape).unsqueeze(0))).shape)
        negative = 0.5 * (1 - sim_score) * torch.max(torch.cat(((self.tau - similarity_matrix**2).unsqueeze(0),
                                                           torch.zeros(sim_score.shape).unsqueeze(0)), axis=0))
        #negative = 0.5 * (1 - sim_off_diag) * torch.max(torch.cat(((((self.tau - sim_mat_off_diag)**2).unsqueeze(0),
        #                                                   torch.zeros(sim_score.shape).unsqueeze(0))), axis=0))
        #print(negative)
        #raise
        return (positive + negative).mean()/batch_size
        #return (positive).mean()
        #return (negative).mean()

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        if(loss.isnan()):
            print(similarity_matrix)
            print(sim_score)
            print(logits)

        return loss #/ (2 * batch_size)


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
