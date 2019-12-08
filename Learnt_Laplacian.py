# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:49:57 2019

@author: admin-local
"""

import numpy as np

""" Learning Laplacian matrix L as described in Structure-aware classification
    using supervised dictionary learning. X. Dong
        
        min tr(Y.*L*Y.T) + beta ||L||^2
         L
        L = L.T, tr(L) = number of signals (vertices)., sum(L[i,:]) = 0., L[i,j] <= 0 (i><j) 

"""

def prox_negative_and_sum_constraint(y,c):
    """ 
    Projection to get all element of y are negative and sum of all element of y is c
    Used in splitting method
    """

    n = len(y)
    k =(c - np.sum(y))/float(n)
    x_0 = y + k
    while len(np.where(x_0 > 0)[0]) != 0:
        idx_negative = np.where(x_0 < 0)[0]
        idx_positive = np.where(x_0 > 0)[0]
        n_0 = len(idx_negative)
        y_0 = x_0[idx_negative]
        k_0 =(c - np.sum(y_0))/float(n_0)
        x_0[idx_negative] = x_0[idx_negative] + k_0
        x_0[idx_positive] = 0.

    return x_0

def FWBW_proximal_gradient_in_sparse_coding(num_vertices,coef_b,beta,init_solu,ite_max,verbose = False):
    """ 
    beta * ||X||^2 + b*vec(X_reduit)
    X_reduit is diagonal and lower diagonal element of X
    """
    
    """ 
    beta_dia = beta (diagonal)
    beta_late (lateral) = 2 * beta
    """
    beta_late = 2 * beta
    beta_dia = beta
    if verbose :
        err = np.zeros(ite_max)
    index_diagonal = [] 
    for i in range (num_vertices):
        j = int(i*num_vertices-i*(i-1)/2)
        index_diagonal.append(j)

    b_reduit = np.zeros(int(num_vertices*(num_vertices-1)/2))
    """ b_reduit : coef premier ordre for lower diagonal elements 
    so now we deal with num_vertices - 1, we recalculate coef from b
    """
    num_vertices_reduit = num_vertices-1
        
    for i in range (num_vertices):
        index_in_b = []
        for j in range(num_vertices):
            if j < i:
                index_in_b.append(int((2*num_vertices-j+1)*j/2+i-j))
            if j > i:   
                index_in_b.append(int((2*num_vertices-i+1)*i/2+j-i))
        
        
        index_in_b_reduit = []
        for j in range(num_vertices):
            if j < i:
                index_in_b_reduit.append(int((2*num_vertices_reduit-j+1)*j/2+i-j-1))
            if j > i:   
                index_in_b_reduit.append(int((2*num_vertices_reduit-i+1)*i/2+j-i-1))
        
        b_reduit[index_in_b_reduit] = b_reduit[index_in_b_reduit] + coef_b[index_in_b]/2. - coef_b[index_diagonal[i]]


    n = len(b_reduit)
    c =-float(num_vertices)/2.
    
    """ index_depend to calculate gradient in next step,
    index_depend between diagonal element and lateral element
    """
    
    index_depend = np.empty((num_vertices-1,num_vertices), dtype = int)
    for k in range (num_vertices): 
        index_in_b_reduit = []
        for j in range(num_vertices):
            if j < k:
                index_in_b_reduit.append(int((2*num_vertices_reduit-j+1)*j/2+k-j-1))
            if j > k:   
                index_in_b_reduit.append(int((2*num_vertices_reduit-k+1)*k/2+j-k-1))
        index_depend[:,k] = np.array(index_in_b_reduit)
    
    """ Splitting Method """
        
    beta_lipschitzienne = np.sqrt(n*((2*beta_late + 4*beta_dia)**2 + 2*(num_vertices-2)*(2*beta_dia)**2))
    taux =  1./float(beta_lipschitzienne)
    

    x_new = np.copy(init_solu)
    
    for i in range (ite_max):
       
        x_old = x_new   
        
        grad_2 = np.zeros(n)    
        for k in range (num_vertices): 
            grad_2[index_depend[:,k]] = grad_2[index_depend[:,k]] + 2 * np.sum(x_new[index_depend[:,k]])    
         
        gradA = b_reduit + 2 * beta_late * x_new + beta_dia * grad_2        
             
        y_n = x_new - taux*gradA
        
        """ Projection to satisfy constrains """
        x_new =  prox_negative_and_sum_constraint(y_n,c)
                 
        if verbose:
            err[i]= b_reduit.dot(x_new) + beta_late*np.linalg.linalg.norm(x_new)**2 + beta_dia*np.sum(np.sum(x_new[index_depend],axis = 0)**2 )

        if np.max(np.abs(x_new - x_old)) < 0.0001 :
            if verbose :
                err = err[:i]
                return x_new,err
            else:
                return x_new
            break                    
             
      
    if i == (ite_max - 1):
        print("number iterative max reached")
    if verbose :
        return x_new,err
    else:
        return x_new

def lateral_add_dia_2_matrix(x,num_vertices):
    """
    Generate matrix L from L_vec
    """
    
    L = np.zeros((num_vertices,num_vertices))
    a = 0
    b = num_vertices-1
    for i in range(num_vertices-1):
        L[i+1:,i] = x[a:b]
        L[i,i+1:] = x[a:b]
        L[i,i] = -np.sum(L[:,i])
        a = b
        b = b + num_vertices - i -2
    L[num_vertices-1,num_vertices-1] = -np.sum(L[:,num_vertices-1])
    
    return L            

def Construct_leanrt_Laplacian(Y,beta,ite_max = 5000,verbose = False):
    
    """ Learning Laplacian matrix L as described in Structure-aware classification
    using supervised dictionary learning. X. Dong
        
        min tr(Y.*L*Y.T) + beta ||L||^2
         L
        L = L.T, tr(L) = number of signals (vertices) , sum(L[i,:]) = 0., L[i,j] <= 0 (i><j) 
    """
    
    num_vertices = Y.shape[1]
    DDT = np.dot(Y.T,Y)
    coef_b = np.zeros(int(num_vertices*(num_vertices+1)/2))
    """ coef_b : coefficients correspond to diagonal and lower diagonal of matrix L (L_reduit)"""
    """ The following code try to express tr(Y_all*L*Y_all.T) = coef_b * vec(L_reduit) """
    a = 0
    b = num_vertices
    for i in range (DDT.shape[0]-1):
        coef_b[a] = DDT[i,i]
        a = a + 1
        coef_b[a:b] = DDT[i+1:,i] + DDT[i,i+1:] 
        a = b
        b = b + num_vertices - i -1
    coef_b[-1] = DDT[-1,-1]    
        
    init_solu = np.zeros(int(num_vertices*(num_vertices-1)/2))
    
    """ Optimize by Splitting Method (Forward Backward) """
    if verbose :
        L_vec,e = FWBW_proximal_gradient_in_sparse_coding(num_vertices, coef_b,beta,init_solu,ite_max,True)
    else : 
        L_vec = FWBW_proximal_gradient_in_sparse_coding(num_vertices, coef_b,beta,init_solu,ite_max,False)
    """ Rewrite L from its diagonal and lower diagonal element """
    L = lateral_add_dia_2_matrix(L_vec,num_vertices)
    
    if verbose :
        return L,coef_b,L_vec,e
    else :
        return L,coef_b,L_vec
    
""" Example """

if __name__ == "__main__":
    signals = np.random.rand(100,1000) # 100 features for each sample and 1000 samples
    beta = 1.
    L,_,_ = Construct_leanrt_Laplacian(signals,beta,ite_max = 5000,verbose = False)