# Leant_Laplacian

This is an implementation of the paper Learning Laplacian Matrix in Smooth Graph Signal Representations

https://arxiv.org/pdf/1406.7842.pdf by [X.Dong et al]

min tr(Y.*L*Y.T) + beta ||L||^2 \\
         L \\
        L = L.T, tr(L) = number of signals (vertices)., sum(L[i,:]) = 0., L[i,j] <= 0 (i><j) 


Only require numpy, this implementation can deal in the case of large number of vertices (num_vertices > 1000). 

Contact : khanhhung92vt@gmail.com
