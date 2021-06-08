import numpy as np

class MultinomialHMM(object):

    def __init__(self, dist_init, n_iter):

        self.dist_init = dist_init
        self.n_iter = n_iter
  
    def log_likelihood_(self, O, A, B):
        alpha = self.forward_algorithm(O, A, B)
        ll = np.log(alpha.sum(axis=1))[-1]
        return ll

    def forward_algorithm(self, O, A, B):
        T = len(O)
        N = A.shape[0]

        alpha = np.zeros((T, N))
        
        alpha[0] = self.dist_init * B[:, O[0]]

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = alpha[t-1] @ (A[:, j] * B[j, O[t]]) # probability that both states lead to j lead to O (therefore @, sumproduct)
        
        return alpha

    def viterbi(self, O, A, B):
        T = O.shape[0]
        N = A.shape[0]
    
        omega = np.zeros((T, N))
        
        omega[0] = np.log(self.dist_init * B[:, O[0]]) # probability that each state lead to first observation
    
        backpointer = np.zeros((T-1, N)) #  saves most probable states, first row stays empty
    
        for t in range(1, T):
            for j in range(N):

                probability = omega[t-1] + np.log(A[:, j]) + np.log(B[j, O[t]]) 
                backpointer[t-1, j] = np.argmax(probability)
                omega[t, j] = np.max(probability)

        q = np.zeros(T)
    
        last_state = np.argmax(omega[T - 1]) 
        q[0] = last_state
    
        backtrack_index = 1
        for i in list(reversed(range(0, T-1))):
            q[backtrack_index] = backpointer[i, int(last_state)]
            last_state = backpointer[i, int(last_state)]
            backtrack_index += 1
    
        q =q[::-1]
        P = omega[-1,q.astype(int)[-1]]

        return [P, q]

    def backward_algorithm(self, V, a, b):
        T = len(V)
        N = a.shape[0]

        beta = np.zeros((T, N))
        beta[-1] = np.ones(N)

        for t in reversed(range(0, T - 1)):
            for i in range(N):
                beta[t, i] =  beta[t+1] @ (a[i, :] * b[:, V[t+1]])

        return beta

    def baum_welch(self, O, A, B):

        T = len(O)
        N = A.shape[0]

        for _ in range(self.n_iter):

            alpha = self.forward_algorithm(O, A, B)
            beta = self.backward_algorithm(O, A, B)

            xi = np.zeros((N, N, T - 1)) # there is no state T+1, therefore T-1

            for t in range(T - 1):

                denominator = alpha[t] @ A * B[:, O[t + 1]] @ beta[t + 1]
                for i in range(N):
                    numerator = alpha[t, i] * A[i] * B[:, O[t + 1]] * beta[t + 1]
                    xi[i, :, t] = numerator / denominator
    
            A = xi.sum(axis=2) / xi.sum(axis=(1,2))
    
            gamma = xi.sum(axis=1)
            gamma = np.hstack((gamma, xi[:, :, -1].sum(axis=0).reshape((-1, 1))))
    
            K = B.shape[1]
            denominator = gamma.sum(axis=1).reshape((-1, 1))
            for l in range(K):
                B[:, l] = gamma[:, O == l].sum(axis=1)
    
            B = np.divide(B, denominator)
 
        return {"A":A, "B":B}

