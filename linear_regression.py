import numpy as np
from scipy import stats

def lars_backup(X, y):

    m = X.shape[1]
    n = X.shape[0]
    A = set()
    beta = np.zeros((m, ))
    sign = np.zeros((m, ))
    cur_pred = np.zeros((n, ))
    beta_path = np.zeros((m, m))

    residual = y - cur_pred
    c_hat = X.T @ residual
    j = np.argmax(np.abs(c_hat), 0)
    A.add(j)
    sign[j] = 1

    for it in range(m):
        gamma = None

        residual = y - cur_pred

        mse = np.sqrt(np.sum(residual**2))
        pred_from_beta = X @ beta

        c_hat = X.T @ residual
        C_hat = np.abs(c_hat).max()

        X_a = X[:, list(A)]
        X_a *= sign[list(A)]
        
        G_a = X_a.T @ X_a
        G_a_inv = np.linalg.inv(G_a)
        A_a = 1/np.sqrt(np.ones((len(A))).T @ G_a_inv @ np.ones((len(A))))
        w_a = A_a * G_a_inv @ np.ones((len(A)))
        u_a = X_a @ w_a
        a = X.T @ u_a

        if it < m - 1:
            next_j = None
            next_sign = 0
            for j in range(m):
                if j in A:
                    continue
                v0 = (C_hat - c_hat[j]) / (A_a - a[j])
                v1 = (C_hat + c_hat[j]) / (A_a + a[j])
                if v0 > 0 and (gamma is None or v0 < gamma):
                    gamma = v0
                    next_j = j
                    next_sign = 1
                if v1 > 0 and (gamma is None or v1 < gamma):
                    gamma = v1
                    next_j = j
                    next_sign = -1
        else:
            gamma = C_hat / A_a

        sb = u_a * gamma
        sx = np.linalg.lstsq(X_a, sb, rcond=None)
        for i, j in enumerate(A):
            beta[j] += sx[0][i] * sign[j]

        cur_pred = X @ beta
        A.add(next_j)
        sign[next_j] = next_sign

        beta_path[it, :] = beta

        sum_abs_coeff = np.sum(np.abs(beta_path), 1)

    return beta

def lars(X, y):

    m = X.shape[1]
    n = X.shape[0]
    A = set()
    beta = np.zeros((m, ))
    sign = np.zeros((m, ))
    cur_pred = np.zeros((n, ))
    beta_path = np.zeros((m, m))

    residual = y - cur_pred
    c_hat = X.T @ residual
    j = np.argmax(np.abs(c_hat), 0)
    A.add(j)
    sign[j] = 1

    for it in range(m):
        gamma = None

        residual = y - cur_pred

        mse = np.sqrt(np.sum(residual**2))
        pred_from_beta = X @ beta

        c_hat = X.T @ residual
        C_hat = np.abs(c_hat).max()

        X_a = X[:, list(A)]
        X_a *= sign[list(A)]
        
        G_a = X_a.T @ X_a
        G_a_inv = np.linalg.inv(G_a)
        A_a = 1/np.sqrt(np.ones((len(A))).T @ G_a_inv @ np.ones((len(A))))
        w_a = A_a * G_a_inv @ np.ones((len(A)))
        u_a = X_a @ w_a
        a = X.T @ u_a

        if it < m - 1:
            next_j = None
            next_sign = 0
            for j in range(m):
                if j in A:
                    continue
                v0 = (C_hat - c_hat[j]) / (A_a - a[j])
                v1 = (C_hat + c_hat[j]) / (A_a + a[j])
                if v0 > 0 and (gamma is None or v0 < gamma):
                    gamma = v0
                    next_j = j
                    next_sign = 1
                if v1 > 0 and (gamma is None or v1 < gamma):
                    gamma = v1
                    next_j = j
                    next_sign = -1
        else:
            gamma = C_hat / A_a

        # LASSO

        d = np.zeros(len(sign))
        for i,j in zip(A, sign[list(A)] * w_a):
            d[i] = j

        gamma_candidates_tilde = -beta[list(A)]/d[list(A)]
        gamma_tilde = np.where(gamma_candidates_tilde > 0, gamma_candidates_tilde, np.inf).min()
        j_tilde = np.where(gamma_candidates_tilde > 0, gamma_candidates_tilde, np.inf).argmin()

        if gamma_tilde < gamma:
            sb = u_a * gamma_tilde
            A.remove(list(A)[j_tilde])

        else:
            sb = u_a * gamma

        sx = np.linalg.lstsq(X_a, sb, rcond=None)
        for i, j in enumerate(A):
            beta[j] += sx[0][i] * sign[j]

        cur_pred = X @ beta
        A.add(next_j)
        sign[next_j] = next_sign

        beta_path[it, :] = beta

        sum_abs_coeff = np.sum(np.abs(beta_path), 1)

    return beta

def batch_gradient_descent(theta, n_samples, X, y, learning_rate, n_iter, gradient, cost):
  
    curr_iter = 0

    while curr_iter < n_iter:
        curr_cost = cost(X, theta, y)
        theta = theta - learning_rate * gradient(n_samples, X, theta, y)
        curr_iter += 1
        print('iteration: ', curr_iter, 'cost: ', curr_cost)

    return theta

def create_mini_batches(y, batch_size):

    batch_indices = []
    n_batch = y.shape[0] // batch_size
    remainder_batch = y.shape[0] % batch_size
    
    for i in range(n_batch):
        batch_indices.append((i*batch_size, (i+1)*batch_size))
    if remainder_batch > 0:
        batch_indices.append(((i+1)*batch_size, (i+1)*batch_size+remainder_batch))
    
    return batch_indices


def mini_batch_gradient_descent(theta, n_samples, X, y, learning_rate, n_iter, batch_size, gradient, cost):
   
    curr_iter = 0

    while curr_iter < n_iter:
        batch_indices = create_mini_batches(y, batch_size)
        for idx in batch_indices:
            X_mini = X[idx[0]:idx[1], :]
            y_mini = y[idx[0]:idx[1]]
            curr_cost = cost(X, theta, y)
            theta = theta - learning_rate * gradient(n_samples, X_mini, theta, y_mini)
            curr_iter += 1
            print('iteration: ', curr_iter, 'cost: ', curr_cost)

    return theta

def mse(actual, pred, df):
    return np.sum(((actual - pred)**2))/df

def r_squared(actual, pred):
    sse = np.sum((pred - actual)**2)
    sst = np.sum((actual - actual.mean())**2)
    return 1 - sse/sst

def f_test(actual, pred, p, n):
    # p without intercept
    ssm = np.sum((pred - actual.mean())**2)
    sse = np.sum((pred - actual)**2)
    dfe = n - p - 1
    dfm = p
    msm = ssm/dfm
    mse = sse/dfe
    f_stat = msm/mse
    f_stat_prob = stats.f.sf(f_stat, dfm, dfe) # 1 - CDF
    return f_stat, f_stat_prob

def f_test2(rsq, n, p):
    f_stat = (rsq * (n - p - 1))/(p * (1 - rsq))
    f_stat_prob = stats.f.sf(f_stat, p, n - p -1) # 1 - CDF
    return f_stat, f_stat_prob

def standard_error_coef(**kwargs):
    X = kwargs['X']
    mse = kwargs['mse']
    se_coef = np.sqrt(mse * np.linalg.inv(np.dot(X.T, X))).diagonal()

    return se_coef

def t_test(estimates, **kwargs):
    n = kwargs['n']
    p = kwargs['p']
    se_coef = standard_error_coef(**kwargs)
    t_stat = estimates.squeeze() / se_coef
    p_value = 2 * (stats.t.sf(t_stat, n - p - 1))
    t_crit = stats.t.ppf(1.95/2., n - p - 1)
    conf_int = [(est - t_crit * se, est + t_crit * se) for est, se in zip(estimates, se_coef)]
    return [se_coef, t_stat, p_value, conf_int]


class LinearRegression(object):
    def __init__(self, method='least-squares', learning_rate=None, n_iter=None, batch_size=None, scale=True):

        if method in ['bgd', 'mbgd']:
            assert(learning_rate), 'gradient descent needs learning rate'
            assert(n_iter), 'gradient descent needs number of iterations'
        elif method == 'mbgd':
            assert(batch_size), 'mini-batch gradient descent needs batch size'

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_samples = None
        self.n_features = None
        self.X = None
        self.y = None
        self.theta = None
        self.method = method
        self.mse = None
        self.r_squared = None
        self.f_stat = None
        self.f_stat_prob = None
        self.se_coef = None
        self.t_stat = None
        self.t_stat_prob = None
        self.t_stat_conf_int = None
        self.coef_ = None
        self.intercept_ = None
        self.scale = scale


    def fit(self, X, y):
        
        if self.scale:
            self.X = np.hstack((np.ones((X.shape[0], 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        else:
            self.X = X
            self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.theta = np.zeros((self.n_features, 1))


        if self.method == 'bgd':
            gradient = lambda n_samples, X, theta, y: 2/n_samples * X.T.dot(X.dot(theta) - y)
            cost = lambda X, theta, y: ((np.dot(X, theta) - y)**2).mean(axis=None)
            
            self.theta = batch_gradient_descent(self.theta, self.n_samples, self.X, self.y,
                                                self.learning_rate, self.n_iter, gradient, cost)
        elif self.method == 'mbgd':
            gradient = lambda n_samples, X, theta, y: 2/n_samples * X.T.dot(X.dot(theta) - y)
            cost = lambda X, theta, y: ((np.dot(X, theta) - y)**2).mean(axis=None)

            self.theta = mini_batch_gradient_descent(self.theta, self.n_samples, self.X, self.y,
                                                     self.learning_rate, self.n_iter, self.batch_size, gradient, cost)
        elif self.method == 'least-squares':
            A = self.X.T @ self.X
            c = self.X.T @ self.y
            self.theta = np.linalg.solve(A, c)
        elif self.method == 'QR':
            Q, R = np.linalg.qr(self.X)
            c = Q.T @ self.y
            self.theta = np.linalg.solve(R, c)
        elif self.method == 'SVD':
            self.theta = np.linalg.pinv(self.X) @ self.y
        elif self.method == 'lars':
            self.theta = lars(self.X, self.y)
        else:
            print('No suitable method selected')

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def score(self):
        
        pred = (np.dot(self.X, self.theta))
        n = self.n_samples
        p = self.n_features
        estimates = self.theta

        self.mse = mse(self.y, pred, n - p - 1)
        self.r_squared = r_squared(self.y, pred)
        self.f_stat, self.f_stat_prob = f_test(self.y, pred, p, n)
        self.se_coef, self.t_stat, self.t_stat_prob, self.t_stat_conf_int = t_test(
            estimates, X=self.X, mse=self.mse, n=n, p=p)
        
    def predict(self, X):

        if self.scale:
            X = np.hstack((np.ones((X.shape[0], 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        else:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        pred = np.dot(X, self.theta)

        return pred

class RidgeRegression(object):

    def __init__(self, l2_penalty=None, method='least-squares'):
        self.n_samples = None
        self.n_features = None
        self.X = None
        self.y = None
        self.theta = None
        self.method = method
        self.l2_penalty = l2_penalty

    def fit(self, X, y):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y
        if self.method == 'least-squares':
            self.n_features = self.X.shape[1]
            self.n_samples = self.X.shape[0]
            A = np.identity(self.n_features)
            A[0, 0] = 0
            A = self.l2_penalty * A

            self.theta = np.linalg.solve(A + self.X.T @ self.X, self.X.T @ self.y)

class LassoRegression(object):

    def __init__(self, l1_penalty=None, method='bgd', n_iter=None, learning_rate=None):
        self.l1_penalty = l1_penalty
        self.method = method
        self.n_iter = n_iter
        self.X = None
        self.y = None
        self.n_samples = None
        self.n_features = None
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y
        self.n_features = self.X.shape[1]
        self.n_samples = self.X.shape[0]

        if self.method == 'lars':
            pass
        elif self.method == 'cd':
            pass
        elif self.method == 'bgd':
            pass
        else:
            print('No suitable method selected')
        
            