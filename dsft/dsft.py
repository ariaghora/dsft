import numpy as np

class DSFT:
    """
    The implementation of domain specific feature transfer (DSFT) by Pengfei et al (2017),
    "A General Domain Specific Feature Transfer Framework for Hybrid Domain Adaptation" in
    IEEE Transactions on Knowledge and Data Engineering.
    
    Attributes
    ----------
    alpha : double
        Weight for MMD minimization
    beta : double
        Weight for regularization
    """
    
    def __init__(self, alpha=0.05, beta=0.01):
        self.alpha = alpha
        self.beta  = beta
        self.trained = False
    
    def get_homogeneous_features(self, Xs_c, Xs_d, Xt_c, Xt_d):
        """
        Compute homogeneous representation for both domains
        
        Parameters
        ----------
        Xs_c : numpy ndarray
            Common feature feature matrix for source domain
        Xs_d : numpy ndarray
            Common feature feature matrix for source domain
        Xt_c : numpy ndarray
            Common feature feature matrix for target domain
        Xt_d : numpy ndarray
            Common feature feature matrix for target domain
            
        Returns
        -------
        Xs_h : numpy ndarray
            Computed homogeneous feature matrix for source domain
        Xt_h : numpy ndarray
            Computed homogeneous feature matrix for target domain
        """
        
        n    = Xs_d.shape[0]
        m    = Xt_d.shape[0]
        ns_d = Xs_d.shape[1]
        nt_d = Xt_d.shape[1]
        n_c  = Xs_c.shape[1]

        # MMD matrix construction
        M12 = np.full((n,m),1/(ns_d*nt_d))
        M21 = np.full((m,n),1/(ns_d*nt_d))
        M22 = np.full((m,m),1/(nt_d**2))
        M11 = np.full((n,n),1/(ns_d**2))
        
        alpha = self.alpha
        beta = self.beta

        A = Xs_d.T @ Xs_c - alpha * Xs_d.T @ M12 @ Xt_c
        B = Xs_c.T @ Xs_c + alpha * Xt_c.T @ M22 @ Xt_c + beta * np.eye(n_c)
        Ws = A @ np.linalg.inv(B)
        C = Xt_d.T @ Xt_c - alpha * Xt_d.T @ M21 @ Xs_c
        D = Xt_c.T @ Xt_c + alpha * Xs_c.T @ M11 @ Xs_c + beta * np.eye(n_c)
        Wt = C @ np.linalg.inv(D)
        Xs_a = Xs_c @ Wt.T
        Xt_a = Xt_c @ Ws.T

        Xs_h = np.hstack([Xs_c, Xs_d, Xs_a])
        Xt_h = np.hstack([Xt_c, Xt_a, Xt_d])
        
        self.trained = True
        self.Ws = Ws
        self.Wt = Wt
        return Xs_h, Xt_h
    
    def make_homogeneous_feature(self, X_c, X_d, is_target):
        if not is_target:
            W = self.Wt
            X_a = X_c @ W.T
            X_h = np.hstack([X_c, X_d, X_a])
        else:
            W = self.Ws
            X_a = X_c @ W.T
            X_h = np.hstack([X_c, X_a, X_d])
        
        return X_h