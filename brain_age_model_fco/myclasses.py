import numpy as np
from scipy.stats import linregress, multivariate_normal
from scipy.special import softmax
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KernelDensity


class BayesianRidge2(BayesianRidge, TransformerMixin):
    def transform(self, X):
        return super().predict(X)


class NBRegressor(BaseEstimator, TransformerMixin):
    """
    p(y|X) prop p(X|y)p(y)
    where p(X|y) = N(k*y+b, Sigma)
    E[y|X] = sum y*p(y|X)
    """
    def __init__(self, y_kde_bandwidth):
        self.y_kde_bandwidth = y_kde_bandwidth
        
    def fit(self, X, y):
        self.coef_ = np.zeros(X.shape[1])
        self.intercepts_ = np.zeros(X.shape[1])
        X2 = np.array(X)
        for i in range(X.shape[1]):
            self.coef_[i], self.intercepts_[i], _, _, _ = linregress(y, X[:,i])
            X2[:,i] = X[:,i] - y*self.coef_[i] - self.intercepts_[i]
        self.cov = np.cov(X2.T)
        self.y_kde = KernelDensity(bandwidth=self.y_kde_bandwidth).fit(y.reshape(-1,1))
        return self
        
    def predict(self, X):
        ys = np.arange(1,100+1,1)
        log_ys = self.y_kde.score_samples(ys.reshape(-1,1))
        log_p_x_y = []
        for i, y in enumerate(ys):
            #print(y)
            log_p_x_y.append( multivariate_normal(self.coef_*y+self.intercepts_, self.cov, allow_singular=True).logpdf(X) )
        p_y_x = softmax(np.array(log_p_x_y).T + log_ys, axis=1)
        yp = np.sum(p_y_x*ys, axis=1)
        return yp
        
    def transform(self, X):
        return self.predict(X)


class BiasCorrection(BaseEstimator, RegressorMixin):
    def __init__(self, method):
        self.method = method
        
    def fit(self, y, yp, sample_weight=None):
        if self.method=='lr':
            model = LinearRegression().fit(y.reshape(-1,1), yp-y, sample_weight=sample_weight)
            self.correction_slope = model.coef_[0]
            self.correction_intercept = model.intercept_
        else:
            raise NotImplementedError(self.method)
        return self
        
    def predict(self, yp, y=None):
        if self.method=='lr':
            yp2 = yp - (self.correction_slope*y + self.correction_intercept)

        # softplus
        yp2 = np.log1p(np.exp(-np.abs(yp2))) + np.maximum(yp2,0)
        return yp2
        

