import numpy as np


def ortho_poly_fit(x, degree = 1):
    n = degree + 1
    x = x.flatten()
    if(degree >= len(np.unique(x))):
            stop("'degree' must be less than number of unique points")
    xbar = np.mean(x)
    x = x - xbar
    X = np.fliplr(np.vander(x, n)) #outer(x, seq_len(n) - 1, "^")
    q,r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
    Z = raw / np.sqrt(norm2)
    Z[:,0] = 1
    return Z, norm2, alpha

def ortho_poly_predict(x, alpha, norm2, degree=1):
    x = x.flatten()
    n = degree + 1
    Z = np.empty((len(x), n))
    Z[:,0] = 1
    if degree > 0:
        Z[:, 1] = x - alpha[0]
    if degree > 1:
      for i in np.arange(1,degree):
          Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
    Z /= np.sqrt(norm2)
    Z[:,0] = 1
    return Z

def sin_transform(X, n):
    X = X.flatten()
    features = np.empty((X.shape[0], 2*n+2))
    features[:, 0] = 1
    #features[:, 1] = np.log(X+100)
    features[:,1] = (X - 4000) / 1000.0
    for i in range(0,n):
        features[:, 2*i+2] = np.sin(X * (2*np.pi*(i+1))/15000.0)
        features[:, 2*i+3] = np.cos(X * (2*np.pi*(i+1))/15000.0)
    return features

def build_ortho_poly_featurizer(X, extract_dim, degree):
    Z, norm2, alpha = ortho_poly_fit(X[:,extract_dim], degree = degree)
    return Z, lambda X : ortho_poly_predict(X[:,extract_dim], alpha, norm2, degree=degree), norm2, alpha

def build_sin_featurizer(extract_dim, degree):
    return lambda X : sin_transform(X[:,extract_dim], degree)

def featurizer_from_string(X, desc, extract_dim=0):
    if desc.startswith("poly"):
        degree = int(desc[4:])
        Z, f, norm2, alpha = build_ortho_poly_featurizer(X, extract_dim, degree)
        return Z, f, {'norm2': norm2, 'alpha': alpha}
    elif desc.startswith("sin"):
        degree = int(desc[3:])
        featurizer = build_sin_featurizer(extract_dim, degree)
        Z = featurizer(X)
        return Z, featurizer, {}

def recover_featurizer(desc, extract_dim, recovery_info):
    if desc.startswith("poly"):
        degree = int(desc[4:])
        norm2 = recovery_info['norm2']
        alpha = recovery_info['alpha']
        return lambda X : ortho_poly_predict(X[:,extract_dim], alpha, norm2, degree=degree), {'norm2': norm2, 'alpha': alpha}
    elif desc.startswith("sin"):
        degree = int(desc[3:])
        featurizer = build_sin_featurizer(extract_dim, degree)
        return featurizer, {}
