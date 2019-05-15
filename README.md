# Domain Specific Feature Transfer (DSFT)

The implementation of domain specific feature transfer (DSFT) by Pengfei et al (2017), "A General Domain Specific Feature Transfer Framework for Hybrid Domain Adaptation" in IEEE Transactions on Knowledge and Data Engineering.

## Usage

```python
import numpy as np
from dsft import DSFT

# create a dummy dataset for hybrid domain adaptation problem
Xs_c = np.random.randn(100, 10) # 100 x 10 feature matrix (source common features)
Xs_d = np.random.randn(100, 5)  # 100 x 5 feature matrix (source specific features)
Xt_c = np.random.randn(150, 10) # 100 x 10 feature matrix (target common features)
Xt_d = np.random.randn(150, 7)  # 100 x 7 feature matrix (target specific features)

# initialize domain adaptor
dsft       = DSFT(alpha=0.05, beta=0.01)
Xs_h, Xt_h = dsft.get_homogeneous_features(Xs_c, Xs_d, Xt_c, Xt_d)

print('Now dimensions for source and target data, respectively:', Xs_h.shape, Xt_h.shape)

# Now we can train a classifier/regressor on Xs_h with source domain label/dependent 
# variable and test the classifier on Xt_h
```

## Installation

You may use pip to install DSFT as follows:

```pip install git+https://github.com/ariaghora/dsft```