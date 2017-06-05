import numpy as np

import matplotlib.pylab as plt
plt.style.use('ggplot')
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

from sdtw.barycenter import sdtw_barycenter
from sdtw.dataset import load_ucr

X_tr = load_ucr("Gun_Point")[0]
X1, X2 = X_tr[7], X_tr[37]

init_25 = 0.25 * X1 + 0.75 * X2
init_50 = 0.50 * X1 + 0.50 * X2
init_75 = 0.75 * X1 + 0.25 * X2

bary_25 = sdtw_barycenter([X1, X2], init_25, gamma=1, max_iter=100,
                          weights=[0.25, 0.75])
bary_50 = sdtw_barycenter([X1, X2], init_50, gamma=1, max_iter=100,
                          weights=[0.50, 0.50])
bary_75 = sdtw_barycenter([X1, X2], init_75, gamma=1, max_iter=100,
                          weights=[0.75, 0.25])

colors = [
    (0, 51./255, 204./255),
    (102./255, 153./255, 255./255),
    (255./255, 102./255, 255./255),
    (255./255, 0, 102./255),
    (1.0, 51./255, 0),
]


fig = plt.figure(figsize=(10,4))

ax = fig.add_subplot(121)

ax.plot(X1.ravel(), c=colors[0], lw=3)
ax.plot(bary_75, c=colors[1], lw=3, alpha=0.75)
ax.plot(bary_50, c=colors[2], lw=3, alpha=0.75)
ax.plot(bary_25, c=colors[3], lw=3, alpha=0.75)
ax.plot(X2.ravel(), c=colors[4], lw=3)
ax.set_title("Soft-DTW geometry")

ax = fig.add_subplot(122)

ax.plot(X1.ravel(), c=colors[0], lw=3)
ax.plot(init_75, c=colors[1], lw=3, alpha=0.75)
ax.plot(init_50, c=colors[2], lw=3, alpha=0.75)
ax.plot(init_25, c=colors[3], lw=3, alpha=0.75)
ax.plot(X2.ravel(), c=colors[4], lw=3)
ax.set_title("Euclidean geometry")

plt.show()
