import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

data = np.load("./pretrain_all_preds_0.npy")
target = np.load("./pretrain_all_targets_0.npy")
data = data.reshape((data.shape[0], -1))


#fig, ax = plt.subplots()
#ax.plot(data[2])
#plt.show()

#print(data.shape)
#print(target[:5])
eq_map = target == 0
heat = np.sum(eq_map == [True, False, True], axis=1) == 3
burgers = np.sum(eq_map == [False, False, True], axis=1) == 3
kdv = np.sum(eq_map == [False, True, False], axis=1) == 3
#print(heat[:5])
#print(heat)

colormap = np.zeros(len(target), dtype=int)
colormap[burgers] = 1
colormap[kdv] = 2

colors = np.array(['r', 'b', 'k'])

ITER = 1000000
X_embedded = TSNE(n_components=2, learning_rate='auto', verbose=2, n_iter=ITER, min_grad_norm=1e-9,
                  init='random', perplexity=10).fit_transform(data)

norm = np.linalg.norm(target, axis=1)

np.save("./x_embed.npy", X_embedded)

fig, ax = plt.subplots()
ax.scatter(X_embedded[:,0], X_embedded[:,1], c=colors[colormap], alpha=norm/max(norm))
#ax.set_xticks([])
#ax.set_yticks([])

legend_elements = [
        Line2D([0], [0], color='r', lw=0, marker='s', label='Heat'),
        Line2D([0], [0], color='b', lw=0, marker='s', label='Burgers'),
        Line2D([0], [0], color='k', lw=0, marker='s', label='KdV'),
]
fig.legend(handles=legend_elements, bbox_to_anchor=[0.1, 0.0, 0.73, 0.13], ncol=3, fontsize=14)
#fig.suptitle("PITT FNO Rollout Pretraining Performance Comparison", fontsize=40)
fig.suptitle("Constant Forcing Term Physics Informed WNTXent", fontsize=18)
plt.tight_layout(rect=[0,0.08,1,1])
plt.savefig("./tsne.png")


plt.show()
