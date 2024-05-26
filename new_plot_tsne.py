import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

X_embedded = np.load("x_embed.npy")
target = np.load("./pretrain_all_targets_0.npy")

eq_map = target == 0
heat = np.sum(eq_map == [True, False, True], axis=1) == 3
burgers = np.sum(eq_map == [False, False, True], axis=1) == 3
kdv = np.sum(eq_map == [True, True, False], axis=1) == 3

colormap = np.zeros(len(target), dtype=int)
colormap[burgers] = 1
colormap[kdv] = 2
colors = np.array(['r', 'b', 'k'])

norm = np.linalg.norm(target, axis=1)
print((target[:,2] > 1.).sum())

region1 = X_embedded[:,0] > -150
not_region1 = np.logical_not(region1)

region2_0 = np.logical_and(X_embedded[:,0] < 1006, X_embedded[:,0] > -400)# and () and ()
region2_1 = np.logical_and(X_embedded[:,1] < 650, X_embedded[:,1] > -980)# and () and ()
region2 = np.logical_and(region2_0, region2_1)
not_region2 = np.logical_not(region2)

#ax.axvline(-110)
#ax.axvline(-157)
#ax.axhline(1)
#ax.axhline(-95)
region3_0 = np.logical_and(X_embedded[:,0] < -110, X_embedded[:,0] > -157)# and () and ()
region3_1 = np.logical_and(X_embedded[:,1] < 1, X_embedded[:,1] > -95)# and () and ()
region3 = np.logical_and(region3_0, region3_1)

only_burgers = np.logical_and(target[:,2] == 0, target[:,0] != 0)
only_heat = np.logical_and(target[:,2] == 0, target[:,0] == 0)

print("Advection region stats:")
#print(target[region1].min(axis=0))
#print(target[region1].max(axis=0))
#print(target[region1].std(axis=0))
#print(target[region1].mean(axis=0))

print("Not Advection region stats:")
print(sum(target[region1][:,2] == 5))
print(sum(target[not_region1][:,2] == 5))
print(sum(target[region1][:,2] == 5) / (sum(target[not_region1][:,2]==5) + sum(target[region1][:,2] == 5)))
#print(target[not_region1].min(axis=0))
#print(target[not_region1].max(axis=0))
#print(target[not_region1].std(axis=0))
#print(target[not_region1].mean(axis=0))
adv_coeff = target[region1][:,2]
#fig, ax = plt.subplots()
xs = [0.5, 1., 2., 5.]
bxs = [0.1, 0.2, 0.5, 1., 2., 5.]
##ax.bar(xs, adv_coeff)
#ax.hist(adv_coeff)
#ax.set_title(r"Advection $\gamma$ Distribution", fontsize=14)
#plt.show()
print()
print("Burgers region stats:")
print(sum(target[region1][:,0] == 5))
print(sum(target[not_region1][:,0] == 5))
print(sum(target[region2][:,0] == 5) / (sum(target[not_region2][:,0]==5) + sum(target[region2][:,0] == 5)))
print(target[np.logical_and(only_burgers, region2)].min(axis=0))
print(target[np.logical_and(only_burgers, region2)].max(axis=0))
print(target[np.logical_and(only_burgers, region2)].std(axis=0))
print(target[np.logical_and(only_burgers, region2)].mean(axis=0))
burgers_coeff = target[np.logical_and(only_burgers, region2)][:,:2]
#fig, ax = plt.subplots(ncols=2)
#ax[0].hist(burgers_coeff[:,0])
#ax[1].hist(burgers_coeff[:,1])
#plt.show()
print()
print("Heat region stats:")
#print(target[np.logical_and(only_heat, region3)].min(axis=0))
#print(target[np.logical_and(only_heat, region3)].max(axis=0))
#print(target[np.logical_and(only_heat, region3)].std(axis=0))
#print(target[np.logical_and(only_heat, region3)].mean(axis=0))
#print(np.unique(target[np.logical_and(only_heat, region3)], return_counts=True))
heat_coeff = target[np.logical_and(only_heat, region3)][:,1]
print(heat_coeff)
#fig, ax = plt.subplots()
#ax.hist(heat_coeff)
#plt.show()
#print()



###fig, ax = plt.subplots(figsize=(30,8), ncols=4)
####### Add adv subaxes
######subaxes_advection = fig.add_axes(mtransforms.Bbox([[0.84, 0.06], [0.97, 0.23]]))
###adv_coeff_count = np.unique(adv_coeff, return_counts=True)[1]
###adv_coeff_count = np.insert(adv_coeff_count, 0, 0)
###adv_coeff_count = np.insert(adv_coeff_count, 0, 0)
###adv_coeff_count = np.insert(adv_coeff_count, 0, 0)
####adv_coeff_count = np.insert(adv_coeff_count, 0, 0)
####adv_coeff_count = np.insert(adv_coeff_count, 0, 0)
###ax[3].bar(xs, adv_coeff_count, width=0.1)
###ax[3].set_xticks(xs)
###ax[3].set_xticklabels([str(x) for x in xs], fontsize=24, rotation=90)
###ax[3].set_title(r"Advection $\gamma$ Distribution", fontsize=40)
######
####### Add heat subaxes
######subaxes_heat = fig.add_axes(mtransforms.Bbox([[0.84, 0.78], [0.97, 0.95]]))
###heat_coeff_count = np.unique(heat_coeff, return_counts=True)[1]
###heat_coeff_count = np.insert(heat_coeff_count, 5, 0)
####print(np.unique(heat_coeff, return_counts=True)[0])
####print(heat_coeff_count)
####print(bxs)
####raise
###ax[0].bar(bxs, heat_coeff_count, width=0.1)
###ax[0].set_xticks(bxs)
###ax[0].set_xticklabels([str(x) for x in bxs], fontsize=24, rotation=90)
###ax[0].set_title(r"Heat $\beta$ Distribution", fontsize=40)
######
####### Add burgers subaxes
######subaxes_burgers = fig.add_axes(mtransforms.Bbox([[0.84, 0.54], [0.97, 0.71]]))
###burgers_coeff_count = np.unique(burgers_coeff[:,0], return_counts=True)[1]
######print(np.unique(burgers_coeff[:,0], return_counts=True))
###ax[1].bar(xs, burgers_coeff_count, width=0.1)
###ax[1].set_xticks(xs)
###ax[1].set_xticklabels([str(x) for x in xs], fontsize=24, rotation=90)
###ax[1].set_title(r"Burgers $\alpha$ Distribution", fontsize=40)
######
######subaxes_burgers = fig.add_axes(mtransforms.Bbox([[0.84, 0.3], [0.97, 0.47]]))
###burgers_coeff_count = np.unique(burgers_coeff[:,1], return_counts=True)[1]
####print(np.unique(burgers_coeff[:,1], return_counts=True))
###ax[2].bar(bxs, burgers_coeff_count, width=0.1)
###ax[2].set_xticks(bxs)
###ax[2].set_xticklabels([str(x) for x in bxs], fontsize=24, rotation=90)
###ax[2].set_title(r"Burgers $\beta$ Distribution", fontsize=40)
###
#### Ticks
###ax[0].set_yticks([0, 35, 70])
###ax[0].set_yticklabels([0, 35, 70], fontsize=24)
###ax[0].set_ylabel("Count", fontsize=32, labelpad=24)
###ax[0].set_xlabel(r"$\beta$ Value", fontsize=32)
###
###ax[1].set_yticks([0, 50, 100])
###ax[1].set_yticklabels([0, 50, 100], fontsize=24)
###ax[2].set_yticks([])
###ax[1].set_ylim(0, 105)
###ax[2].set_ylim(0, 105)
###ax[1].set_xlabel(r"$\alpha$ Value", fontsize=32)
###ax[2].set_xlabel(r"$\beta$ Value", fontsize=32)
###
###ax[3].set_yticks([0, 350, 700])
###ax[3].set_yticklabels([0, 350, 700], fontsize=32)
###ax[3].set_ylim(0, 800)
###ax[3].set_xlabel(r"$\gamma$ Value", fontsize=32)
###
#### Rectangles
###fig.patches.extend([plt.Rectangle((0.02,0.215),0.23,0.775,
###                    fill=False, color='limegreen', zorder=1000, lw=5, linestyle='--',
###                    transform=fig.transFigure, figure=fig)])
###
###fig.patches.extend([plt.Rectangle((0.255,0.215),0.485,0.775,
###                    fill=False, color='goldenrod', zorder=1000, lw=5, linestyle='-.',
###                    transform=fig.transFigure, figure=fig)])
###
###fig.patches.extend([plt.Rectangle((0.745,0.215),0.25,0.775,
###                    fill=False, color='violet', zorder=1000, lw=5, linestyle='dotted',
###                    transform=fig.transFigure, figure=fig)])
###
###legend_elements = [
###        Line2D([0], [0], color='r', lw=0, marker='o', label='Heat', markersize=20),
###        Line2D([0], [0], color='limegreen', lw=6, label='Heat Region', linestyle='--'),
###        Line2D([0], [0], color='b', lw=0, marker='o', label='Burgers', markersize=20),
###        Line2D([0], [0], color='goldenrod', lw=6, label='Burgers Region', linestyle='-.'),
###        Line2D([0], [0], color='k', lw=0, marker='o', label='Advection', markersize=20),
###        Line2D([0], [0], color='violet', lw=6, label='Advection Region', linestyle='dotted'),
###]
###fig.legend(handles=legend_elements, bbox_to_anchor=[0.1, 0.0, 0.71, 0.23], ncol=3, fontsize=35)
###
###plt.tight_layout(rect=[0.,0.2,0.99,1])
####plt.subplots_adjust(wspace=0.2, top=0.9, bottom=0.15, right=0.98)
###plt.subplots_adjust(wspace=0.2, right=0.98)
###plt.savefig("./param_dist.pdf")
###plt.savefig("./param_dist.png")
####plt.show()

fig, ax = plt.subplots(figsize=(12,10))
ax.scatter(X_embedded[:,0], X_embedded[:,1], c=colors[colormap], alpha=norm/max(norm), s=50)
#ax.add_patch(Rectangle((110,-35), 55, 90, linewidth=2, edgecolor='#888888', facecolor='none', linestyle='--'))
#ax.add_patch(Rectangle((150,-50), 70, 96, linewidth=4, edgecolor='violet', facecolor='none', linestyle='dotted'))

#ax.add_patch(Rectangle((40,-15), 66, 80, linewidth=2, edgecolor='#888888', facecolor='none', linestyle='-.'))
#ax.add_patch(Rectangle((53,-98), 47, 33, linewidth=4, edgecolor='goldenrod', facecolor='none', linestyle='-.'))

#ax.add_patch(Rectangle((-157,-95), 47, 96, linewidth=4, edgecolor='limegreen', facecolor='none',
#                       linestyle='--'))

#ax.axvline(-110)
#ax.axvline(-157)
#ax.axhline(1)
#ax.axhline(-95)

#legend_elements = [
#        Line2D([0], [0], color='r', lw=0, marker='s', label='Heat'),
#        Line2D([0], [0], color='mediumseagreen', lw=3, label='Heat Region', linestyle='--'),
#        Line2D([0], [0], color='b', lw=0, marker='s', label='Burgers'),
#        Line2D([0], [0], color='goldenrod', lw=3, label='Burgers Region', linestyle='-.'),
#        Line2D([0], [0], color='k', lw=0, marker='s', label='Advection'),
#        Line2D([0], [0], color='violet', lw=3, label='Advection Region', linestyle='dotted'),
#]
#fig.legend(handles=legend_elements, bbox_to_anchor=[0.1, 0.0, 0.86, 0.12], ncol=3, fontsize=20)
#fig.legend(handles=legend_elements, bbox_to_anchor=[0.1, 0.0, 0.67, 0.12], ncol=3, fontsize=22)

#ax.set_xticks([i for i in np.arange(-150, 151, 150)])
#ax.set_yticks([i for i in np.arange(-150, 151, 150)])

#ax.set_xticklabels([i for i in np.arange(-150, 151, 150)], fontsize=20)
#ax.set_yticklabels([i for i in np.arange(-150, 151, 150)], fontsize=20)

fig.suptitle("1D Fixed-Future t-SNE Embeddings", fontsize=38, y=0.98, x=0.54)

ax.set_xlabel("t-SNE Embedding Dimension 1", fontsize=26)
ax.set_ylabel("t-SNE Embedding Dimension 2", fontsize=26)


#plt.tight_layout(rect=[0,0.10,0.8,0.95])
plt.tight_layout(rect=[0,0.,1,1])


#fig.subplots_adjust(bottom=0.18)
plt.savefig("./new_ff_final_tsne.png")
plt.savefig("./new_ff_final_tsne.pdf")
#plt.show()
