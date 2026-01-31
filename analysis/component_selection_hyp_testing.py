pq_grid = np.meshgrid(alphas_1, alphas_1)
pq_grid_arr = np.transpose(np.array(pq_grid), (1, 2, 0))
cov = (np.min(pq_grid_arr, axis=2) - pq_grid_arr[:, :, 0] * pq_grid_arr[:, :, 1])*np.sqrt(alphas_1[1] - alphas_1[0])

noise_est_1 = np.zeros((len(alphas_1), len(alphas_1)))
for i, rRTs in enumerate(rRTs_1):
    kde_dens = gaussian_kde(rRTs)(Q_1[i])
    diag_recip_dens = np.diag(1 / kde_dens)
    noise_i = diag_recip_dens @ cov @ diag_recip_dens/len(rRTs)
    noise_est_1 += noise_i

# import pdb; pdb.set_trace()

noise_est_1 /= len(rRTs_1)
noise_eigvals_1, noise_eigvec_1 = np.linalg.eig(noise_est_1)
plt.plot(np.sort(noise_eigvals_1)[::-1])
plt.plot(pca_1.explained_variance_)
plt.show()

noise_est_2 = np.zeros((len(alphas_1), len(alphas_1)))
for i, rRTs in enumerate(rRTs_2):
    kde_dens = gaussian_kde(rRTs)(Q_2[i])
    diag_recip_dens = np.diag(1 / kde_dens)
    noise_i = diag_recip_dens @ cov @ diag_recip_dens/len(rRTs)
    noise_est_2 += noise_i

noise_est_2 /= len(rRTs_2)
noise_eigvals_2, noise_eigvec_2 = np.linalg.eig(noise_est_2)
plt.plot(np.sort(noise_eigvals_2)[::-1])
plt.plot(pca_2.explained_variance_)
plt.show()