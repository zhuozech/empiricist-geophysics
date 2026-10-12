# %% [markdown]
# # 2.06 · Matrix and notation primer: the linear algebra an inversion actually uses
#
# **Before this:** 2.03 (variance, covariance, the covariance matrix and the Gaussian ellipse).
# **You will be able to:** read the shape and units of every symbol in $\mathbf d = \mathbf G\mathbf m + \mathbf e$;
# say what a transpose, an inverse and a determinant do; read a covariance matrix through its eigenvalues and tell a
# valid one from a singular or impossible one; use the Cholesky factor to draw correlated noise, whiten residuals,
# solve systems and compute log-determinants, and recognise when it fails; solve a linear system without forming an
# inverse; read rank and null space off a singular value decomposition.
# **Optional sections:** 5e (how the Cholesky factorisation fails) and 8 (log-determinant overflow) on a first reading.
#
# Every later notebook writes the inverse problem with vectors and matrices. This one collects the few operations
# that keep coming back, each on a tiny fault with a handful of slip patches and GPS stations, so that every number can
# be checked by eye. It is a reference: read it once now, and come back when a later notebook uses a word from the list
# below.
#
# **Words this notebook defines:** scalar, vector, matrix, Green's function, matrix-vector product, transpose, dot
# product, gradient, inverse, determinant, singular matrix, eigenvalue and eigenvector, positive definite and
# semi-definite, Cholesky factor, pivot, triangular solve, whitening, jitter, round-off, condition number, least squares,
# normal equations, QR factorisation, singular value decomposition (SVD), rank, null space, log-determinant.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | two-patch fault (sections 1-4) | 2 slip values (m): 0-10 km and 10-20 km depth | 3 surface displacements (mm) at 3, 10, 30 km from the fault | antiplane screw-dislocation Green's functions, $\mathbf d = \mathbf G\mathbf m$ | linear |
# | correlated GPS noise (section 5) | none: noise models only | 2 to 4 stations; 30 points of a smooth field | exponential and squared-exponential covariances | n/a |
# | ill-conditioned test matrices (section 6) | 6 | 40 | $\mathbf d = \mathbf G\mathbf m$ with chosen singular values | linear |
# | five-patch fault (section 7) | 5 slip values, 0-25 km | the same 3 stations | screw dislocation | linear |
# | many stations (section 8) | none: a likelihood evaluation | 10 to 3,000 | independent noise, σ = 2 mm | n/a |

# %%
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import bayeskit as bk
from bayeskit import plotting as BP, style as S

T0 = time.time()
np.set_printoptions(precision=4, suppress=True)


def patch_disp(x, z1, z2):
    """Surface displacement per unit slip of an antiplane patch between depths z1 and z2 (screw dislocation)."""
    return (np.arctan(z2 / x) - np.arctan(z1 / x)) / np.pi


# %% [markdown]
# ## 1. Notation, shapes and units
#
# **In plain words.** A **scalar** is one number, a **vector** is a list of numbers, a **matrix** is a table. The
# forward model of a linear inverse problem says: the data are a weighted sum of the unknowns. Writing it with a
# matrix is just bookkeeping for "station $i$ sees $G_{i1}$ times the slip of patch 1 plus $G_{i2}$ times the slip of
# patch 2". Each **column** of $\mathbf G$ is the fingerprint one patch leaves on all stations; each **row** is what
# one station sees of all patches. The entries are **Green's functions**: the displacement at one station caused by
# unit slip on one patch, computed once from a physical model of the Earth.
#
# The example fault here is the simplest one used in this course: a long vertical strike-slip fault in a uniform
# elastic half-space, with slip only along strike, so that the ground moves only parallel to the fault ("antiplane"
# motion). Its Green's functions have a closed form, the screw dislocation (Segall 2010): a patch between depths
# $z_1$ and $z_2$ moves a surface point at distance $x$ by $[\arctan(z_2/x) - \arctan(z_1/x)]/\pi$ per unit slip.
#
# **Precisely.** The notation used in every notebook:
#
# | symbol | what it is | shape | units in this notebook |
# |---|---|---|---|
# | $a$, $\sigma$, $\lambda$ (italic) | scalars | 1 | various |
# | $\mathbf d$ (bold lower case) | data vector | $N_d$ | mm |
# | $\mathbf m$ | model (unknowns) vector | $N_m$ | m of slip |
# | $\mathbf G$ (bold upper case) | forward operator: Green's functions | $N_d\times N_m$ | mm per m of slip |
# | $\mathbf e$ | noise vector | $N_d$ | mm |
# | $\mathbf C_d$ | data (noise) covariance | $N_d\times N_d$ | mm² |
# | $\mathbf C_m$ | prior covariance of the model | $N_m\times N_m$ | m² |
# | $\mathbf C_p$ | prediction (model-error) covariance (2.13) | $N_d\times N_d$ | mm² |
# | $\mathbf C_\chi = \mathbf C_d + \mathbf C_p$ | total misfit covariance (2.13) | $N_d\times N_d$ | mm² |
# | $\mathbf r = \mathbf d - \mathbf G\mathbf m$ | residual: what the model fails to explain | $N_d$ | mm |
# | $\mathbf I$ | identity matrix: ones on the diagonal, zeros elsewhere | | |
#
# The **matrix-vector product** $\mathbf G\mathbf m$ has entries $(\mathbf G\mathbf m)_i = \sum_j G_{ij}m_j$. It is
# defined only when the number of columns of $\mathbf G$ equals the length of $\mathbf m$, and the result has the length
# of the data. Checking shapes and units is the cheapest bug test there is: $(N_d\times N_m)(N_m)\to(N_d)$, and
# (mm per m)(m) = mm.
#
# **Worked example.** Three stations at 3, 10 and 30 km from the fault; two patches, 0-10 km and 10-20 km deep; slip
# 2 m on the shallow patch and 1 m on the deep one.

# %%
x_st = np.array([3.0, 10.0, 30.0])                    # km from the fault
z2 = np.array([0.0, 10.0, 20.0])                      # patch edges (km)
G = 1000.0 * np.array([[patch_disp(x, z2[j], z2[j + 1]) for j in range(2)] for x in x_st])   # mm per m
m_ex = np.array([2.0, 1.0])                           # m
d_ex = G @ m_ex
print("G (rows: stations at 3, 10, 30 km; columns: shallow, deep patch), mm per m of slip:\n", G)
print(f"shapes: G {G.shape}, m {m_ex.shape}, d = G m {d_ex.shape}")
print("d = G m (mm):", d_ex)
print("the same, column by column: 2 x column 1 + 1 x column 2 =", 2 * G[:, 0] + 1 * G[:, 1])
print("the same, row by row (station 1): G[0,0]*2 + G[0,1]*1 =", G[0, 0] * 2 + G[0, 1] * 1)

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[1.15, 0.9, 1.1]))
ax = axs[0]
for j, c in enumerate([S.BLUE, S.ORANGE]):
    ax.add_patch(plt.Rectangle((-0.6, z2[j]), 1.2, z2[j + 1] - z2[j], color=c, alpha=0.75))
    ax.text(1.2, 0.5 * (z2[j] + z2[j + 1]), f"patch {j + 1}: {m_ex[j]:.0f} m", va="center", fontsize=8.5, color=S.INK)
ax.plot(x_st, np.zeros(3), "v", color=S.INK, ms=9, clip_on=False)
for i, x in enumerate(x_st):
    ax.text(x, -1.5, f"station {i + 1}", ha="center", fontsize=8)
ax.set_xlim(-2, 33); ax.set_ylim(22, -3)
ax.set_xlabel("distance from the fault (km)"); ax.set_ylabel("depth (km)")
ax.set_title("(a) Two patches, three stations")
ax = axs[1]
ax.imshow(G, cmap="Greys", vmin=0, vmax=G.max() * 1.25)
for i in range(3):
    for j in range(2):
        ax.text(j, i, f"{G[i, j]:.0f}", ha="center", va="center", fontsize=10, color=S.SURFACE if G[i, j] > 0.6 * G.max() else S.INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["patch 1\n(column 1)", "patch 2\n(column 2)"])
ax.set_yticks([0, 1, 2]); ax.set_yticklabels([f"station {i + 1} (row {i + 1})" for i in range(3)])
ax.set_title("(b) G: mm of displacement per m of slip")
ax = axs[2]
w = 0.35
ax.bar(np.arange(3), 2 * G[:, 0], w * 2, color=S.BLUE, alpha=0.8, label="2 m × column 1 (shallow)")
ax.bar(np.arange(3), 1 * G[:, 1], w * 2, bottom=2 * G[:, 0], color=S.ORANGE, alpha=0.8, label="1 m × column 2 (deep)")
ax.hlines(d_ex, np.arange(3) - 0.38, np.arange(3) + 0.38, color=S.INK, lw=2, label="d = G m")
ax.set_xticks(np.arange(3)); ax.set_xticklabels([f"station {i + 1}\n{x:.0f} km" for i, x in enumerate(x_st)])
ax.set_ylabel("surface displacement (mm)")
ax.set_title("(c) The product: columns weighted by slip, added")
ax.legend(fontsize=7.8, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-06_forward_product")
plt.show()

# %% [markdown]
# **How to read it.** (a) The geometry. (b) The matrix: each entry is the displacement at one station per metre of
# slip on one patch. The shallow patch (column 1) is seen strongly by the near station and weakly far away; the deep
# patch (column 2) is seen more evenly. (c) The prediction is the sum of the two columns, each scaled by its slip.
#
# **Common misreadings.**
# * *"Matrix multiplication is something abstract."* Here it is a weighted sum of fingerprints; that is all
#   $\mathbf G\mathbf m$ ever is, for 2 patches or for 2,000.
# * *"Units do not matter as long as the code runs."* Mixing m and mm in $\mathbf G$ changes the answer by a factor of
#   1000 without any error message; the unit check above catches it.

# %% [markdown]
# ## 2. The transpose: sending data back to the parameters
#
# **In plain words.** $\mathbf G$ takes slip to data. Its **transpose** $\mathbf G^\top$ (the same table with rows and
# columns swapped, $(\mathbf G^\top)_{ij} = G_{ji}$) goes the other way: it takes something defined at the stations,
# such as a residual, and tells each patch how much of it that patch could explain. That is why $\mathbf G^\top$
# appears in every gradient and every normal equation.
#
# **Precisely.** The **dot product** of two vectors of the same length is $\langle\mathbf a,\mathbf b\rangle =
# \mathbf a^\top\mathbf b = \sum_i a_ib_i$: large when they point the same way, zero when they are perpendicular. For any
# slip vector $\mathbf m$ and any data-space vector $\mathbf y$,
# $\langle\mathbf G\mathbf m,\mathbf y\rangle = \langle\mathbf m,\mathbf G^\top\mathbf y\rangle$ (the "dot-product
# test", used to check codes that implement $\mathbf G^\top$ separately; 2.22). The **gradient** of a function of
# several unknowns is the vector of its partial derivatives: it points in the direction in which the function rises
# fastest. The misfit $\chi^2(\mathbf m) = \mathbf r^\top\mathbf C_d^{-1}\mathbf r$ with $\mathbf r = \mathbf d -
# \mathbf G\mathbf m$ ($\mathbf C_d^{-1}$ is the inverse of the noise covariance, section 3) has gradient
# $-2\,\mathbf G^\top\mathbf C_d^{-1}\mathbf r$: the noise-weighted residual sent back through $\mathbf G^\top$.
#
# **Worked example.** Data made from the slip of section 1 plus noise (σ = 2 mm), compared with a guess of 1 m on each
# patch; the gradient is checked against finite differences (the change in χ² for a tiny change of one slip value).

# %%
sig = 2.0                                                             # mm
d_obs = d_ex + sig * np.random.default_rng(3).standard_normal(3)
m_guess = np.array([1.0, 1.0])
r = d_obs - G @ m_guess
Cd = sig ** 2 * np.eye(3)
grad = -2 * G.T @ np.linalg.solve(Cd, r)
chi2 = lambda m: (d_obs - G @ m) @ np.linalg.solve(Cd, d_obs - G @ m)
h = 1e-6
grad_fd = np.array([(chi2(m_guess + h * e) - chi2(m_guess - h * e)) / (2 * h) for e in np.eye(2)])
print("observed d (mm):", d_obs, "  residual r = d - G m_guess (mm):", r)
print("gradient of chi^2 from G^T:", grad, "  from finite differences:", grad_fd)
y_rand = np.random.default_rng(4).standard_normal(3)
m_rand = np.random.default_rng(5).standard_normal(2)
print(f"dot-product test: <G m, y> = {(G @ m_rand) @ y_rand:.12f},  <m, G^T y> = {m_rand @ (G.T @ y_rand):.12f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax = axs[0]
ax.bar(np.arange(3), r / sig, color=S.MUTED, width=0.55)
ax.axhline(0, color=S.AXIS, lw=0.8)
ax.set_xticks(np.arange(3)); ax.set_xticklabels([f"station {i + 1}" for i in range(3)])
ax.set_ylabel("residual / σ")
ax.set_title("(a) Data space: what the guess misses")
ax = axs[1]
ax.bar([0, 1], -grad, color=[S.BLUE, S.ORANGE], width=0.55)
ax.axhline(0, color=S.AXIS, lw=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["patch 1 (shallow)", "patch 2 (deep)"])
ax.set_ylabel("−∂χ²/∂m  (per m of slip)")
ax.set_title("(b) Parameter space: 2 Gᵀ C_d⁻¹ r, the downhill direction")
fig.tight_layout()
bk.savefig(fig, "2-06_transpose")
plt.show()

# %% [markdown]
# **How to read it.** (a) The guess under-predicts every station (positive residuals). (b) The same residuals sent
# back through $\mathbf G^\top$: both patches should slip more, the shallow one more strongly, because the near
# station, which sees mainly the shallow patch, has the largest residual. The printed gradient equals the
# finite-difference one.
#
# **Common misreadings.**
# * *"$\mathbf G^\top$ is the inverse of $\mathbf G$."* It is not; it does not undo $\mathbf G$ (it does not even have
#   the right units). It only says which direction in parameter space reduces the misfit.

# %% [markdown]
# ## 3. The inverse and the determinant
#
# **In plain words.** A square matrix $\mathbf A$ moves points around: it stretches, squeezes, tilts and possibly
# flips space. Its **inverse** $\mathbf A^{-1}$ moves them back, so that $\mathbf A^{-1}\mathbf A = \mathbf I$. The
# **determinant** $\det\mathbf A$ (also written $\lvert\mathbf A\rvert$) is the factor by which $\mathbf A$ changes
# areas (in 2-D) or volumes (in more dimensions), with a minus sign if it flips orientation. If some direction is
# squashed flat, the area becomes zero, $\det\mathbf A = 0$, and nothing can undo the squashing: the matrix is
# **singular** and has no inverse.
#
# **Precisely.** For $\mathbf A = \begin{pmatrix}a & b\\ c & d\end{pmatrix}$: $\det\mathbf A = ad - bc$ and
# $\mathbf A^{-1} = \frac{1}{ad-bc}\begin{pmatrix}d & -b\\ -c & a\end{pmatrix}$. In general
# $\det(\mathbf A\mathbf B) = \det\mathbf A\,\det\mathbf B$, and the determinant is the product of the eigenvalues
# (section 4). Why it matters here: the inverse of the noise covariance weights the misfit (section 2), and the
# determinant of a covariance sets the normalising constant of the Gaussian density (2.03), which matters whenever a
# noise level is estimated (2.30). In computer code, inverses are almost never formed explicitly (section 6).
#
# **Worked example.** The unit square mapped by an invertible matrix and by a singular one.

# %%
A_ok = np.array([[1.5, 0.6], [0.3, 1.0]])
A_sing = np.array([[1.5, 0.75], [0.6, 0.3]])                         # second column = half the first
for nm, A_ in [("invertible", A_ok), ("singular", A_sing)]:
    print(f"{nm:10s} A = {A_.tolist()}: det = {np.linalg.det(A_):.4f}", end="")
    if abs(np.linalg.det(A_)) > 1e-12:
        print(f";  A^-1 = {np.round(np.linalg.inv(A_), 4).tolist()};  A^-1 A = {np.round(np.linalg.inv(A_) @ A_, 12).tolist()}")
    else:
        print("  -> no inverse: the whole plane is squashed onto one line")

# %%
sq = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0.0]])
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, A_, ttl in [(axs[0], A_ok, "(a) Invertible: area scaled by det A"), (axs[1], A_sing, "(b) Singular: det A = 0")]:
    ax.fill(sq[0], sq[1], color=S.MUTED, alpha=0.25, label="unit square (area 1)")
    im_ = A_ @ sq
    ax.fill(im_[0], im_[1], color=S.BLUE, alpha=0.35, label=f"A × square (area {abs(np.linalg.det(A_)):.2f})")
    ax.plot(im_[0], im_[1], color=S.BLUE, lw=1.4)
    for k, c in enumerate([S.ORANGE, S.AQUA]):
        ax.annotate("", xy=A_[:, k], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
    ax.plot([], [], color=S.ORANGE, lw=1.6, label="A × (1, 0) = column 1")
    ax.plot([], [], color=S.AQUA, lw=1.6, label="A × (0, 1) = column 2")
    ax.set_xlim(-0.3, 2.4); ax.set_ylim(-0.3, 1.8); ax.set_aspect("equal")
    ax.set_title(ttl); ax.legend(fontsize=7.6, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-06_inverse_determinant")
plt.show()

# %% [markdown]
# **How to read it.** (a) The square becomes a parallelogram spanned by the two columns of $\mathbf A$; its area is
# $\lvert\det\mathbf A\rvert$ (printed), and $\mathbf A^{-1}$ maps it back. (b) The second column is a multiple of the
# first, so the parallelogram collapses onto a line: zero area, and many points land on the same image point, so no
# inverse can tell them apart. That is exactly the situation of slip patterns the stations cannot see (section 7).
#
# **Common misreadings.**
# * *"A tiny determinant means the matrix is nearly singular."* Not by itself: scaling all entries by 0.001 makes the
#   determinant tiny without making the matrix any harder to invert. Nearness to singularity is measured by the
#   condition number (section 6).

# %% [markdown]
# ## 4. Covariance matrices: eigenvalues, positive-definiteness, and when it fails
#
# **Reminder (2.03).** The *variance* of a quantity is the average squared distance from its mean; the *covariance* of
# two quantities ("co-": varying together) is positive when they rise together and negative when one rises as the
# other falls. The **covariance matrix** $\mathbf C$ holds the variances on its diagonal and the covariances off it; it
# is symmetric, $\mathbf C = \mathbf C^\top$.
#
# **In plain words.** Every covariance matrix has special directions, its **eigenvectors**, along which the
# uncertainty is simply "more" or "less", without any tilt; the amounts are the **eigenvalues**, the variances along
# those directions. In two dimensions they are the axes of the uncertainty ellipse. A valid covariance must give every
# possible combination of the quantities a variance of zero or more, which means every eigenvalue is zero or more.
#
# **Precisely.** A vector $\mathbf q\ne\mathbf 0$ with $\mathbf C\mathbf q = \lambda\mathbf q$ is an eigenvector and
# $\lambda$ its eigenvalue. A symmetric $\mathbf C$ has $n$ perpendicular unit eigenvectors, so
# $\mathbf C = \mathbf Q\boldsymbol\Lambda\mathbf Q^\top$ (**eigen-decomposition**) with the eigenvectors as columns of
# $\mathbf Q$ and the eigenvalues on the diagonal of $\boldsymbol\Lambda$. The variance of any weighted combination
# $y = \mathbf a^\top\mathbf x$ is $\mathrm{Var}(y) = \mathbf a^\top\mathbf C\mathbf a$ (expand the square of
# $\sum_i a_i(x_i - \mu_i)$ and average). Since a variance cannot be negative, $\mathbf a^\top\mathbf C\mathbf a\ge 0$
# for every $\mathbf a$: $\mathbf C$ is **positive semi-definite**; all $\lambda\ge 0$. It is **positive definite**
# (all $\lambda > 0$; then $\det\mathbf C > 0$, $\mathbf C^{-1}$ exists and so does the Cholesky factor of section 5)
# when no combination is exactly fixed. Why it matters: the posterior covariance, the noise covariance and the prior
# covariance all have to be valid in this sense, and inverting, factorising or sampling them fails when they are not.
#
# **Worked example (a).** The posterior covariance of the two-patch problem with σ = 2 mm and a prior standard
# deviation of 3 m on each patch, $\mathbf C_{\rm post} = (\mathbf G^\top\mathbf C_d^{-1}\mathbf G + \mathbf C_m^{-1})^{-1}$
# (derived in 2.08).

# %%
Cm = 9.0 * np.eye(2)                                                     # (3 m)^2
P_post = G.T @ np.linalg.solve(Cd, G) + np.linalg.inv(Cm)                # posterior precision (1/m^2)
C_post = np.linalg.inv(P_post)                                           # 2 x 2 only: fine to invert (section 6)
m_post = C_post @ (G.T @ np.linalg.solve(Cd, d_obs))
lam, Q = np.linalg.eigh(C_post)
fmt = dict(formatter={"float_kind": lambda v: f"{v: .3e}"})
print("posterior mean (m):", m_post, "  true:", m_ex)
print("posterior covariance (m^2):\n", np.array2string(C_post, **fmt))
print("symmetric:", np.allclose(C_post, C_post.T), "  eigenvalues (m^2):", np.array2string(lam, **fmt), "  all > 0:", bool(np.all(lam > 0)))
print("sd of each patch (m):", np.sqrt(np.diag(C_post)),
      "  correlation:", C_post[0, 1] / np.sqrt(C_post[0, 0] * C_post[1, 1]))
print("ellipse half-axes, 1 sd (m):", np.sqrt(lam), " along eigenvectors (columns):\n", Q)
print("check C q = lambda q for the first eigenvector:", np.allclose(C_post @ Q[:, 0], lam[0] * Q[:, 0]))
print(f"check det C = product of eigenvalues: {np.linalg.det(C_post):.4e} vs {np.prod(lam):.4e}")
bad = np.array([[1.0, 1.2], [1.2, 1.0]])
print("a 'covariance' with correlation 1.2:", bad.tolist(), " eigenvalues", np.linalg.eigvalsh(bad), "-> not a covariance")

# %%
rng_e = np.random.default_rng(7)
draws = m_post + rng_e.standard_normal((400, 2)) @ np.linalg.cholesky(C_post).T
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axs[0]
ax.scatter(draws[:, 0], draws[:, 1], s=4, color=S.BLUE, alpha=0.35, label="400 draws from the posterior")
for nsig in [1, 2]:
    BP.ellipse(ax, m_post, C_post, nsig=nsig, color=S.INK, lw=1.2, ls="-" if nsig == 1 else "--",
               label=f"{nsig}-sd ellipse")
for k in range(2):
    v = Q[:, k] * np.sqrt(lam[k])
    ax.annotate("", xy=m_post + v, xytext=m_post, arrowprops=dict(arrowstyle="-|>", color=S.ORANGE, lw=1.6))
ax.plot([], [], color=S.ORANGE, lw=1.6, label="eigenvectors × √eigenvalue")
ax.plot(*m_ex, "*", color=S.INK, ms=12, mec=S.SURFACE, label="true slip")
ax.set_xlabel("slip on patch 1, shallow (m)"); ax.set_ylabel("slip on patch 2, deep (m)")
ax.set_title("(a) Posterior covariance as an ellipse")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_aspect("equal", adjustable="datalim")
ax = axs[1]
t = np.linspace(0, 2 * np.pi, 200)
u = np.stack([np.cos(t), np.sin(t)])
q_good = np.einsum("in,ij,jn->n", u, C_post / np.sqrt(np.outer(np.diag(C_post), np.diag(C_post))), u)
q_bad = np.einsum("in,ij,jn->n", u, bad, u)
ax.plot(np.degrees(t), q_good, color=S.BLUE, lw=1.8, label="correlation matrix of (a)")
ax.plot(np.degrees(t), q_bad, color=S.BAD, lw=1.8, ls="--", label="'correlation' 1.2: not a covariance")
ax.axhline(0, color=S.AXIS, lw=0.8)
ax.set_xlabel("direction of the combination (degrees)"); ax.set_ylabel("variance of that combination, uᵀCu")
ax.set_title("(b) Positive-definite: every direction has variance > 0")
ax.set_ylim(top=1.35 * max(q_good.max(), q_bad.max()))               # headroom so the legend clears the curves
ax.legend(fontsize=7.8, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-06_covariance_ellipse")
plt.show()

# %% [markdown]
# **How to read it.** (a) The draws fill a tilted ellipse: the two slips trade off (negative correlation, printed),
# because more shallow slip and less deep slip predict similar data. The orange arrows are the eigenvectors scaled
# by the square roots of the eigenvalues: the long axis is the poorly constrained combination, the short axis the
# well constrained one. (b) For every unit direction $\mathbf u$, the variance of the combination $\mathbf u^\top\mathbf m$
# is $\mathbf u^\top\mathbf C\mathbf u$. A valid covariance gives a positive value in every direction; the matrix with
# "correlation" 1.2 goes negative in some directions, so it describes no possible random vector.
#
# **Worked example (b): when a covariance is only semi-definite, or not one at all.** Each case below is computed:
# 1. *Perfectly correlated quantities*: $x_2 = 2x_1$ exactly. The combination $2x_1 - x_2$ never varies.
# 2. *A constraint*: three proportions that must add up to one (Dirichlet shares, 2.11). Their sum never varies.
# 3. *Too few samples*: a covariance estimated from 5 draws (ensemble members, 2.24) of 10 quantities. Five draws span
#    at most four directions around their mean; the others get zero estimated variance.
# 4. *A smoothing operator with a null space*: the second-difference roughness penalty used in 2.09 does not penalise a
#    constant or a straight-line slip profile at all, so its precision matrix (inverse covariance, 2.08) has two zero
#    eigenvalues: those patterns are completely free.
# 5. *Hand-built and impossible*: correlations +0.9 (A with B), +0.9 (A with C) and −0.9 (B with C).
# 6. *Round-off*: a smooth-field covariance on 30 closely spaced points (squared exponential, 2.12). In exact arithmetic
#    it is positive definite; in floating-point arithmetic its smallest eigenvalues are tiny numbers of either sign.

# %%
def eig_report(name, M):
    ev = np.linalg.eigvalsh(M)
    try:
        np.linalg.cholesky(M)
        chol = "succeeds"
    except np.linalg.LinAlgError:
        chol = "FAILS"
    print(f"{name:44s} smallest eigenvalues {np.array2string(np.sort(ev)[:3], precision=3):34s} Cholesky {chol}")
    return ev, chol


cases = {}
cases["1 perfectly correlated (x2 = 2 x1)"] = np.array([[1.0, 2.0], [2.0, 4.0]])
al = np.ones(3)
cases["2 proportions summing to 1 (Dirichlet)"] = (np.diag(al) * al.sum() - np.outer(al, al)) / (al.sum() ** 2 * (al.sum() + 1))
cases["3 from 5 samples of 10 quantities"] = np.cov(np.random.default_rng(0).standard_normal((5, 10)).T)
D2 = np.diff(np.eye(10), 2, axis=0)
cases["4 second-difference precision D^T D"] = D2.T @ D2
cases["5 hand-built correlations (+0.9, +0.9, -0.9)"] = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
z30 = np.linspace(0.0, 10.0, 30)
K30 = np.exp(-0.5 * ((z30[:, None] - z30[None, :]) / 2.0) ** 2)
cases["6 smooth field, 30 points (round-off)"] = K30
rep = {k: eig_report(k, M) for k, M in cases.items()}

# %%
fig, axs = plt.subplots(1, 6, figsize=(17, 3.6))
for ax, (k, (ev, chol)) in zip(axs, rep.items()):
    ev = np.sort(ev)[::-1]
    lo, stub = 1e-17, 1e-16                                      # exact zeros drawn as a short stub
    cols = [S.BLUE if v > 1e-10 * ev.max() else (S.BAD if v < -1e-10 * ev.max() else S.MUTED) for v in ev]
    ax.bar(np.arange(1, len(ev) + 1), np.maximum(np.abs(ev), stub), color=cols, width=0.7)
    ax.set_yscale("log"); ax.set_ylim(lo, max(30.0, ev.max() * 3))
    ax.set_xticks(np.arange(1, len(ev) + 1, 1 if len(ev) <= 10 else 5))
    ax.set_xlabel("eigenvalue number")
    ax.set_title(k[2:], fontsize=8.6)
    singular = bool(np.any(ev <= 1e-10 * ev.max()))           # a singular matrix is not safe even if Cholesky runs
    label = f"Cholesky {chol}" + (" (by luck of rounding)" if chol == "succeeds" and singular else "")
    S.badge(ax, label, ok=(chol == "succeeds" and not singular), loc="upper right", fs=7.5)
axs[0].set_ylabel("|eigenvalue|")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=S.BLUE, label="positive"), Patch(color=S.MUTED, label="zero up to round-off"),
                    Patch(color=S.BAD, label="clearly negative")], loc="lower center", ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1))
bk.savefig(fig, "2-06_when_not_pd")
plt.show()

# %% [markdown]
# **How to read it.** Each panel shows the size of every eigenvalue on a log scale; grey bars are zero up to round-off,
# red ones clearly negative. Cases 1 to 4 are positive semi-definite: valid covariances (or precisions) with exactly
# zero variance in some direction, because something is fixed or free. Case 5 has a clearly negative eigenvalue: not a
# covariance at all. Case 6 should be positive definite but has eigenvalues at round-off level, of both signs.
# Exact zeros are drawn as short stubs. The badges show what that means in practice: the Cholesky factorisation (section 5) fails for some singular cases
# and, by luck of rounding, succeeds for others (case 2), so success alone does not prove a matrix is safely positive
# definite. Section 5e shows how the failure happens and the usual fix.
#
# **Common misreadings.**
# * *"The standard deviations on the diagonal are the uncertainty."* They are the uncertainty of each quantity alone.
#   The combination along the long axis of (a) is far less certain than either diagonal entry suggests, and the one
#   along the short axis far more certain (eigenvalues printed).
# * *"Any symmetric matrix of the right size is a covariance."* It must also be positive semi-definite (case 5).
# * *"A singular covariance is an error."* Often it is a true statement: a constraint (case 2) or a free pattern
#   (case 4). It only becomes a problem when the code needs an inverse or a Cholesky factor.

# %% [markdown]
# ## 5. The Cholesky factorisation: a square root of a covariance
#
# ### 5a. What it is
#
# **In plain words.** A positive number has a square root. A positive-definite covariance matrix has a "square root"
# too, the **Cholesky factor** $\mathbf L$: a lower-triangular matrix (zeros above the diagonal) with
# $\mathbf C = \mathbf L\mathbf L^\top$. It is built one quantity at a time. The first diagonal entry is the standard
# deviation of the first quantity. Each later diagonal entry is the standard deviation that is *left* for that quantity
# once the parts it shares with the earlier quantities are taken out. That left-over variance, computed at each step,
# is called the **pivot**; the factorisation takes its square root, so every pivot must be positive.
#
# **Precisely.** For $j = 1,\dots,n$: the pivot is $p_j = C_{jj} - \sum_{k<j}L_{jk}^2$, then $L_{jj} = \sqrt{p_j}$ and
# $L_{ij} = (C_{ij} - \sum_{k<j}L_{ik}L_{jk})/L_{jj}$ for $i > j$. The pivot $p_j$ equals the variance of quantity $j$
# conditional on quantities $1,\dots,j-1$ (conditioning, 2.04). It is positive for every $j$ exactly when $\mathbf C$ is
# positive definite. Why it matters: one factorisation, computed once, gives correlated random draws (5b), whitening
# (5c), linear solves and the log-determinant (5d); and its failure is the standard alarm for an invalid covariance (5e).
#
# **Worked example.** Noise at four GPS stations at 3, 10, 18 and 30 km along a profile, with standard deviation 2 mm
# and a correlation that decays with distance, $C_{ij} = (2\ \text{mm})^2 e^{-\lvert x_i - x_j\rvert/20\ \text{km}}$
# (an exponential covariance; 2.14 builds such models from real noise).

# %%
def chol_with_pivots(C):
    """Plain Cholesky, column by column, returning L, the pivots, and whether it finished."""
    n = len(C)
    L = np.zeros_like(C, dtype=float)
    piv = []
    for j in range(n):
        p = C[j, j] - L[j, :j] @ L[j, :j]
        piv.append(p)
        if not p > 0:                                         # the square root of the pivot is impossible: stop
            return L, np.array(piv), False
        L[j, j] = np.sqrt(p)
        for i in range(j + 1, n):
            L[i, j] = (C[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L, np.array(piv), True


x4 = np.array([3.0, 10.0, 18.0, 30.0])
expcov = lambda x, lam_, s: s ** 2 * np.exp(-np.abs(x[:, None] - x[None, :]) / lam_)
C4 = expcov(x4, 20.0, 2.0)
L4, piv4, ok4 = chol_with_pivots(C4)
print("C (mm^2):\n", C4)
print("L (mm), lower triangular:\n", L4)
print("pivots (left-over variances, mm^2):", piv4, "  all > 0:", ok4)
print("check L L^T = C:", np.allclose(L4 @ L4.T, C4), "  same as numpy's cholesky:", np.allclose(L4, np.linalg.cholesky(C4)))
print("L_11 = sd of station 1 =", L4[0, 0], "mm;  station 2: total sd 2 mm, left-over sd after station 1 =", L4[1, 1], "mm")

# %%
fig, axs = plt.subplots(1, 3, figsize=(14, 4.1), gridspec_kw=dict(width_ratios=[1, 1, 1]))
lab4 = [f"{x:.0f} km" for x in x4]
for ax, M, ttl, cm in [(axs[0], C4, "(a) C (mm²): symmetric", "Blues"), (axs[1], L4, "(b) L (mm): zero above the diagonal", "Oranges"),
                       (axs[2], L4 @ L4.T, "(c) L Lᵀ (mm²): equals C", "Blues")]:
    ax.imshow(M, cmap=cm, vmin=0, vmax=max(C4.max(), L4.max()) * 1.1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color=S.INK if M[i, j] < 0.6 * C4.max() else S.SURFACE)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab4); ax.set_yticks(range(4)); ax.set_yticklabels(lab4)
    ax.set_title(ttl)
fig.tight_layout()
bk.savefig(fig, "2-06_cholesky_factor")
plt.show()

# %% [markdown]
# **How to read it.** (a) The covariance: 4 mm² on the diagonal, smaller covariances off it, decaying with distance.
# (b) Its Cholesky factor: the upper triangle is exactly zero. The first column is the first station's influence on
# all others; down the diagonal the entries shrink because each station shares part of its variance with the stations
# before it (printed pivots). (c) Multiplying back gives the covariance again (printed check).

# %% [markdown]
# ### 5b. Use 1: making correlated noise
#
# **In plain words.** A computer easily makes independent standard-normal numbers $\mathbf z$ (mean 0, variance 1,
# no correlation). Multiplying by $\mathbf L$ mixes them so that the result has exactly the covariance $\mathbf C$:
# $\mathbf x = \boldsymbol\mu + \mathbf L\mathbf z$. This is how every prior draw, noise realisation and posterior
# draw of a Gaussian in this course is made.
#
# **Precisely.** $\mathrm{Cov}(\mathbf L\mathbf z) = \mathbf L\,\mathrm{Cov}(\mathbf z)\,\mathbf L^\top =
# \mathbf L\mathbf I\mathbf L^\top = \mathbf C$. In two dimensions, $x_1 = L_{11}z_1$ and $x_2 = L_{21}z_1 + L_{22}z_2$:
# the second quantity borrows part of the first one's randomness (that is the correlation) and adds its own.
#
# **Worked example.** Stations 1 and 2 of 5a (3 and 10 km).

# %%
C2 = C4[:2, :2]
L2 = np.linalg.cholesky(C2)
Z = np.random.default_rng(12).standard_normal((50_000, 2))          # many draws for the printed covariances
X = Z @ L2.T
z_pt = np.array([1.2, 0.8])
x_pt = L2 @ z_pt
print("C (stations 1, 2):\n", C2, "\nL:\n", L2)
print("sample covariance of z:\n", np.cov(Z.T), "\nsample covariance of x = L z:\n", np.cov(X.T))
print(f"one point: z = {z_pt} -> x = L z = [{L2[0, 0]:.3f} x {z_pt[0]}, {L2[1, 0]:.3f} x {z_pt[0]} + {L2[1, 1]:.3f} x {z_pt[1]}] = {x_pt}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
ax = axs[0]
ax.scatter(Z[:3000, 0], Z[:3000, 1], s=3, color=S.MUTED, alpha=0.35)
BP.ellipse(ax, np.zeros(2), np.eye(2), nsig=2, color=S.INK, lw=1.3, label="2-sd circle")
ax.plot(*z_pt, "o", color=S.INK, ms=8, mec=S.SURFACE, label=f"z = ({z_pt[0]}, {z_pt[1]})", zorder=5)
ax.set_xlabel("z₁"); ax.set_ylabel("z₂"); ax.set_aspect("equal"); ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
ax.set_title("(a) Independent standard normals z")
ax.legend(fontsize=7.8, loc="upper left")
ax = axs[1]
ax.scatter(X[:3000, 0], X[:3000, 1], s=3, color=S.BLUE, alpha=0.3)
BP.ellipse(ax, np.zeros(2), C2, nsig=2, color=S.INK, lw=1.3, label="2-sd ellipse of C")
ax.plot(*x_pt, "o", color=S.INK, ms=8, mec=S.SURFACE, label="the same point, x = L z", zorder=5)
y_b = L2[1, 0] * z_pt[0]                                   # the part of x2 borrowed from z1
ax.annotate("", xy=(x_pt[0], 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=S.AQUA, lw=1.6))
ax.annotate("", xy=(x_pt[0], y_b), xytext=(x_pt[0], 0), arrowprops=dict(arrowstyle="-|>", color=S.ORANGE, lw=1.6, ls="--"))
ax.annotate("", xy=x_pt, xytext=(x_pt[0], y_b), arrowprops=dict(arrowstyle="-|>", color=S.VIOLET, lw=1.6))
ax.plot([], [], color=S.AQUA, lw=1.6, label="x₁ = L₁₁ z₁")
ax.plot([], [], color=S.ORANGE, lw=1.6, ls="--", label="x₂, borrowed: L₂₁ z₁")
ax.plot([], [], color=S.VIOLET, lw=1.6, label="x₂, its own: L₂₂ z₂")
ax.set_xlabel("noise at station 1, 3 km (mm)"); ax.set_ylabel("noise at station 2, 10 km (mm)")
ax.set_aspect("equal"); ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.set_title("(b) Correlated noise x = L z")
ax.legend(fontsize=7.6, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-06_cholesky_sampling")
plt.show()

# %% [markdown]
# **How to read it.** (a) The round cloud of independent standard normals. (b) The same draws after multiplication by
# $\mathbf L$: a tilted ellipse with the covariance $\mathbf C$ (printed sample covariances). The black point shows the
# mechanics: its first coordinate is only scaled (aqua); its second coordinate borrows from the first standard normal
# (orange, dashed: the shared part, which is what makes the noise correlated) and then adds its own (violet).

# %% [markdown]
# ### 5c. Use 2: whitening
#
# **In plain words.** Correlated noise is hard to think about; independent noise of unit size is easy. **Whitening**
# runs 5b backwards: $\mathbf L^{-1}\mathbf r$ turns correlated residuals into independent ones with standard deviation
# 1 (the name comes from "white noise", noise without correlation). The misfit is then just a sum of squares.
#
# **Precisely.** If $\mathbf e\sim\mathcal N(\mathbf 0,\mathbf C)$ then $\mathbf L^{-1}\mathbf e\sim\mathcal N(\mathbf 0,\mathbf I)$,
# and $\chi^2 = \mathbf r^\top\mathbf C^{-1}\mathbf r = \lVert\mathbf L^{-1}\mathbf r\rVert^2$. The product
# $\mathbf L^{-1}\mathbf r$ is computed by a **triangular solve** of $\mathbf L\mathbf y = \mathbf r$: because
# $\mathbf L$ is triangular, its first equation has one unknown, the second two (one already known), and so on
# ("forward substitution"); no inverse is formed.
#
# **Worked example.** GPS noise at three stations with an independent part (2 mm per station) and a common-mode part
# shared by all three (3 mm), as happens when a reference-frame error moves the whole network.

# %%
s_w, s_c = 2.0, 3.0
Cd_c = s_w ** 2 * np.eye(3) + s_c ** 2 * np.ones((3, 3))
L = np.linalg.cholesky(Cd_c)
E = np.random.default_rng(11).standard_normal((5000, 3)) @ L.T          # 5000 noise vectors with covariance Cd_c
W = sl.solve_triangular(L, E.T, lower=True).T                             # whitened
print("C_d with common mode (mm^2):\n", Cd_c)
print("Cholesky factor L (mm):\n", L)
print("sample covariance of the raw noise:\n", np.cov(E.T))
print("sample covariance after whitening (should be close to I):\n", np.cov(W.T))
rr = np.array([4.0, 5.0, 3.0])
print(f"chi^2 of a residual {rr.tolist()} mm: with the inverse {rr @ np.linalg.inv(Cd_c) @ rr:.6f}, "
      f"with a triangular solve {np.sum(sl.solve_triangular(L, rr, lower=True) ** 2):.6f}, "
      f"pretending the noise is independent (sigma 2 mm) {np.sum((rr / s_w) ** 2):.6f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, Xw, ttl, unit in [(axs[0], E, "(a) Raw noise: stations share a common error", "mm"),
                          (axs[1], W, "(b) Whitened: L⁻¹e is independent, sd 1", "units of sd")]:
    ax.scatter(Xw[:2000, 0], Xw[:2000, 1], s=3, color=S.MUTED, alpha=0.4)
    BP.ellipse(ax, np.zeros(2), np.cov(Xw[:, :2].T), nsig=2, color=S.BLUE, lw=1.5, label="2-sd ellipse of the sample")
    ax.set_xlabel(f"noise at station 1 ({unit})"); ax.set_ylabel(f"noise at station 2 ({unit})")
    ax.set_title(ttl); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7.8, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-06_whitening")
plt.show()

# %% [markdown]
# **How to read it.** (a) The noise at stations 1 and 2 is strongly correlated: the common mode moves both together.
# (b) After the triangular solve the cloud is round with unit spread (sample covariances printed). The printed χ²
# values show what is at stake: the same residual is scored very differently when the common mode is ignored, because a
# residual that is similar at all stations is cheap under the true covariance (it looks like the common mode) but
# expensive under independent noise.
#
# **Common misreadings.**
# * *"Whitening changes the data."* It changes the coordinates in which residuals are measured, not the information:
#   the χ² is identical with the inverse and with the triangular solve (printed).
# * *"A covariance is just a list of standard deviations."* Here every station has the same diagonal entry, yet
#   ignoring the off-diagonal common mode changes the misfit of the printed residual by a large factor.

# %% [markdown]
# ### 5d. Use 3: solving systems and the log-determinant
#
# **In plain words.** Solving $\mathbf C\mathbf x = \mathbf b$ (needed for every $\mathbf C^{-1}\mathbf b$ in a
# formula) takes two cheap triangular solves once $\mathbf L$ is known. The determinant comes for free as well: it is
# the product of the squared diagonal entries of $\mathbf L$, and its logarithm is a sum, which never overflows.
#
# **Precisely.** $\mathbf C\mathbf x = \mathbf b \iff \mathbf L(\mathbf L^\top\mathbf x) = \mathbf b$: solve
# $\mathbf L\mathbf y = \mathbf b$ by forward substitution, then $\mathbf L^\top\mathbf x = \mathbf y$ by back
# substitution (the same idea from the last row upwards). Since $\det\mathbf L = \prod_i L_{ii}$ (triangular) and
# $\det\mathbf C = (\det\mathbf L)^2$: $\ln\det\mathbf C = 2\sum_i\ln L_{ii}$, the term needed in the Gaussian
# log-likelihood (section 8).

# %%
b4 = np.array([1.0, -2.0, 0.5, 3.0])
y4 = sl.solve_triangular(L4, b4, lower=True)                 # forward substitution
x4s = sl.solve_triangular(L4.T, y4, lower=False)             # back substitution
print("solve C x = b with two triangular solves:", x4s)
print("np.linalg.solve(C, b):                    ", np.linalg.solve(C4, b4))
print(f"ln det C: 2 sum ln L_ii = {2 * np.sum(np.log(np.diag(L4))):.10f},  slogdet = {np.linalg.slogdet(C4)[1]:.10f},  "
      f"ln(det C) = {np.log(np.linalg.det(C4)):.10f}")

# %% [markdown]
# ### 5e. How it fails, and the jitter fix
#
# **In plain words.** The factorisation stops the moment a pivot (a left-over variance) is zero or negative, because it
# cannot take the square root. A zero pivot means some quantity is completely determined by the earlier ones (the
# matrix is singular). A negative pivot means the matrix claims a negative variance: it is not a covariance. A third
# failure is purely numerical. Computers store numbers with about 16 significant digits, and every operation rounds the
# result (**round-off**). When a pivot should be tiny but positive, round-off can push it below zero. The usual fix is
# **jitter**: add a tiny number $\varepsilon$ to the diagonal before factorising.
#
# **Precisely.** Adding $\varepsilon\mathbf I$ is the same as adding independent noise of variance $\varepsilon$ to every
# quantity. It raises every eigenvalue by $\varepsilon$ and every pivot by at least $\varepsilon$, so the factorisation
# succeeds. It is harmless when $\varepsilon$ is far below every variance that matters to the problem (its effect on a
# covariance is exactly $\varepsilon$ per variance), and it hides a real error when it is used to "fix" a matrix that is invalid
# (a clearly negative eigenvalue) rather than one that is merely nearly singular.
#
# **Worked example.** Pivots for four matrices: the station covariance of 5a; the same with a fifth station 1 m away
# from the second one (two almost identical stations, so almost nothing is left over for the duplicate); the
# impossible hand-built correlation matrix of section 4; and the 30-point smooth-field covariance of section 4,
# without and with jitter.

# %%
x5 = np.array([3.0, 10.0, 10.001, 18.0, 30.0])
C5 = expcov(x5, 20.0, 2.0)
pv = {}
for nm, M in [("station covariance (5a)", C4), ("with a station 1 m from another", C5),
              ("hand-built, impossible (section 4)", cases["5 hand-built correlations (+0.9, +0.9, -0.9)"])]:
    _, p_, ok_ = chol_with_pivots(M)
    pv[nm] = (p_, ok_)
    print(f"{nm:38s} pivots {np.array2string(p_, precision=4):56s} finished: {ok_};  condition number {np.linalg.cond(M):.3g}")
eps_j = 1e-10
_, pK, okK = chol_with_pivots(K30)
_, pKj, okKj = chol_with_pivots(K30 + eps_j * np.eye(30))
print(f"\nsmooth field, 30 points: smallest eigenvalue {np.linalg.eigvalsh(K30).min():.2e}; Cholesky stops at step {len(pK)} of 30 "
      f"(pivot {pK[-1]:.2e}); numpy: {rep['6 smooth field, 30 points (round-off)'][1]}")
print(f"with jitter eps = {eps_j:g}: finished {okKj}, smallest pivot {pKj.min():.2e}; jitter sd sqrt(eps) = {np.sqrt(eps_j):.0e} "
      f"against the field's sd 1")
Lj = np.linalg.cholesky(K30 + eps_j * np.eye(30))
print(f"what the jitter changed: largest |L L^T - C| = {np.abs(Lj @ Lj.T - K30).max():.1e} (= eps, on the diagonal only); "
      f"sd of each point {np.sqrt(1 + eps_j):.12f} instead of 1")

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
offs = [-0.25, 0.0, 0.25]
for (nm, (p_, ok_)), c, o in zip(pv.items(), [S.BLUE, S.ORANGE, S.BAD], offs):
    steps = np.arange(1, len(p_) + 1)
    ax.bar(steps + o, np.abs(p_), 0.24, color=c, alpha=0.85, label=nm + ("" if ok_ else ": stops"))
    for s_, v in zip(steps, p_):
        if v <= 0:
            ax.text(s_ + o, abs(v) * 1.6, "negative", ha="center", fontsize=7.5, color=S.BAD, rotation=90, va="bottom")
ax.set_yscale("log"); ax.set_ylim(1e-5, 500)
ax.set_xlabel("step j of the factorisation"); ax.set_ylabel("|pivot| (left-over variance)")
ax.set_title("(a) Pivots: positive, nearly zero, negative")
ax.legend(fontsize=7.4, loc="upper right")
ax = axs[1]
lo = 1e-17
st_ = np.arange(1, len(pK) + 1)
ax.semilogy(st_, np.maximum(np.abs(pK), lo), "o-", color=S.BAD, ms=4, lw=1.2, label="no jitter")
neg = pK <= 0
ax.semilogy(st_[neg], np.maximum(np.abs(pK[neg]), lo), "X", color=S.BAD, ms=11, label="negative pivot: stops")
ax.semilogy(np.arange(1, 31), pKj, "s-", color=S.BLUE, ms=3.5, lw=1.2, label=f"jitter ε = {eps_j:g}: finishes")
ax.axhline(eps_j, color=S.MUTED, lw=0.8, ls=":")
ax.text(29.5, eps_j * 2.2, "ε", ha="right", fontsize=8.5, color=S.INK2)
ax.set_xlabel("step j"); ax.set_ylabel("|pivot|")
ax.set_title("(b) Round-off on a smooth-field covariance")
ax.legend(fontsize=7.6, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-06_cholesky_pivots")
plt.show()

# %% [markdown]
# **How to read it.** (a) For the ordinary station covariance every pivot is a healthy left-over variance. Adding a
# station 1 m from another leaves almost nothing for the duplicate (a pivot near zero, printed): still positive, but the
# matrix is nearly singular (huge condition number). For the impossible matrix the third pivot is negative and the
# factorisation stops. (b) The smooth-field covariance should be positive definite, but its pivots shrink quickly
# (neighbouring points are almost perfectly correlated) until round-off makes one negative and the factorisation
# stops. With a jitter of $10^{-10}$ every pivot stays above $\varepsilon$ (dotted line) and the factorisation
# finishes; the covariance it factorises differs from the original by exactly $\varepsilon$ on the diagonal (printed).
# The factor itself can differ a lot between two jitter sizes, because in nearly singular directions very different
# factors give almost the same covariance: compare covariances or distributions, never individual draws.
#
# **Common misreadings.**
# * *"Jitter is a hack that changes the model."* It does change the model, by adding a tiny independent noise term.
#   Choose it far below every variance that matters, and say so.
# * *"If Cholesky fails, add jitter."* First look at the eigenvalues: a clearly negative one (case 5 of section 4) means
#   the matrix is wrong, and jitter would only hide that.

# %% [markdown]
# ## 6. Solve, do not invert
#
# **In plain words.** The **least-squares** answer to $\mathbf d = \mathbf G\mathbf m + \mathbf e$ is the $\mathbf m$
# that makes the sum of squared residuals smallest (2.07 derives it and explains why). On paper it is
# $(\mathbf G^\top\mathbf G)^{-1}\mathbf G^\top\mathbf d$, the solution of the **normal equations**
# $\mathbf G^\top\mathbf G\,\mathbf m = \mathbf G^\top\mathbf d$. In a computer, forming that inverse, or even forming
# $\mathbf G^\top\mathbf G$, throws away accuracy when the problem is close to singular, which inverse problems usually
# are. The cure costs nothing: ask the library to *solve* the system, and solve the least-squares problem from
# $\mathbf G$ itself.
#
# **Precisely.** The **condition number** $\kappa(\mathbf G) = s_{\max}/s_{\min}$, the ratio of the largest to the
# smallest singular value (section 7; for a covariance, the ratio of its largest to smallest eigenvalue), measures how
# close a matrix is to singular, and bounds how much relative round-off error (about $\epsilon\approx 2\times10^{-16}$
# in double precision) can grow in the answer. Forming $\mathbf G^\top\mathbf G$ squares it:
# $\kappa(\mathbf G^\top\mathbf G) = \kappa(\mathbf G)^2$. The **QR factorisation** writes $\mathbf G = \mathbf Q\mathbf R$
# with orthonormal columns in $\mathbf Q$ (they do not stretch anything) and an upper-triangular $\mathbf R$, so the
# least-squares problem becomes the triangular system $\mathbf R\mathbf m = \mathbf Q^\top\mathbf d$. Methods that work
# on $\mathbf G$ directly (QR or SVD: `np.linalg.lstsq`, `scipy.linalg.lstsq`) keep the error near $\kappa\,\epsilon$;
# the normal equations give about $\kappa^2\epsilon$, whether solved by an explicit inverse or by Cholesky. Separately,
# an explicit inverse leaves a larger residual $\lVert\mathbf A\mathbf x - \mathbf b\rVert$ than a solve of the same
# system.
#
# **Worked example.** Test matrices of 40 data and 6 unknowns, built with chosen singular values so that
# $\kappa(\mathbf G)$ runs from 10 to $10^8$; noise-free data from a known $\mathbf m$, so the error of each method
# is exact.

# %%
rng5 = np.random.default_rng(0)
Nd5, Nm5 = 40, 6
Q1, _ = np.linalg.qr(rng5.standard_normal((Nd5, Nm5)))
Q2, _ = np.linalg.qr(rng5.standard_normal((Nm5, Nm5)))
m5 = rng5.standard_normal(Nm5)
kappas = 10.0 ** np.arange(1, 9)
rel = lambda x: np.linalg.norm(x - m5) / np.linalg.norm(m5)
res = {k: [] for k in ["normal eq., explicit inverse", "normal eq., Cholesky solve", "least squares on G (QR)", "least squares on G (SVD, lstsq)"]}
bres = {"explicit inverse": [], "solve": []}
print(" kappa(G)  inverse(GtG)  Cholesky(GtG)   QR on G    lstsq on G  | residual of GtG x = Gtd: inverse   solve")
for kap in kappas:
    sv5 = np.logspace(0, -np.log10(kap), Nm5)
    G5 = Q1 @ np.diag(sv5) @ Q2.T
    d5 = G5 @ m5
    A5, b5 = G5.T @ G5, G5.T @ d5
    x_inv = np.linalg.inv(A5) @ b5
    try:
        x_cho = sl.cho_solve(sl.cho_factor(A5), b5)
    except np.linalg.LinAlgError:
        x_cho = np.full(Nm5, np.nan)
    q5, r5 = np.linalg.qr(G5)
    x_qr = sl.solve_triangular(r5, q5.T @ d5)
    x_ls = np.linalg.lstsq(G5, d5, rcond=None)[0]
    for k_, x_ in zip(res, [x_inv, x_cho, x_qr, x_ls]):
        res[k_].append(rel(x_))
    nb = lambda x: np.linalg.norm(A5 @ x - b5) / (np.linalg.norm(A5, 2) * np.linalg.norm(x))
    bres["explicit inverse"].append(nb(x_inv)); bres["solve"].append(nb(np.linalg.solve(A5, b5)))
    print(f"  {kap:7.0e}  {res['normal eq., explicit inverse'][-1]:11.1e}  {res['normal eq., Cholesky solve'][-1]:12.1e}  "
          f"{res['least squares on G (QR)'][-1]:9.1e}  {res['least squares on G (SVD, lstsq)'][-1]:10.1e}  | "
          f"{bres['explicit inverse'][-1]:24.1e}  {bres['solve'][-1]:7.1e}")

# %%
eps = np.finfo(float).eps
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axs[0]
styles = {"normal eq., explicit inverse": (S.BAD, "o-"), "normal eq., Cholesky solve": (S.ORANGE, "s--"),
          "least squares on G (QR)": (S.BLUE, "^-"), "least squares on G (SVD, lstsq)": (S.AQUA, "v--")}
for k_, v in res.items():
    c, mk = styles[k_]
    ax.loglog(kappas, np.maximum(v, 1e-17), mk, color=c, lw=1.4, ms=5, label=k_)
ax.loglog(kappas, kappas * eps, ":", color=S.MUTED, lw=1, label="κ ε")
ax.loglog(kappas, kappas ** 2 * eps, ":", color=S.INK2, lw=1, label="κ² ε")
ax.set_ylim(1e-17, 10)
ax.set_xlabel("condition number κ(G)"); ax.set_ylabel("relative error of the solution")
ax.set_title("(a) Normal equations lose twice the digits")
ax.legend(fontsize=7.4, loc="upper left")
ax = axs[1]
ax.loglog(kappas, bres["explicit inverse"], "o-", color=S.BAD, label="x = inv(A) b")
ax.loglog(kappas, bres["solve"], "s-", color=S.BLUE, label="x = solve(A, b)")
ax.set_xlabel("condition number κ(G)  (A = GᵀG)"); ax.set_ylabel("relative residual ‖Ax − b‖ / (‖A‖ ‖x‖)")
ax.set_title("(b) An explicit inverse leaves a larger residual")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-06_solve_vs_inverse")
plt.show()

# %% [markdown]
# **How to read it.** (a) The two normal-equation methods stay near the κ²ε line, the two methods that work on
# $\mathbf G$ stay near or below κε (numbers printed). At the largest κ shown, the normal equations have lost
# essentially all digits while QR and SVD still give many. For still larger κ, round-off can make
# $\mathbf G^\top\mathbf G$ numerically indefinite, and then its Cholesky factorisation fails outright (5e).
# (b) Even for the same normal-equation system, multiplying by an explicit inverse leaves a residual that grows with
# κ, while a solve keeps it at round-off level.
#
# **Common misreadings.**
# * *"Matrix inversion is how linear systems are solved."* Libraries solve by factorisation (LU, Cholesky, QR); an
#   explicit inverse is slower and less accurate. Write `np.linalg.solve(A, b)`, `scipy.linalg.cho_solve`, or
#   `np.linalg.lstsq(G, d)`.
# * *"The formulas in a paper say $(\mathbf G^\top\mathbf G)^{-1}$, so the code must form it."* The formula says what
#   is computed, not how. Small matrices (the 2 × 2 of section 4) are harmless to invert; regularisation (2.09) also
#   improves the condition number, which is part of why it helps.
#
# In a real slip inversion the condition number of $\mathbf G$ is often large because deep patches leave nearly
# identical, weak fingerprints; the next section shows the extreme case.

# %% [markdown]
# ## 7. The SVD: rank, null space, and what the data cannot see
#
# **In plain words.** The **singular value decomposition** splits any matrix, square or not, into independent channels:
# a slip pattern goes in, a data pattern comes out, multiplied by a gain (the **singular value**). Channels with large
# gain are well measured, channels with tiny gain are drowned in noise, and when there are more unknowns than data some
# slip patterns produce no data at all. Those invisible patterns form the **null space**.
#
# **Precisely.** $\mathbf G = \mathbf U\boldsymbol\Sigma\mathbf V^\top$: columns of $\mathbf V$ (slip patterns) and of
# $\mathbf U$ (data patterns) are orthonormal, $\boldsymbol\Sigma$ holds the singular values $s_1\ge s_2\ge\dots\ge 0$,
# and $\mathbf G\mathbf v_i = s_i\mathbf u_i$. The singular values are the square roots of the eigenvalues of
# $\mathbf G^\top\mathbf G$. The **rank** is the number of nonzero $s_i$: the number of independent combinations of
# the unknowns that the data measure; it cannot exceed $\min(N_d, N_m)$. Every $\mathbf v_i$ with $s_i = 0$, and every
# combination of them, satisfies $\mathbf G\mathbf v = \mathbf 0$: that set is the null space. 2.07 uses the same
# decomposition for the least-squares solution and for truncated SVD.
#
# **Worked example.** The same three stations, now with five patches from 0 to 25 km.

# %%
z5 = np.linspace(0.0, 25.0, 6)
G5f = 1000.0 * np.array([[patch_disp(x, z5[j], z5[j + 1]) for j in range(5)] for x in x_st])
U6, s6, Vt6 = np.linalg.svd(G5f)
rank6 = int(np.sum(s6 > s6[0] * 1e-12))
print("G (3 x 5), mm per m:\n", G5f)
print("singular values (mm per m):", s6, f"  rank = {rank6}; condition number of the nonzero part {s6[0] / s6[rank6 - 1]:.1f}")
null = Vt6[rank6:]                                                     # (5 - rank) x 5
print(f"null space: {null.shape[0]} independent slip patterns; |G v| for each:", np.abs(G5f @ null.T).max(0))
m_a = np.array([1.0, 2.0, 1.5, 0.5, 0.2])
m_b = m_a + 3.0 * null[0]
print("two very different slip models:", m_a, "and", np.round(m_b, 3))
print("their predictions (mm):", G5f @ m_a, "and", G5f @ m_b)

# %%
zc5 = 0.5 * (z5[1:] + z5[:-1])
fig, axs = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
ax = axs[0]
ax.semilogy(np.arange(1, len(s6) + 1), s6, "o-", color=S.INK, ms=6)
ax.set_xticks(np.arange(1, len(s6) + 1))
ax.set_xlabel("channel i"); ax.set_ylabel("singular value sᵢ (mm per m)")
ax.set_title(f"(a) {len(s6)} channels, rank {rank6}")
ax = axs[1]
for i, c in zip(range(rank6), [S.BLUE, S.ORANGE, S.AQUA]):
    ax.plot(Vt6[i], zc5, "o-", color=c, lw=1.5, ms=4, label=f"v{i + 1} (gain {s6[i]:.0f})")
for k in range(null.shape[0]):
    ax.plot(null[k], zc5, "s--", color=S.MUTED if k else S.INK2, lw=1.3, ms=4, label=f"null pattern {k + 1} (gain 0)")
ax.axvline(0, color=S.AXIS, lw=0.8)
ax.invert_yaxis(); ax.set_xlabel("slip pattern (unit length)"); ax.set_ylabel("depth (km)")
ax.set_title("(b) Slip patterns: seen and invisible")
ax.legend(fontsize=7.2, loc="lower left")
ax = axs[2]
ax.plot(m_a, zc5, "o-", color=S.BLUE, lw=1.6, label="model A")
ax.plot(m_b, zc5, "s-", color=S.ORANGE, lw=1.6, label="model B = A + 3 × null pattern 1")
ax.invert_yaxis(); ax.set_xlabel("slip (m)"); ax.set_ylabel("depth (km)")
ax.set_title("(c) Different slip, identical data")
ax.legend(fontsize=7.8, loc="lower right")
fig.tight_layout()
bk.savefig(fig, "2-06_svd_nullspace")
plt.show()

# %% [markdown]
# **How to read it.** (a) Three stations allow at most three channels; the other two singular values are zero to
# round-off. (b) The three seen patterns (coloured) and the two null patterns (grey, dashed). (c) Two slip models that
# differ by a null pattern predict the same three displacements to round-off (printed): the data cannot choose between
# them. Something else must, and that something is the prior (2.08, 2.09, 2.11).
#
# **Common misreadings.**
# * *"With enough computing power we can solve for five patches from three stations."* No amount of computing changes
#   the rank; the null space is a property of the experiment.
# * *"A null-space pattern is noise."* It is any slip, possibly large, that the stations cannot see.

# %% [markdown]
# ## 8. The log-determinant: the Gaussian likelihood without overflow (optional)
#
# **In plain words.** The Gaussian log-likelihood has two parts: the misfit, and a term that depends on the size of
# the noise, $-\tfrac12\ln\det(2\pi\mathbf C_d)$. The second part matters whenever the noise level is estimated
# (hierarchical models, 2.30). The determinant of a big covariance is an astronomically large or small number that a
# computer cannot store; its logarithm is an ordinary number.
#
# **Precisely.**
# $\ln p(\mathbf d\mid\mathbf m) = -\tfrac12\mathbf r^\top\mathbf C_d^{-1}\mathbf r - \tfrac12\ln\det\mathbf C_d - \tfrac{N_d}{2}\ln 2\pi$.
# With the Cholesky factor, $\ln\det\mathbf C_d = 2\sum_i\ln L_{ii}$ (5d); `np.linalg.slogdet` returns the same safely.
# Computing `np.log(np.linalg.det(C))` instead underflows to $-\infty$ or overflows to $+\infty$ once $N_d$ is large
# (**underflow**: a number too small to store becomes 0; **overflow**: too large becomes infinity), and which one
# happens depends on the units.
#
# **Worked example.** Independent noise of σ = 2 mm at $N_d$ stations, written in metres and in millimetres.

# %%
print("  N_d   units   det(C)        log(det(C))      slogdet         2 sum log L_ii    exact N ln sigma^2")
rows7 = []
for Nn in [10, 100, 300, 1000, 3000]:
    for unit, sg in [("m", 2e-3), ("mm", 2.0)]:
        C7 = sg ** 2 * np.eye(Nn)
        with np.errstate(divide="ignore", over="ignore"):
            dt = np.linalg.det(C7)
            ld_naive = np.log(dt)
        ld_slog = np.linalg.slogdet(C7)[1]
        ld_chol = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(C7))))
        rows7.append((Nn, unit, ld_naive, ld_slog))
        print(f"{Nn:5d}   {unit:4s}  {dt:11.3e}   {ld_naive:13.4f}   {ld_slog:13.4f}   {ld_chol:13.4f}   {Nn * np.log(sg ** 2):13.4f}")

# %%
Ns = sorted({r_[0] for r_ in rows7})
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
for unit, c, mk in [("m", S.ORANGE, "o"), ("mm", S.BLUE, "s")]:
    good = [r_[3] for r_ in rows7 if r_[1] == unit]
    naive = [r_[2] for r_ in rows7 if r_[1] == unit]
    ax.plot(Ns, good, mk + "-", color=c, lw=1.5, ms=5, label=f"slogdet, σ in {unit}")
    fin = [(n_, v) for n_, v in zip(Ns, naive) if np.isfinite(v)]
    ax.plot([f[0] for f in fin], [f[1] for f in fin], mk, color=S.INK, ms=11, mfc="none", mew=1.2)
ax.plot([], [], "o", color=S.INK, ms=9, mfc="none", label="log(det(C)), where finite")
ax.axhline(0, color=S.AXIS, lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("number of data N_d"); ax.set_ylabel("ln det C_d")
ax.set_title("(a) ln det C_d grows in proportion to N_d")
ax.legend(fontsize=7.8, loc="lower left")
ax = axs[1]
for row, unit in enumerate(["m", "mm"]):
    for col, n_ in enumerate(Ns):
        v = [r_[2] for r_ in rows7 if r_[1] == unit and r_[0] == n_][0]
        ok = bool(np.isfinite(v))
        txt = "finite" if ok else ("det = 0\nlog = −inf" if v < 0 else "det = inf\nlog = +inf")
        ax.text(col, row, ("✓ " if ok else "✗ ") + txt, ha="center", va="center", fontsize=8,
                color=S.INK, bbox=dict(boxstyle="round,pad=0.35", fc=S.SURFACE, ec=S.GOOD if ok else S.BAD, lw=1))
ax.set_xlim(-0.6, len(Ns) - 0.4); ax.set_ylim(1.6, -0.6)
ax.set_xticks(range(len(Ns))); ax.set_xticklabels([str(n_) for n_ in Ns])
ax.set_yticks([0, 1]); ax.set_yticklabels(["σ = 0.002 m", "σ = 2 mm"])
ax.set_xlabel("number of data N_d")
for sp in ["left", "bottom"]:
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
ax.set_title("(b) np.log(np.linalg.det(C)): the same noise, two units")
fig.tight_layout()
bk.savefig(fig, "2-06_logdet")
plt.show()

# %% [markdown]
# **How to read it.** (a) The coloured lines (slogdet) grow in proportion to $N_d$ and agree with the exact value
# $N_d\ln\sigma^2$ (printed); the open circles mark where the plain logarithm of the determinant is still finite.
# (b) The plain route fails for the same noise level in both units: in metres the determinant underflows to 0, in
# millimetres it overflows to infinity, at the station counts marked with a cross (printed table).
#
# **Common misreadings.**
# * *"The determinant term is a constant, so it never matters."* It is constant only while $\mathbf C_d$ is fixed.
#   When a noise scale or a correlation length is estimated, it is the term that stops the noise from growing without
#   limit (2.29, 2.30).

# %% [markdown]
# ## Summary: from formula to code
#
# | on paper | in code | why |
# |---|---|---|
# | $\mathbf G\mathbf m$ | `G @ m` | check shapes $(N_d\times N_m)(N_m)$ and units |
# | $\mathbf A^{-1}\mathbf b$ | `np.linalg.solve(A, b)` | an explicit inverse is slower and less accurate |
# | $(\mathbf G^\top\mathbf G)^{-1}\mathbf G^\top\mathbf d$ | `np.linalg.lstsq(G, d)` | the normal equations square κ |
# | is $\mathbf C$ a valid covariance? | `np.linalg.eigvalsh(C)` | all ≥ 0; a clearly negative one means it is not |
# | $\mathbf C = \mathbf L\mathbf L^\top$ | `L = np.linalg.cholesky(C)` (add a tiny jitter only for round-off) | square root of a covariance |
# | draw from $\mathcal N(\boldsymbol\mu,\mathbf C)$ | `mu + L @ z` | correlated noise from independent noise |
# | $\mathbf r^\top\mathbf C^{-1}\mathbf r$ | `sum(solve_triangular(L, r, lower=True)**2)` | whitening; no inverse |
# | $\ln\det\mathbf C$ | `np.linalg.slogdet(C)[1]` or `2*sum(log(diag(L)))` | no overflow |
# | rank, null space | `np.linalg.svd(G)` | what the data can and cannot see |
#
# **The image to keep.** A matrix is a table of fingerprints. Its transpose sends residuals back to the parameters, a
# covariance is read through its eigenvalues, its Cholesky factor makes, whitens and scores correlated noise, its SVD
# lists what the data can see, and in code you solve rather than invert.
#
# Next: **2.07 · Least squares**.

# %%
print(f"notebook run time: {time.time() - T0:.0f} s")
