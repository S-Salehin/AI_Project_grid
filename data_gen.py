import numpy as np
import os

def gen_points_matrix(pts):
    """Given an array of (x,y) coords, build a 25×25 binary matrix."""
    mat = np.zeros((25,25), dtype=int)
    for x,y in pts:
        mat[x, y] = 1
    return mat

def task_A_sample(seed=None):
    """Task A: exactly 2 points → Euclidean distance."""
    if seed is not None: np.random.seed(seed)
    # choose 2 distinct cells
    idx = np.random.choice(25*25, 2, replace=False)
    xs, ys = np.unravel_index(idx, (25,25))
    pts = list(zip(xs, ys))
    mat = gen_points_matrix(pts)
    # distance
    dx, dy = xs[0]-xs[1], ys[0]-ys[1]
    label = np.hypot(dx, dy)
    return mat, label

def task_B_sample(seed=None):
    """Task B: 3–10 points → closest-pair distance."""
    if seed is not None: np.random.seed(seed)
    N = np.random.randint(3, 11)
    idx = np.random.choice(25*25, N, replace=False)
    xs, ys = np.unravel_index(idx, (25,25))
    pts = list(zip(xs, ys))
    mat = gen_points_matrix(pts)
    # compute all pairwise distances
    dists = [np.hypot(x1-x2, y1-y2)
             for i,(x1,y1) in enumerate(pts)
             for (x2,y2) in pts[i+1:]]
    label = float(min(dists))
    return mat, label

def task_C_sample(seed=None):
    """Task C: 3–10 points → farthest-pair distance."""
    if seed is not None: np.random.seed(seed)
    N = np.random.randint(3, 11)
    idx = np.random.choice(25*25, N, replace=False)
    xs, ys = np.unravel_index(idx, (25,25))
    pts = list(zip(xs, ys))
    mat = gen_points_matrix(pts)
    dists = [np.hypot(x1-x2, y1-y2)
             for i,(x1,y1) in enumerate(pts)
             for (x2,y2) in pts[i+1:]]
    label = float(max(dists))
    return mat, label

def task_D_sample(seed=None):
    """Task D: 1–10 points → count of points."""
    if seed is not None: np.random.seed(seed)
    N = np.random.randint(1, 11)
    idx = np.random.choice(25*25, N, replace=False)
    xs, ys = np.unravel_index(idx, (25,25))
    pts = list(zip(xs, ys))
    mat = gen_points_matrix(pts)
    label = N
    return mat, label

def task_E_sample(seed=None):
    """Task E: 1–10 random squares → count of squares."""
    if seed is not None: np.random.seed(seed)
    N = np.random.randint(1, 11)
    mat = np.zeros((25,25), dtype=int)
    for _ in range(N):
        s = np.random.randint(1, 13)  # square size between 1 and 12
        x = np.random.randint(0, 25 - s + 1)
        y = np.random.randint(0, 25 - s + 1)
        mat[x:x+s, y:y+s] = 1
    label = N
    return mat, label

def generate_dataset(task_fn, out_dir, n_train=800, n_test=200):
    """Generate and save train/test for a given task function."""
    os.makedirs(out_dir, exist_ok=True)
    for split, n in [('train', n_train), ('test', n_test)]:
        X = np.zeros((n,25,25), dtype=int)
        y = np.zeros((n,), dtype=float)
        for i in range(n):
            mat,label = task_fn(seed=i)  # seed ensures reproducibility
            X[i], y[i] = mat, label
        np.savez(os.path.join(out_dir, f"{split}.npz"), X=X, y=y)
        print(f"Saved {split}: X{X.shape}, y{y.shape}")

if __name__ == "__main__":
    # Choose a subfolder per task
    generate_dataset(task_A_sample, "data/taskA")
    generate_dataset(task_B_sample, "data/taskB")
    generate_dataset(task_C_sample, "data/taskC")
    generate_dataset(task_D_sample, "data/taskD")
    generate_dataset(task_E_sample, "data/taskE")
