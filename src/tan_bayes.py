from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer

@dataclass
class TANStructure:
    root: int
    parent: List[Optional[int]]  # parent feature index for each feature (None for root)
    order: List[int]             # topological order of features

class UnionFind:
    def __init__(self, n:int):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x:int)->int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x:int, y:int)->bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

class TanBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Tree-Augmented Naïve Bayes (TAN) classifier.
    - Discretizes continuous features (KBinsDiscretizer, quantiles by default).
    - Learns a Chow–Liu tree over features with conditional mutual information I(Xi; Xj | Y).
    - Y is a parent of every Xi; additionally, each Xi has at most one feature parent.
    - Uses Laplace smoothing for CPTs.
    """
    def __init__(self,
                 categorical_features: Optional[List[str]] = None,
                 numeric_features: Optional[List[str]] = None,
                 n_bins:int = 5,
                 discretizer_strategy:str = "quantile",
                 smoothing:float = 1.0,
                 root_feature: Optional[str] = None,
                 random_state: Optional[int] = 42):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.n_bins = n_bins
        self.discretizer_strategy = discretizer_strategy
        self.smoothing = smoothing
        self.root_feature = root_feature
        self.random_state = random_state

        self.feature_names_: List[str] = []
        self.classes_: np.ndarray = None
        self.class_prior_: np.ndarray = None
        self.structure_: TANStructure = None

        self._num_bins_: Dict[int, int] = {}   # per-feature number of discrete values
        self._cat_maps_: Dict[int, Dict[int,int]] = {}  # raw->index map for categorical
        self._disc_: KBinsDiscretizer = None
        self._imp_num_: SimpleImputer = None
        self._imp_cat_: SimpleImputer = None

        self._cpt_: Dict[int, np.ndarray] = {}  # feature index -> CPT array
        # shapes:
        #  - for root feature r: shape [n_classes, n_vals_r]
        #  - for others i: shape [n_classes, n_vals_parent, n_vals_i]

    # -------------------- utils --------------------
    def _check_and_prepare_columns(self, X):
        if self.feature_names_:
            return
        if isinstance(X, np.ndarray):
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
        else:
            self.feature_names_ = list(X.columns)

        if self.categorical_features is None or self.numeric_features is None:
            # heuristic: known Cleveland schema or infer by dtype/uniques
            known_cats = {"sex","cp","fbs","restecg","exang","slope","ca","thal"}
            known_nums = {"age","trestbps","chol","thalach","oldpeak"}
            cols = set(self.feature_names_)
            cats = [c for c in self.feature_names_ if c in known_cats]
            nums = [c for c in self.feature_names_ if c in known_nums]
            # anything not matched goes to numeric by default
            remain = [c for c in self.feature_names_ if c not in set(cats+nums)]
            nums += remain
            self.categorical_features = cats
            self.numeric_features = nums

    def _fit_discretizers(self, X):
        # imputers
        if isinstance(X, np.ndarray):
            raise ValueError("Provide a pandas DataFrame for named columns.")
        Xcat = X[self.categorical_features].copy()
        Xnum = X[self.numeric_features].copy()

        self._imp_cat_ = SimpleImputer(strategy="most_frequent")
        self._imp_num_ = SimpleImputer(strategy="median")
        Xcat_imp = self._imp_cat_.fit_transform(Xcat)
        Xnum_imp = self._imp_num_.fit_transform(Xnum)

        # map categorical to contiguous integers
        self._cat_maps_ = {}
        Xcat_idx_list = []
        for j, col in enumerate(self.categorical_features):
            vals = Xcat_imp[:, j]
            uniq = np.unique(vals)
            mapping = {int(v): i for i, v in enumerate(uniq)}
            self._cat_maps_[j] = mapping
            Xcat_idx_list.append(np.vectorize(lambda v: mapping[int(v)])(vals))
            self._num_bins_[j] = len(uniq)

        # discretize numeric
        if Xnum_imp.shape[1] > 0:
            self._disc_ = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal",
                                           strategy=self.discretizer_strategy)
            Xnum_disc = self._disc_.fit_transform(Xnum_imp).astype(int)
            # record bins
            for j in range(Xnum_disc.shape[1]):
                self._num_bins_[len(self.categorical_features)+j] = self.n_bins
        else:
            self._disc_ = None
            Xnum_disc = np.empty((Xnum_imp.shape[0], 0), dtype=int)

        X_disc = np.column_stack(Xcat_idx_list + [Xnum_disc] if len(Xnum_disc.shape)==2 else Xcat_idx_list)
        return X_disc  # shape [n_samples, n_features]

    def _transform_discrete(self, X):
        # impute then map to discrete space
        Xcat = X[self.categorical_features].copy()
        Xnum = X[self.numeric_features].copy()
        Xcat_imp = self._imp_cat_.transform(Xcat)
        Xnum_imp = self._imp_num_.transform(Xnum)

        Xcat_idx_list = []
        for j, col in enumerate(self.categorical_features):
            mapping = self._cat_maps_[j]
            def mapv(v):
                v = int(v)
                if v in mapping:
                    return mapping[v]
                # unseen -> map to closest index (or last)
                return max(mapping.values())
            Xcat_idx_list.append(np.vectorize(mapv)(Xcat_imp[:, j]))

        if self._disc_ is not None and Xnum_imp.shape[1] > 0:
            Xnum_disc = self._disc_.transform(Xnum_imp).astype(int)
        else:
            Xnum_disc = np.empty((Xnum_imp.shape[0], 0), dtype=int)

        X_disc = np.column_stack(Xcat_idx_list + [Xnum_disc] if Xnum_imp.shape[1] > 0 else Xcat_idx_list)
        return X_disc

    def _conditional_mutual_information(self, Xi, Xj, Y, n_xi, n_xj, n_y, eps=1e-12):
        # Xi, Xj, Y are integer arrays
        N = len(Y)
        cmi = 0.0
        # count tables
        # shape [n_y, n_xi, n_xj]
        cnt = np.zeros((n_y, n_xi, n_xj), dtype=np.int64)
        for a, b, c in zip(Y, Xi, Xj):
            cnt[a, b, c] += 1
        # marginals
        cnt_y = cnt.sum(axis=(1,2)) + eps
        for y in range(n_y):
            C_y = cnt[y] + eps
            pxz_y = C_y / (cnt_y[y])
            px_y = C_y.sum(axis=1, keepdims=True) / cnt_y[y]
            pz_y = C_y.sum(axis=0, keepdims=True) / cnt_y[y]
            ratio = pxz_y / (px_y * pz_y + eps)
            cmi += (cnt_y[y] / N) * np.sum(pxz_y * np.log(ratio + eps))
        return float(cmi)

    def _learn_structure(self, Xd, y):
        n_samples, n_features = Xd.shape
        n_classes = len(self.classes_)
        # number of values per feature
        nvals = [self._num_bins_[j] for j in range(n_features)]
        # compute conditional MI for all pairs
        W = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i+1, n_features):
                W[i, j] = W[j, i] = self._conditional_mutual_information(
                    Xd[:, i], Xd[:, j], y, nvals[i], nvals[j], n_classes
                )
        # maximum spanning tree via Kruskal on negative weights (or sort descending)
        edges = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                edges.append((W[i, j], i, j))
        edges.sort(reverse=True)  # max spanning tree

        uf = UnionFind(n_features)
        mst = []
        for w, i, j in edges:
            if uf.union(i, j):
                mst.append((i, j))
            if len(mst) == n_features - 1:
                break

        # choose root
        import numpy as _np
        rng = _np.random.RandomState(self.random_state)
        if self.root_feature is not None and self.root_feature in self.feature_names_:
            root = self.feature_names_.index(self.root_feature)
        else:
            root = rng.randint(0, n_features)

        # orient edges away from root
        adj = {i: [] for i in range(n_features)}
        for i, j in mst:
            adj[i].append(j)
            adj[j].append(i)
        parent = [None] * n_features
        order = []
        stack = [root]
        seen = set([root])
        parent[root] = None
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    parent[v] = u
                    stack.append(v)
        return TANStructure(root=root, parent=parent, order=order)

    def _estimate_cpts(self, Xd, y):
        n_samples, n_features = Xd.shape
        n_classes = len(self.classes_)
        counts_y = np.bincount(y, minlength=n_classes).astype(float)
        self.class_prior_ = (counts_y + self.smoothing) / (n_samples + self.smoothing * n_classes)

        self._cpt_ = {}
        for i in range(n_features):
            n_xi = self._num_bins_[i]
            if self.structure_.parent[i] is None:  # root Xi | Y
                table = np.zeros((n_classes, n_xi), dtype=float)
                for cls in range(n_classes):
                    mask = (y == cls)
                    Xi_vals = Xd[mask, i]
                    cnt = np.bincount(Xi_vals, minlength=n_xi).astype(float)
                    table[cls, :] = (cnt + self.smoothing) / (cnt.sum() + self.smoothing * n_xi)
                self._cpt_[i] = table
            else:
                p = self.structure_.parent[i]
                n_xp = self._num_bins_[p]
                table = np.zeros((n_classes, n_xp, n_xi), dtype=float)
                for cls in range(n_classes):
                    mask_c = (y == cls)
                    Xi_vals = Xd[mask_c, i]
                    Xp_vals = Xd[mask_c, p]
                    cnt = np.zeros((n_xp, n_xi), dtype=float)
                    for a, b in zip(Xp_vals, Xi_vals):
                        cnt[a, b] += 1.0
                    # smoothing on parent-child table
                    for a in range(n_xp):
                        row = cnt[a]
                        table[cls, a, :] = (row + self.smoothing) / (row.sum() + self.smoothing * n_xi)
                self._cpt_[i] = table

    # -------------------- sklearn API --------------------
    def fit(self, X, y):
        import pandas as pd
        if not isinstance(X, (pd.DataFrame,)):
            raise ValueError("TanBayesClassifier expects a pandas DataFrame X with named columns.")
        self._check_and_prepare_columns(X)

        # map class labels to [0..K-1]
        y = np.asarray(y).astype(int)
        classes = np.unique(y)
        # ensure consecutive 0..K-1
        mapy = {c:i for i,c in enumerate(classes)}
        y_idx = np.array([mapy[c] for c in y], dtype=int)
        self.classes_ = classes

        Xd = self._fit_discretizers(X)  # discrete matrix (cats mapped, nums binned)
        self.structure_ = self._learn_structure(Xd, y_idx)
        self._estimate_cpts(Xd, y_idx)
        return self

    def predict_proba(self, X):
        import pandas as pd
        if not isinstance(X, (pd.DataFrame,)):
            raise ValueError("TanBayesClassifier expects a pandas DataFrame X with named columns.")
        Xd = self._transform_discrete(X)
        n_samples, n_features = Xd.shape
        n_classes = len(self.classes_)

        logp = np.zeros((n_samples, n_classes), dtype=float)
        # log P(Y) + sum log P(Xi | Y, parent)
        for cls_idx in range(n_classes):
            logp[:, cls_idx] = np.log(self.class_prior_[cls_idx])
        for i in range(n_features):
            p = self.structure_.parent[i]
            if p is None:
                table = self._cpt_[i]  # [n_classes, n_xi]
                vals = Xd[:, i]
                for cls_idx in range(n_classes):
                    logp[:, cls_idx] += np.log(table[cls_idx, vals])
            else:
                table = self._cpt_[i]  # [n_classes, n_xp, n_xi]
                vals = Xd[:, i]
                pvals = Xd[:, p]
                for cls_idx in range(n_classes):
                    logp[:, cls_idx] += np.log(table[cls_idx, pvals, vals])
        # normalize
        m = logp.max(axis=1, keepdims=True)
        proba = np.exp(logp - m)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
