"""MyTwin · motor de recomendación para la web (reconstruye el híbrido del notebook 04)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

PARTS = ['content', 'knn', 'svd', 'segpop', 'pop']
EVENT_W_DEFAULT = {'visto': 0.2, 'guardado': 0.5, 'descartado': -0.5, 'comprado': 1.0}
TIERS = ['bajo', 'medio', 'alto', 'sin_precio']


def _z(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return (x - x.mean()) / (x.std() + 1e-6)


def _rating_w(r):
    return np.select([r <= 1.5, r <= 2.5, r <= 3.5, r <= 4.5], [-0.8, -0.4, 0.2, 0.7], 1.0).astype(np.float32)


def _l2_rows(M):
    M = M.tocsr()
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    return (sp.diags(1.0 / np.maximum(n, 1e-12)) @ M).tocsr().astype(np.float32)


class Engine:
    def __init__(self, data_dir):
        d = Path(data_dir)
        self.cfg = json.loads((d / 'hybrid_config.json').read_text())
        self.weights = {k: float(self.cfg.get('weights', {}).get(k, 0)) for k in PARTS}
        self.event_w = self.cfg.get('event_weights', EVENT_W_DEFAULT)
        self.half_life = float(self.cfg.get('half_life_days', 365))

        # --- catálogo e índices (mismo orden que en el notebook 04) ---
        cat = pd.read_parquet(d / 'items_catalog.parquet').reset_index(drop=True)
        self.item_ids = pd.Index(cat.parent_asin)
        self.users = pd.Index(pd.read_parquet(d / 'users_index.parquet').user_id)
        self.n_items, self.n_users = len(self.item_ids), len(self.users)
        self.title = cat.title.fillna('').astype(str).to_numpy()
        self.title_l = np.array([t.lower() for t in self.title])
        self.brand = cat.brand.fillna('desconocida').astype(str).to_numpy()
        self.platform = cat.platform.fillna('desconocida').astype(str).to_numpy()
        self.price = cat.price_num.to_numpy(dtype=float)
        self.tier = cat.price_tier.fillna('sin_precio').astype(str).to_numpy()

        # --- modelos guardados ---
        self.F = sp.load_npz(d / 'item_content_F.npz').tocsr().astype(np.float32)
        self.W = sp.load_npz(d / 'knn_W.npz').tocsr().astype(np.float32)
        self.V = np.load(d / 'svd_V.npy').astype(np.float32)
        self.scale = np.load(d / 'svd_scale.npy').astype(np.float32)

        # --- historia de interacciones (train + val) y producto "oculto" (test) ---
        it = pd.read_parquet(d / 'interactions_light.parquet',
                             columns=['user_id', 'parent_asin', 'rating', 'event_ts', 'split'])
        it['split'] = it['split'].astype(str)
        it['u'] = self.users.get_indexer(it.user_id)
        it['i'] = self.item_ids.get_indexer(it.parent_asin)
        it = it[(it.u >= 0) & (it.i >= 0)]
        self.heldout = it[it.split == 'test'].drop_duplicates('u').set_index('u').i.to_dict()
        trv = it[it.split != 'test'].sort_values(['u', 'event_ts']).reset_index(drop=True)
        self.n_inter = len(trv)
        self.h_i, self.h_r = trv.i.to_numpy(), trv.rating.to_numpy(dtype=np.float32)
        self.starts = np.searchsorted(trv.u.to_numpy(), np.arange(self.n_users + 1))
        self.R = sp.csr_matrix((np.ones(len(trv), dtype=np.float32), (trv.u.values, trv.i.values)),
                               shape=(self.n_users, self.n_items))
        self.pop = np.asarray(self.R.sum(0)).ravel().astype(np.float32)
        self.pop_order = np.argsort(-self.pop)

        # perfil de contenido por usuario (rating x recencia), igual que el notebook
        age = (trv.event_ts.max() - trv.event_ts).dt.days.to_numpy(dtype=np.float32)
        w = _rating_w(self.h_r) * (0.5 ** (age / self.half_life))
        Wui = sp.csr_matrix((w, (trv.u.values, trv.i.values)), shape=(self.n_users, self.n_items))
        self.P = _l2_rows(Wui @ self.F)

        # segmentos
        seg = pd.read_parquet(d / 'user_segments.parquet', columns=['user_id', 'segment'])
        self.seg_u = pd.Series(self.users).map(seg.set_index('user_id').segment).fillna(-1).astype(int).values
        codes, uniq = pd.factorize(self.seg_u, sort=True)
        self.seg_code = codes
        Z = sp.csr_matrix((np.ones(self.n_users), (np.arange(self.n_users), codes)), shape=(self.n_users, len(uniq)))
        M = (Z.T @ self.R).toarray().astype(np.float32)
        M = M / np.maximum(M.max(1, keepdims=True), 1)
        self.M = M + 1e-3 * (self.pop / max(float(self.pop.max()), 1.0))
        self.seg_names = {-1: 'Atípicos (ráfagas)'}
        f = d / 'segment_names.json'
        if f.exists():
            self.seg_names.update({int(k): v for k, v in json.loads(f.read_text()).items()})
        self.metrics = None
        f = d / 'resultados_modelos.csv'
        if f.exists():
            self.metrics = pd.read_csv(f).round(4).to_dict('records')

    # ------------------------------------------------------------------ utilidades
    def seg_name(self, u):
        s = int(self.seg_u[u])
        return self.seg_names.get(s, f'Segmento {s}')

    def _hist(self, u):
        a, b = self.starts[u], self.starts[u + 1]
        return self.h_i[a:b], self.h_r[a:b]

    def _item(self, i):
        p = self.price[i]
        return {'i': int(i), 'title': self.title[i], 'brand': self.brand[i], 'platform': self.platform[i],
                'price': None if np.isnan(p) else round(float(p), 2)}

    def info(self):
        bs = pd.Series(self.pop).groupby(self.brand).sum().drop('desconocida', errors='ignore').sort_values(ascending=False)
        ps = pd.Series(self.pop).groupby(self.platform).sum().drop('desconocida', errors='ignore').sort_values(ascending=False)
        pr = self.price[~np.isnan(self.price)]
        return {'n_users': self.n_users, 'n_items': self.n_items, 'n_interactions': self.n_inter,
                'weights': self.weights, 'event_weights': self.event_w, 'half_life_days': self.half_life,
                'brands': bs.index[:14].tolist(), 'platforms': ps.index[:10].tolist(),
                'max_price': float(np.ceil(np.percentile(pr, 95) / 10) * 10) if len(pr) else 100.0,
                'metrics': self.metrics}

    def _user_card(self, u):
        hi, _ = self._hist(u)
        pl = pd.Series(self.platform[hi]); pl = pl[pl != 'desconocida']
        return {'idx': int(u), 'id': str(self.users[u]), 'n_hist': int(len(hi)), 'segment': int(self.seg_u[u]),
                'segment_name': self.seg_name(u), 'top_platform': pl.value_counts().index[0] if len(pl) else '—'}

    def sample_users(self, n=10, seed=None, q=None):
        if q:
            pick = np.where(self.users.str.contains(q, case=False, regex=False))[0][:n]
        else:
            cand = np.where(np.diff(self.R.indptr) >= 5)[0]
            rng = np.random.RandomState(seed)
            pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
        return [self._user_card(int(u)) for u in pick]

    def search(self, q, n=10):
        q = (q or '').lower().strip()
        if len(q) < 2:
            return []
        idx = np.where(np.char.find(self.title_l.astype(str), q) >= 0)[0]
        idx = idx[np.argsort(-self.pop[idx])][:n]
        return [self._item(i) for i in idx]

    def popular(self, platform=None, n=12):
        idx = self.pop_order if not platform else self.pop_order[self.platform[self.pop_order] == platform]
        return [self._item(i) for i in idx[:n]]

    # ------------------------------------------------------------------ perfil declarado (usuario nuevo)
    def _declared_profile(self, prefs):
        plats = [x for x in (prefs.get('platforms') or []) if x]
        brands = [str(x).lower() for x in (prefs.get('brands') or []) if x]
        kws = [w for w in re.split(r'[,\s]+', str(prefs.get('keywords') or '').lower()) if len(w) >= 3]
        mp = prefs.get('max_price')
        cons = {}
        if kws: cons['palabras clave'] = np.array([all(k in t for k in kws) for t in self.title_l])
        if brands: cons['marcas'] = np.isin(self.brand, brands)
        if mp: cons['presupuesto'] = ~(self.price > float(mp))
        if plats: cons['plataformas'] = np.isin(self.platform, plats)
        active, relaxed = list(cons), []
        def current():
            m = np.ones(self.n_items, bool)
            for k in active: m &= cons[k]
            return m
        m = current()
        while m.sum() < 5 and active:
            relaxed.append(active.pop(0)); m = current()
        info = {'constraints': len(cons), 'matched': int(m.sum()) if cons else 0, 'relaxed': relaxed}
        if not cons or m.sum() == 0:
            return np.zeros(self.F.shape[1], np.float32), np.array([], dtype=int), info
        seeds = self.pop_order[m[self.pop_order]][:40]
        wts = np.log1p(self.pop[seeds]) + 1e-3
        p = np.asarray(self.F[seeds].T @ wts).ravel().astype(np.float32)
        return p / (np.linalg.norm(p) + 1e-9), seeds, info

    # ------------------------------------------------------------------ perfil visible
    def _profile(self, liked, hist_for_price):
        def share(arr):
            s = pd.Series(arr); s = s[s != 'desconocida']
            vc = (s.value_counts() / max(int(s.value_counts().sum()), 1)).head(5)
            return [{'name': k, 'share': round(float(v), 3)} for k, v in vc.items()]
        tiers = pd.Series(self.tier[hist_for_price]).value_counts(normalize=True) if len(hist_for_price) else pd.Series(dtype=float)
        return {'brands': share(self.brand[liked]) if len(liked) else [],
                'platforms': share(self.platform[liked]) if len(liked) else [],
                'tiers': {t: round(float(tiers.get(t, 0)), 3) for t in TIERS}}

    def _reason(self, i, dom, brands, platforms, mode):
        b, pl = self.brand[i], self.platform[i]
        if dom == 'content':
            if b in brands: return f'Se parece a lo que te gusta · marca {b}'
            if pl in platforms: return f'Se parece a lo que te gusta · plataforma {pl}'
            return 'Se parece a lo que ya valoraste (marca, plataforma, precio y título)'
        if dom in ('knn', 'svd'):
            return 'Usuarios que eligieron lo mismo que tú también compraron esto'
        if dom == 'segpop':
            return 'Muy popular entre usuarios de tu segmento'
        return 'Muy popular en el catálogo' if mode == 'existing' else 'Popular entre perfiles parecidos al tuyo'

    # ------------------------------------------------------------------ recomendación
    def recommend(self, req):
        mode = req.get('mode', 'existing'); k = int(req.get('k', 10))
        filt = req.get('filters') or {}
        events = []
        for e in req.get('events') or []:
            try:
                i = int(e['i'])
            except (KeyError, TypeError, ValueError):
                continue
            if e.get('kind') in self.event_w and 0 <= i < self.n_items:
                events.append((i, e['kind']))

        h = np.zeros(self.n_items, np.float32); seen, held = set(), None
        extra = {}
        if mode == 'existing':
            u = int(req['user'])
            if not (0 <= u < self.n_users): raise ValueError('usuario inválido')
            hi, hr = self._hist(u)
            h[hi] = 1.0; seen.update(hi.tolist())
            p = self.P[u].toarray().ravel().astype(np.float32)
            seg_scores = self.M[self.seg_code[u]]
            w = dict(self.weights)
            liked = hi[hr >= 4] if (hr >= 4).any() else hi
            liked = list(liked); price_idx = list(hi)
            held = self.heldout.get(u)
            who = {'mode': 'existing', 'user': self._user_card(u)}
            avg_rating = round(float(hr.mean()), 2) if len(hr) else None
        else:
            prefs = req.get('prefs') or {}
            p, seeds, dinfo = self._declared_profile(prefs)
            seg_scores = None
            w = dict(self.weights); w['segpop'] = 0.0; w['content'] = max(w['content'], 1.0); w['pop'] = max(w['pop'], 0.3)
            liked = list(seeds[:10]); price_idx = list(seeds[:10])
            who = {'mode': 'new', 'declared': {'platforms': prefs.get('platforms') or [], 'brands': prefs.get('brands') or [],
                   'keywords': prefs.get('keywords') or '', 'max_price': prefs.get('max_price')}, 'seed_info': dinfo}
            avg_rating = None

        for i, kind in events:                                  # actualización dinámica del perfil
            wv = float(self.event_w[kind])
            p = 0.9 * p + wv * self.F[i].toarray().ravel()
            if wv > 0: h[i] = max(h[i], wv); liked.append(i)
            seen.add(i); price_idx.append(i)

        H = sp.csr_matrix(h.reshape(1, -1))
        parts = {
            'content': np.asarray(self.F @ (p / (np.linalg.norm(p) + 1e-9))).ravel(),
            'knn': np.asarray((H @ self.W).toarray()).ravel(),
            'svd': np.asarray((((H @ sp.diags(self.scale)) @ self.V) @ self.V.T)).ravel(),
            'segpop': np.asarray(seg_scores).ravel() if seg_scores is not None else np.zeros(self.n_items, np.float32),
            'pop': self.pop,
        }
        Z = {kk: _z(v) for kk, v in parts.items()}
        total = sum(w[kk] * Z[kk] for kk in PARTS if w[kk] > 0)
        if not isinstance(total, np.ndarray): total = np.zeros(self.n_items, np.float32)

        mask = np.zeros(self.n_items, bool); mask[list(seen)] = True
        base_rank_scores = np.where(mask, -np.inf, total)
        mp = filt.get('max_price')
        if mp: mask |= (self.price > float(mp))
        if filt.get('platform'): mask |= (self.platform != filt['platform'])
        tm = np.where(mask, -np.inf, total)
        ok = int(np.isfinite(tm).sum()); kk_ = min(k, ok)
        top = np.array([], dtype=int)
        if kk_ > 0:
            top = np.argpartition(-tm, kk_ - 1)[:kk_]; top = top[np.argsort(-tm[top])]

        prof = self._profile(np.array(liked, dtype=int), np.array(price_idx, dtype=int))
        b_set = {x['name'] for x in prof['brands'][:3]}; p_set = {x['name'] for x in prof['platforms'][:3]}
        recs = []
        for r, i in enumerate(top, 1):
            contrib = {kk: round(float(w[kk] * Z[kk][i]), 3) for kk in PARTS}
            dom = max(contrib, key=contrib.get)
            recs.append({**self._item(i), 'rank': r, 'score': round(float(total[i]), 3), 'parts': contrib,
                         'reason': self._reason(i, dom, b_set, p_set, mode)})

        out = {**who, 'profile': prof, 'avg_rating': avg_rating, 'recs': recs, 'weights': w,
               'n_events': len(events), 'filtered_out': int(self.n_items - ok - len(seen))}
        if mode == 'existing':
            hi, hr = self._hist(int(req['user']))
            last = list(range(len(hi)))[-6:][::-1]
            out['history'] = [{**self._item(hi[j]), 'rating': float(hr[j])} for j in last]
            if held is not None and held not in seen:
                hv = base_rank_scores[held]
                rank = int((base_rank_scores >= hv).sum())
                out['heldout'] = {**self._item(held), 'rank': rank, 'n_items': self.n_items, 'in_topk': bool(rank <= k)}
        return out
