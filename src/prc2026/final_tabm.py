"""TabM regressor on the final-phase 483-column matrix.

Piecewise-linear numeric embeddings, 8-d category embeddings, TabM k-member ensemble,
per-member MSE on (y - anchor) / 600 s with original labels, on the float32 matrix of
final_pipeline: numeric columns are asinh-scaled with fit-row median/p10-p90,
categorical columns carry training-vocabulary codes (NaN = missing or unseen -> 0).
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import rtdl_num_embeddings
import tabm
import torch
from torch import nn

TABM = dict(quantile_bins=16, numeric_embedding_dimension=4, category_dimension=8, ensemble_members=16, n_blocks=3,
            width=256, dropout=0.1, epochs=20, batch_size=4096, inference_batch_size=32768, learning_rate=0.002,
            weight_decay=0.0003, gradient_clip_norm=10.0, target_scale_seconds=600.0, quantile_sample_rows=400000)


class Encoder:
    def __init__(self, numeric, categorical, medians, scales, bins, cat_sizes):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.medians, self.scales, self.bins, self.cat_sizes = list(medians), list(scales), list(bins), list(cat_sizes)

    @classmethod
    def fit(cls, X, features, categorical, vocab, cfg, seed):
        numeric = [j for j, c in enumerate(features) if c not in categorical]
        cats = [j for j, c in enumerate(features) if c in categorical]
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(X), min(len(X), cfg['quantile_sample_rows']), replace=False)
        medians, scales, bins = [], [], []
        for j in numeric:
            v = X[sample, j].astype(np.float64); v = v[np.isfinite(v)]
            if len(v):
                lo, med, hi = np.quantile(v, [.1, .5, .9]); scale = max(float(hi - lo), 1e-6)
                knots = np.unique(np.quantile(np.arcsinh((v - med) / scale), np.linspace(0, 1, cfg['quantile_bins'] + 1))
                                  .astype(np.float32))
            else:
                med, scale, knots = 0.0, 1.0, np.array([])
            if len(knots) < 2:
                knots = np.array([-1.0, 1.0])
            medians.append(float(med)); scales.append(scale); bins.append([float(k) for k in knots.astype(np.float32)])
        sizes = [len(vocab.get(features[j], {})) + 1 for j in cats]
        return cls(numeric, cats, medians, scales, bins, sizes)

    def transform(self, X, chunk=200000):
        n = len(X)
        scalar = np.empty((n, len(self.numeric)), np.float32); missing = np.empty((n, len(self.numeric)), bool)
        cats = np.empty((n, len(self.categorical)), np.int64)
        med, scale = np.asarray(self.medians, np.float32), np.asarray(self.scales, np.float32)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            v = X[a:b][:, self.numeric]; ok = np.isfinite(v)
            missing[a:b] = ~ok
            scalar[a:b] = np.arcsinh((np.where(ok, v, med) - med) / scale)
            c = X[a:b][:, self.categorical]
            cats[a:b] = np.where(np.isfinite(c), c + 1, 0).astype(np.int64)
        cats = np.minimum(cats, np.asarray(self.cat_sizes, np.int64) - 1)
        return scalar, missing, cats

    def state(self):
        return dict(vars(self))


class Net(nn.Module):
    def __init__(self, enc, cfg):
        super().__init__()
        d, k = cfg['numeric_embedding_dimension'], cfg['ensemble_members']
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'The [0-9]+-th feature has just two bin edges.*', category=UserWarning)
            self.numeric = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                [torch.tensor(b, dtype=torch.float32) for b in enc.bins], d, activation=False, version='B')
        self.cats = nn.ModuleList([nn.Embedding(n, cfg['category_dimension'], padding_idx=0) for n in enc.cat_sizes])
        dims = [d + 2] * len(enc.numeric) + [cfg['category_dimension']] * len(enc.categorical)
        self.view = tabm.EnsembleView(k=k)
        self.backbone = tabm.make_tabm_backbone(d_in=sum(dims), n_blocks=cfg['n_blocks'], d_block=cfg['width'],
                                                dropout=cfg['dropout'], activation='ReLU', k=k, arch_type='tabm',
                                                start_scaling_init='normal', start_scaling_init_chunks=dims)
        self.head = tabm.LinearEnsemble(cfg['width'], 1, k=k)

    def forward(self, scalar, missing, cats):
        num = torch.cat([self.numeric(scalar), scalar[..., None], missing.to(scalar.dtype)[..., None]], dim=-1).flatten(1)
        x = torch.cat([num, *[layer(cats[:, i]) for i, layer in enumerate(self.cats)]], dim=-1)
        return self.head(self.backbone(self.view(x))).squeeze(-1)


class TabmRegressor:
    def __init__(self, enc, net, anchor, cfg):
        self.enc, self.net, self.anchor, self.cfg = enc, net, anchor, cfg

    def predict(self, X):
        device = next(self.net.parameters()).device
        scalar, missing, cats = self.enc.transform(X)
        out = np.empty(len(X)); bs = self.cfg['inference_batch_size']
        self.net.eval()
        with torch.inference_mode():
            for a in range(0, len(X), bs):
                b = min(len(X), a + bs)
                batch = [torch.as_tensor(t[a:b], device=device) for t in (scalar, missing, cats)]
                out[a:b] = self.net(*batch).mean(dim=1).double().cpu().numpy()
        values = self.anchor + self.cfg['target_scale_seconds'] * out
        if not np.isfinite(values).all():
            raise ValueError('Non-finite TabM prediction')
        return values

    def save(self, path):
        torch.save({'state': self.net.state_dict(), 'encoder': self.enc.state(), 'anchor': self.anchor, 'cfg': self.cfg}, path)

    @classmethod
    def load(cls, path, device='cuda'):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        enc = Encoder(**saved['encoder']); net = Net(enc, saved['cfg']); net.load_state_dict(saved['state'])
        return cls(enc, net.to(device), saved['anchor'], saved['cfg'])


def fit_tabm(X, y, features, categorical, vocab, seed, cfg=TABM, log=lambda m: print(m, flush=True)):
    torch.manual_seed(seed); device = torch.device('cuda')
    enc = Encoder.fit(X, features, categorical, vocab, cfg, seed)
    anchor = float(np.median(y))
    tensors = [torch.as_tensor(t, device=device) for t in enc.transform(X)]
    target = torch.as_tensor(((y - anchor) / cfg['target_scale_seconds']).astype(np.float32), device=device)
    net = Net(enc, cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    gen = torch.Generator(device=device)
    for epoch in range(cfg['epochs']):
        started = time.time(); net.train(); gen.manual_seed(seed + epoch)
        order = torch.randperm(len(target), generator=gen, device=device); total = 0.0
        for a in range(0, len(order), cfg['batch_size']):
            idx = order[a:a + cfg['batch_size']]
            opt.zero_grad(set_to_none=True)
            loss = (net(*[t[idx] for t in tensors]) - target[idx, None]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite TabM loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm']); opt.step()
            total += float(loss.detach()) * len(idx)
        log(json.dumps({'tabm_epoch': epoch + 1, 'train_rmse_sec': float(np.sqrt(total / len(order))) * cfg['target_scale_seconds'],
                        'seconds': round(time.time() - started, 1)}))
    del tensors, target
    torch.cuda.empty_cache()
    return TabmRegressor(enc, net, anchor, cfg)


class TabmClassifier(TabmRegressor):
    """Same backbone; members output logits, prediction is the mean member probability."""
    def predict(self, X):
        device = next(self.net.parameters()).device
        scalar, missing, cats = self.enc.transform(X)
        out = np.empty(len(X)); bs = self.cfg['inference_batch_size']
        self.net.eval()
        with torch.inference_mode():
            for a in range(0, len(X), bs):
                b = min(len(X), a + bs)
                batch = [torch.as_tensor(t[a:b], device=device) for t in (scalar, missing, cats)]
                out[a:b] = torch.sigmoid(self.net(*batch)).mean(dim=1).double().cpu().numpy()
        return out


def fit_tabm_classifier(X, s, weight, features, categorical, vocab, seed, cfg=TABM, log=lambda m: print(m, flush=True)):
    """Weighted per-member binary cross-entropy (weights as in the XGB classifier, mean 1)."""
    torch.manual_seed(seed); device = torch.device('cuda')
    enc = Encoder.fit(X, features, categorical, vocab, cfg, seed)
    tensors = [torch.as_tensor(t, device=device) for t in enc.transform(X)]
    target = torch.as_tensor(np.asarray(s, np.float32), device=device)
    w = torch.as_tensor(np.ones(len(s), np.float32) if weight is None else np.asarray(weight, np.float32), device=device)
    net = Net(enc, cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    gen = torch.Generator(device=device); bce = nn.BCEWithLogitsLoss(reduction='none')
    for epoch in range(cfg['epochs']):
        started = time.time(); net.train(); gen.manual_seed(seed + epoch)
        order = torch.randperm(len(target), generator=gen, device=device); total = 0.0
        for a in range(0, len(order), cfg['batch_size']):
            idx = order[a:a + cfg['batch_size']]
            opt.zero_grad(set_to_none=True)
            logits = net(*[t[idx] for t in tensors])
            loss = (bce(logits, target[idx, None].expand_as(logits)) * w[idx, None]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite TabM classifier loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm']); opt.step()
            total += float(loss.detach()) * len(idx)
        log(json.dumps({'tabm_class_epoch': epoch + 1, 'weighted_bce': total / len(order), 'seconds': round(time.time() - started, 1)}))
    del tensors, target, w
    torch.cuda.empty_cache()
    return TabmClassifier(enc, net, 0.0, cfg)
