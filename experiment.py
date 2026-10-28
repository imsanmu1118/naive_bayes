import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
    'ShapeFactor3', 'ShapeFactor4',
]
K = 3          # 离散特征取值个数
SEED = 42


class DiscreteNaiveBayes(ClassifierMixin, BaseEstimator):
    """有限多值朴素贝叶斯，条件概率使用 alpha 平滑。"""

    def __init__(self, cardinalities, alpha=1.0, prior_alpha=0.0):
        self.cardinalities = cardinalities
        self.alpha = alpha
        self.prior_alpha = prior_alpha

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=int), np.asarray(y)
        self.classes_, enc = np.unique(y, return_inverse=True)
        counts = np.bincount(enc)
        self.log_prior_ = np.log(
            (counts + self.prior_alpha) / (len(y) + len(counts) * self.prior_alpha))
        self.log_prob_ = []
        for i, k in enumerate(self.cardinalities):
            table = np.zeros((len(counts), k))
            np.add.at(table, (enc, X[:, i]), 1)
            self.log_prob_.append(
                np.log((table + self.alpha) / (counts[:, None] + k * self.alpha)))
        return self

    def scores(self, X):
        X = np.asarray(X, dtype=int)
        s = np.tile(self.log_prior_, (len(X), 1))
        for i, log_prob in enumerate(self.log_prob_):
            s += log_prob[:, X[:, i]].T
        return s

    def predict(self, X):
        return self.classes_[self.scores(X).argmax(1)]

    def predict_proba(self, X):
        s = self.scores(X)
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(1, keepdims=True)


def wilson(correct, n, z=1.959963984540054):
    p = correct / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - r, c + r


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return n01, n10, binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0


def load(data_dir):
    names = ('bean_train_1000', 'bean_train', 'bean_test', 'bean_raw')
    small, train, test, raw = [pd.read_csv(data_dir / f'{n}.csv') for n in names]
    assert (len(small), len(train), len(test)) == (1000, 9526, 4085)
    assert set(small.id) <= set(train.id) and not set(train.id) & set(test.id)
    assert not train.isna().any().any() and not test.isna().any().any()
    assert np.isin(train[FEATURES], range(K)).all()
    assert np.isin(test[FEATURES], range(K)).all()
    assert set(raw.loc[raw.partition == 'TRAIN', 'id']) == set(train.id)
    assert set(raw.loc[raw.partition == 'TEST', 'id']) == set(test.id)
    return small, train, test, raw


def evaluate(name, y, pred, labels):
    correct = int((np.asarray(y) == np.asarray(pred)).sum())
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    lo, hi = wilson(correct, len(y))
    row = {'model': name, 'correct': correct, 'test_size': len(y),
           'accuracy': correct / len(y), 'macro_f1': report['macro avg']['f1-score'],
           'weighted_f1': report['weighted avg']['f1-score'],
           'ci95_low': lo, 'ci95_high': hi}
    per_class = [{'model': name, 'class': c, 'precision': report[c]['precision'],
                  'recall': report[c]['recall'], 'f1_score': report[c]['f1-score'],
                  'support': report[c]['support']} for c in labels]
    return row, per_class


def analyze_correlations(raw_train):
    corr = raw_train[FEATURES].corr()
    pairs = sorted(((abs(corr.loc[a, b]), a, b) for i, a in enumerate(FEATURES)
                    for b in FEATURES[i + 1:]), reverse=True)
    top = [{'feature_1': a, 'feature_2': b, 'pearson_r': float(corr.loc[a, b])}
           for _, a, b in pairs[:5]]
    within = []
    for p in top:
        sub = {c: raw_train[raw_train['Class'] == c] for c in sorted(raw_train['Class'].unique())}
        rs = [sub[c][p['feature_1']].corr(sub[c][p['feature_2']]) for c in sub]
        within.append({'feature_1': p['feature_1'], 'feature_2': p['feature_2'],
                       'overall_r': p['pearson_r'], 'within_mean': float(np.mean(rs)),
                       'within_min': float(np.min(rs)), 'within_max': float(np.max(rs))})
    reduced = []
    for f in FEATURES:
        if all(abs(corr.loc[f, g]) < 0.99 for g in reduced):
            reduced.append(f)
    return {'reduced_features': reduced,
            'dropped_features': [f for f in FEATURES if f not in reduced],
            'top_pairs': top, 'within_class': within,
            'equiv_diameter_max_abs_error': float(np.max(np.abs(
                raw_train['EquivDiameter'] - np.sqrt(4 * raw_train['Area'] / np.pi))))}


def redundancy(train, test, raw_train, raw_test, reduced):
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    out = {}
    for name, fs in [('full_16', FEATURES), ('reduced', reduced)]:
        nb = DiscreteNaiveBayes([K] * len(fs))
        nb_cv = cross_val_score(nb, train[fs], train['Class'], cv=cv).mean()
        nb_test = accuracy_score(test['Class'], nb.fit(train[fs], train['Class']).predict(test[fs]))
        gnb = Pipeline([('scale', StandardScaler()), ('nb', GaussianNB())])
        gnb_cv = cross_val_score(gnb, raw_train[fs], raw_train['Class'], cv=cv).mean()
        gnb_test = accuracy_score(
            raw_test['Class'], gnb.fit(raw_train[fs], raw_train['Class']).predict(raw_test[fs]))
        out[name] = {'n_features': len(fs), 'nb_cv': float(nb_cv), 'nb_test': float(nb_test),
                     'gaussian_cv': float(gnb_cv), 'gaussian_test': float(gnb_test)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    small, train, test, raw = load(args.data_dir)
    labels = sorted(train['Class'].unique())
    raw_train = raw[raw['partition'] == 'TRAIN'].copy()
    raw_test = raw.set_index('id').loc[test['id']].reset_index()
    preds = pd.DataFrame({'id': test['id'], 'true_class': test['Class']})
    rows, per_class = [], []

    # 离散 NB：1000 条与完整训练集
    for name, data in [('nb_1000', small), ('nb_full', train)]:
        nb = DiscreteNaiveBayes([K] * len(FEATURES)).fit(data[FEATURES], data['Class'])
        ref = CategoricalNB(alpha=1, min_categories=K).fit(data[FEATURES], data['Class'])
        np.testing.assert_allclose(nb.predict_proba(test[FEATURES]),
                                   ref.predict_proba(test[FEATURES]), rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(nb.predict(test[FEATURES]), ref.predict(test[FEATURES]))
        preds[name] = nb.predict(test[FEATURES])
        row, pc = evaluate(name, test['Class'], preds[name], labels)
        rows.append(row); per_class += pc

    # 连续特征：高斯 NB
    gnb = Pipeline([('scale', StandardScaler()), ('nb', GaussianNB())])
    preds['gaussian_raw'] = gnb.fit(raw_train[FEATURES], raw_train['Class']).predict(raw_test[FEATURES])
    row, pc = evaluate('gaussian_raw', test['Class'], preds['gaussian_raw'], labels)
    rows.append(row); per_class += pc

    # 改进模型：标准化 + 多层感知机
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    model = Pipeline([('scale', StandardScaler()), ('mlp', MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='tanh', solver='adam', alpha=1e-4,
        batch_size=64, learning_rate_init=1e-3, max_iter=500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=SEED))])
    cv_accuracy = cross_val_score(model, raw_train[FEATURES], raw_train['Class'], cv=cv).mean()
    started = time.perf_counter()
    model.fit(raw_train[FEATURES], raw_train['Class'])
    fit_seconds = time.perf_counter() - started
    preds['mlp_raw'] = model.predict(raw_test[FEATURES])
    row, pc = evaluate('mlp_raw', test['Class'], preds['mlp_raw'], labels)
    rows.append(row); per_class += pc

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / 'metrics.csv', index=False)
    pd.DataFrame(per_class).to_csv(args.output_dir / 'per_class_metrics.csv', index=False)
    preds.to_csv(args.output_dir / 'test_predictions.csv', index=False)

    names = ['nb_1000', 'nb_full', 'gaussian_raw', 'mlp_raw']
    correct = {n: preds[n].to_numpy() == preds['true_class'].to_numpy() for n in names}
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n01, n10, p = mcnemar(correct[a], correct[b])
            pairs.append({'model_a': a, 'model_b': b, 'a_only_correct': n01,
                          'b_only_correct': n10, 'mcnemar_exact_p': p})

    corr = analyze_correlations(raw_train)
    red = redundancy(train, test, raw_train, raw_test, corr['reduced_features'])
    summary = {
        'metrics': rows, 'paired_mcnemar': pairs,
        'mlp': {'hidden_layer_sizes': [128, 64], 'activation': 'tanh', 'alpha': 1e-4,
                'cv_accuracy': float(cv_accuracy), 'n_iter': int(model.named_steps['mlp'].n_iter_),
                'fit_seconds': fit_seconds},
        'correlation_analysis': corr, 'redundancy_analysis': red,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'sklearn': sklearn.__version__},
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(metrics.to_string(index=False))
    print(f'MLP CV accuracy: {cv_accuracy:.4f}')


if __name__ == '__main__':
    main()
