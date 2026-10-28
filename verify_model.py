"""DiscreteNaiveBayes 数值校验。"""
import numpy as np
from sklearn.naive_bayes import CategoricalNB

from experiment import DiscreteNaiveBayes

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array(['A', 'A', 'A', 'B'])
q = np.array([[0, 0], [1, 1], [0, 3]])  # 含训练集未出现但合法的取值 3

m = DiscreteNaiveBayes([2, 4]).fit(X, y)
r = CategoricalNB(alpha=1, min_categories=[2, 4]).fit(X, y)

np.testing.assert_allclose(m.predict_proba(q), r.predict_proba(q), atol=1e-12)
np.testing.assert_array_equal(m.predict(q), r.predict(q))
np.testing.assert_allclose(m.predict_proba(q).sum(1), 1)
np.testing.assert_allclose(np.exp(m.log_prob_[1][0]), [3 / 7, 2 / 7, 1 / 7, 1 / 7])
np.testing.assert_allclose(
    np.exp(DiscreteNaiveBayes([2, 4], prior_alpha=1).fit(X, y).log_prior_), [4 / 6, 2 / 6])

print('PASS: 与 CategoricalNB 一致，平滑和先验正确。')
