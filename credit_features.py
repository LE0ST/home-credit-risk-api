"""Stable Home Credit inference feature transformer."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RATIO_FEATURES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO",
    "EMPLOYED_TO_AGE_RATIO",
    "INCOME_PER_PERSON",
]
ALL_ENGINEERED_FEATURES = [
    "DAYS_EMPLOYED_365243", *RATIO_FEATURES, "EXT_SOURCE_MEAN", "EXT_SOURCE_STD",
]


class HomeCreditFeatures(BaseEstimator, TransformerMixin):
    """Add safe ratios and external-source aggregates with train-fitted clipping."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        generated = self._generate(X)
        self.clip_bounds_ = {}
        for name in RATIO_FEATURES:
            values = generated[name].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                lo, hi = values.quantile([0.005, 0.995]).tolist()
                self.clip_bounds_[name] = (float(lo), float(hi))
            else:
                self.clip_bounds_[name] = (float("nan"), float("nan"))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    @staticmethod
    def _divide(
        numerator: pd.Series, denominator: pd.Series, *, positive_denominator: bool = True
    ) -> pd.Series:
        valid_denominator = denominator > 0 if positive_denominator else denominator.ne(0)
        denominator = denominator.where(valid_denominator)
        result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def _generate(cls, X: pd.DataFrame) -> pd.DataFrame:
        # Prevent infinities in raw numeric inputs from reaching SimpleImputer.
        out = X.replace([np.inf, -np.inf], np.nan).copy()
        employed = out["DAYS_EMPLOYED"]
        out["DAYS_EMPLOYED_365243"] = employed.eq(365243).astype("int8")
        # 365243 is Home Credit's sentinel for unavailable employment duration.
        # Other nonnegative employment-day values are also invalid in this schema.
        out["DAYS_EMPLOYED"] = employed.mask(employed >= 0)
        out["CREDIT_INCOME_RATIO"] = cls._divide(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
        out["ANNUITY_INCOME_RATIO"] = cls._divide(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
        out["CREDIT_ANNUITY_RATIO"] = cls._divide(out["AMT_CREDIT"], out["AMT_ANNUITY"])
        out["EMPLOYED_TO_AGE_RATIO"] = cls._divide(
            out["DAYS_EMPLOYED"], out["DAYS_BIRTH"], positive_denominator=False
        )
        out["INCOME_PER_PERSON"] = cls._divide(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])
        ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        out["EXT_SOURCE_MEAN"] = ext.mean(axis=1, skipna=True)
        out["EXT_SOURCE_STD"] = ext.std(axis=1, skipna=True, ddof=0)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._generate(X)
        for name, (lo, hi) in self.clip_bounds_.items():
            out[name] = out[name].clip(lower=lo, upper=hi)
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        names.extend([
            "DAYS_EMPLOYED_365243", *RATIO_FEATURES,
            "EXT_SOURCE_MEAN", "EXT_SOURCE_STD",
        ])
        return np.asarray(names, dtype=object)
