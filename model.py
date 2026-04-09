import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)

EXCLUDE_COLS = {"y_forward_return", "return"}


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def build_model(model_type="random_forest"):
    estimators = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            max_features=0.5, random_state=42, n_jobs=-1,
        ),
        "ridge": Ridge(alpha=10.0),
        "lasso": Lasso(alpha=0.01, max_iter=10_000),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    }
    if model_type not in estimators:
        raise ValueError(f"Unknown model_type '{model_type}'")
    return Pipeline([("scaler", StandardScaler()), ("model", estimators[model_type])])


def train_and_predict(df, model_type="random_forest", train_frac=0.85):
    feat_cols = get_feature_cols(df)
    X = df[feat_cols]
    y = df["y_forward_return"]
    split = int(len(df) * train_frac)

    model = build_model(model_type)
    model.fit(X.iloc[:split], y.iloc[:split])

    preds_test = model.predict(X.iloc[split:])
    last_pred = model.predict(X.tail(1))[0]

    eval_df = pd.DataFrame({
        "actual": y.iloc[split:].values,
        "predicted": preds_test,
    }, index=y.iloc[split:].index)

    r2 = r2_score(y.iloc[split:], preds_test)
    ic = np.corrcoef(preds_test, y.iloc[split:].values)[0, 1]
    logger.info(f"  Hold-out R²={r2:.4f}  IC={ic:.4f}  last_pred={last_pred:.5f}")

    return model, last_pred, eval_df


def get_feature_importance(model, feat_cols):
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        imp = np.abs(estimator.coef_)
    else:
        return pd.Series(dtype=float)
    return pd.Series(imp, index=feat_cols).sort_values(ascending=False)


def train_all_models(feature_dict, model_type="random_forest"):
    results = {}
    for ticker, df in feature_dict.items():
        logger.info(f"Training model for {ticker}  ({len(df)} obs)")
        model, last_pred, eval_df = train_and_predict(df, model_type=model_type)
        feat_cols = get_feature_cols(df)
        results[ticker] = {
            "model": model,
            "last_pred": last_pred,
            "eval_df": eval_df,
            "feature_importance": get_feature_importance(model, feat_cols),
        }
    return results


def get_predicted_returns(model_results):
    return pd.Series({ticker: res["last_pred"] for ticker, res in model_results.items()})