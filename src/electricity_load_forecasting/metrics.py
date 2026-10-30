from sklearn.metrics import d2_pinball_score, mean_absolute_percentage_error


def split_by_quantile(pred):
    quantile_cols = {}
    for c in pred.columns:
        quantile_cols.setdefault(c.split("__")[1], []).append(c)
    return {
        q: pred.select(cols).rename(lambda c: c.split("__")[0])
        for q, cols in quantile_cols.items()
    }


def neg_mape(y_true, y_pred, quantile_regression=False):
    if quantile_regression:
        quantile_predictions = split_by_quantile(y_pred)
        scores = {}
        for q, q_pred in quantile_predictions.items():
            q_neg_mape = neg_mape(y_true, q_pred, quantile_regression=False)
            scores.update({f"{k}__{q}": v for k, v in q_neg_mape.items()})
            if q == "q_0.5":
                # Pick the median if available for comparison with non-quantile
                # models
                scores.update(q_neg_mape)
        return scores
    average = mean_absolute_percentage_error(y_true, y_pred)
    detail = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    return {"neg_mape__average": -average} | {
        f"neg_mape__{c}": -float(s) for c, s in zip(y_true.columns, detail)
    }


def neg_mape_scorer(estimator, X, y, quantile_regression=False):
    return neg_mape(y, estimator.predict(X), quantile_regression=quantile_regression)


def pinball(y_true, y_pred):
    quantile_predictions = split_by_quantile(y_pred)
    scores = {}
    for q, q_pred in quantile_predictions.items():
        alpha = float(q.removeprefix("q_"))
        scores[f"d2_pinball_score__average__{q}"] = d2_pinball_score(
            y_true, q_pred, alpha=alpha
        )
        detail = d2_pinball_score(
            y_true, q_pred, alpha=alpha, multioutput="raw_values"
        )
        scores.update(
            {
                f"d2_pinball_score__{c}__{q}": float(s)
                for c, s in zip(y_true.columns, detail)
            }
        )
    return scores


def pinball_scorer(estimator, X, y):
    return pinball(y, estimator.predict(X))


def coverage(y_true, y_pred, q_low=0.05, q_high=0.95):
    result = {}
    for horizon in y_true.columns:
        low_col = f"{horizon}__q_{q_low}"
        high_col = f"{horizon}__q_{q_high}"
        actual = y_true[horizon]
        inside = (actual >= y_pred[low_col]) & (actual <= y_pred[high_col])
        result[f"coverage_{q_low}_{q_high}__{horizon}"] = inside.mean()
    return result


def coverage_scorer(estimator, X, y, q_low=0.05, q_high=0.95):
    return coverage(y, estimator.predict(X), q_low=q_low, q_high=q_high)


def mean_width(y_true, y_pred, q_low=0.05, q_high=0.95):
    result = {}
    for horizon in y_true.columns:
        low = y_pred[f"{horizon}__q_{q_low}"]
        high = y_pred[f"{horizon}__q_{q_high}"]
        result[f"mean_width_{q_low}_{q_high}__{horizon}"] = (high - low).abs().mean()
    return result


def mean_width_scorer(estimator, X, y, q_low=0.05, q_high=0.95):
    return mean_width(y, estimator.predict(X), q_low=q_low, q_high=q_high)
