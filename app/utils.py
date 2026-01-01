def align_features(df, feature_columns):
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]
