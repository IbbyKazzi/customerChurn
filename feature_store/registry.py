import importlib

FEATURES = ['loyalty_band', 'charge_velocity']

def get_features(df, selected=FEATURES):
    result = df[['customer_id']].copy()
    for feat in selected:
        module = importlib.import_module(f'definitions.{feat}')
        func = getattr(module, feat)
        result = result.merge(func(df), on='customer_id')
    return result
