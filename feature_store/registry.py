import importlib

FEATURES = ['loyalty_band', 'charge_velocity']

def get_features(df, selected=FEATURES):
    result = df[['customerID']].copy()
    for feat in selected:
        module = importlib.import_module(f'feature_store.definitions.{feat}')
        func = getattr(module, feat)
        result = result.merge(func(df), on='customerID')
    return result

def save_selected_features(name, features):
    import json
    with open(f"feature_store/{name}.json", "w") as f:
        json.dump(features, f)

def load_selected_features(name):
    import json
    with open(f"feature_store/{name}.json", "r") as f:
        return json.load(f)
