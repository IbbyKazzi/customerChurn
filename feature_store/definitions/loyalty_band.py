# loyalty_band.py
def loyalty_band(df):
    def tenure_group(t):
        if t < 12: return 'New'
        elif t < 36: return 'Intermediate'
        else: return 'Loyal'
    df['loyalty_band'] = df['tenure'].apply(tenure_group)
    return df[['customer_id', 'loyalty_band']]
