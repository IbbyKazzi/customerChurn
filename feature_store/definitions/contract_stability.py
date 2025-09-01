#contract_stability.py
def contract_stability(df):
    def score(row):
        score = 0
        if row['Contract'] == 'Month-to-month':
            score += 2
        elif row['Contract'] == 'One year':
            score += 1
        # Two year = 0 (most stable)

        if row['PaymentMethod'] in ['Electronic check', 'Mailed check']:
            score += 1  # Less automated = more churn-prone

        if row['PaperlessBilling'] == 'Yes':
            score -= 1  # Slightly more stable

        return score

    df['contract_stability'] = df.apply(score, axis=1)
    return df[['customerID', 'contract_stability']]
