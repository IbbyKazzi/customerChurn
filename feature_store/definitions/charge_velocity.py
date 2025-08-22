# charge_velocity.py
def charge_velocity(df):
    df['charge_velocity'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-5)
    return df[['customerID', 'charge_velocity']]
