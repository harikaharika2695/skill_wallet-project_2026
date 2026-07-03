from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 1. Load all 3 files needed
app = pd.read_csv('application_record.csv')  # has SK_ID_CURR
bureau = pd.read_csv('bureau.csv')           # has both SK_ID_CURR and SK_ID_BUREAU  
credit = pd.read_csv('credit_record.csv')    # has SK_ID_BUREAU, MONTHS_BALANCE, STATUS

print("1. App rows:", len(app))
print("2. Bureau rows:", len(bureau))  
print("3. Credit balance rows:", len(credit))

# 2. Map STATUS to binary target: 0=paid, C=closed, X=no loan are good
def create_binary_target(status):
    return 1 if str(status).strip().upper() in ['0', 'C', 'X'] else 0

credit['TARGET'] = credit['STATUS'].apply(create_binary_target)
print("4. STATUS distribution:", credit['STATUS'].value_counts().head())
print("5. TARGET after mapping:", credit['TARGET'].value_counts(normalize=True))

# 3. Aggregate to loan level first
credit_agg = credit.groupby('SK_ID_BUREAU')['TARGET'].min().reset_index()
credit_agg.columns = ['SK_ID_BUREAU', 'CREDIT_TARGET']

# 4. Join bureau_balance to bureau to get SK_ID_CURR
bureau_full = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].merge(credit_agg, on='SK_ID_BUREAU', how='left')

# 5. Aggregate to applicant level: if ANY loan was bad, applicant is bad
applicant_credit = bureau_full.groupby('SK_ID_CURR')['CREDIT_TARGET'].min().reset_index()

# 6. Merge with main application data
final_df = app.merge(applicant_credit, on='SK_ID_CURR', how='left')

# 7. Applicants with no bureau data = assume good
final_df['CREDIT_TARGET'] = final_df['CREDIT_TARGET'].fillna(1)

print("\n=== FINAL RESULT ===")
print("Final shape:", final_df.shape)
print("New target distribution:")
print(final_df['CREDIT_TARGET'].value_counts(normalize=True))

# --- Handling Categorical Values using LabelEncoder ---
# fit_transform() identifies unique category names in each column and converts
# them into integer values. Makes dataset suitable for ML algorithms.
# Encoded values do not represent ranking. Works with Decision Tree, Random Forest, XGBoost.

df = final_df.copy()

categorical_cols = df.select_dtypes(include=['object']).columns
print("7. Categorical columns to encode:", list(categorical_cols))

for col in categorical_cols:
    df[col] = df[col].fillna('Unknown') # LabelEncoder crashes on NaN
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"8. Encoded {col}: {len(le.classes_)} categories")

print("9. Remaining object columns:", df.select_dtypes(include=['object']).columns.tolist())

# Save the ML-ready dataset
df.to_csv('final_dataset.csv', index=False)
print("10. final_dataset.csv saved - ready for tree-based models")
