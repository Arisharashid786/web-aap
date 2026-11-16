"""
Train ThoraxCare's educational model using the REAL UCI Thoracic Surgery dataset.
Run: pip install -r requirements.txt && python train_model.py
"""
from ucimlrepo import fetch_ucirepo
import pandas as pd, pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer

# UCI dataset ID 277; original 470 patient records, 16 predictors and Risk1Y target.
ds=fetch_ucirepo(id=277)
X=ds.data.features.copy()
y=ds.data.targets.iloc[:,0].copy()
# Normalize binary/nominal values and target labels, without inventing rows.
X=X.astype(str).replace({"?":None,"":None})
for col in ["PRE4","PRE5","AGE"]:
    X[col]=pd.to_numeric(X[col],errors="coerce")
y=y.astype(str).str.strip()
# UCI labels are T (death within one year) / F (survival beyond one year)
y=y.replace({"True":"T","False":"F","1":"T","0":"F"})
categorical=[c for c in X.columns if c not in ["PRE4","PRE5","AGE"]]
numeric=["PRE4","PRE5","AGE"]
pre=ColumnTransformer([
 ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
 ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
])
model=Pipeline([("prep",pre),("clf",RandomForestClassifier(n_estimators=500,class_weight="balanced",random_state=42,min_samples_leaf=3))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
model.fit(Xtr,ytr)
pred=model.predict(Xte)
print("Balanced accuracy:",balanced_accuracy_score(yte,pred))
print(classification_report(yte,pred))
with open("thoracic_model.pkl","wb") as f: pickle.dump(model,f)
X.assign(Risk1Y=y).to_csv("thoracic_uci_data.csv",index=False)
print("Saved thoracic_model.pkl and thoracic_uci_data.csv")
