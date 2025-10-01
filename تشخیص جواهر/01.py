"""
کد ساده آموزش مدل برای پیش‌بینی ستون "shape" از دیتاست الماس‌ها
نام فایل: "diamonds (cleaned).csv"

در این نسخه از CatBoost استفاده شده و کد ساده‌تر شده است.
"""

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------- بارگذاری داده --------------------
df = pd.read_csv(r"C:\Users\Ali\PycharmProjects\PythonProject43\diamonds (cleaned).csv")

# انتخاب ویژگی‌ها و هدف
X = df.drop("Shape", axis=1)
y = df["Shape"]

# تبدیل برچسب هدف به عددی
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# جدا کردن داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# ستون‌های دسته‌ای (object یا category)
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols]

# -------------------- مدیریت مقادیر گمشده --------------------
for c in X_train.select_dtypes(include=["number"]).columns:
    X_train[c] = X_train[c].fillna(X_train[c].median())
    X_test[c] = X_test[c].fillna(X_train[c].median())

for c in cat_cols:
    X_train[c] = X_train[c].astype(str).fillna("__MISSING__")
    X_test[c] = X_test[c].astype(str).fillna("__MISSING__")

# تعریف مدل CatBoost
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    verbose=100,
    random_seed=42
)

# آموزش مدل
train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_features_idx)
model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# پیش‌بینی روی داده تست
y_pred = model.predict(X_test)

# محاسبه دقت
acc = accuracy_score(y_test, y_pred)
print(f"دقت مدل روی داده تست: {acc:.4f}")

# -------------------- ذخیره مدل --------------------
model.save_model("catboost_shape_model.cbm")
