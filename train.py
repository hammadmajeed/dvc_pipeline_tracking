import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Load data
df = pd.read_csv("data_processed.csv")

# Prepare labels
y = df.pop("cons_general").to_numpy()
y[y < 4] = 0
y[y >= 4] = 1

# Prepare features
X = preprocessing.scale(df.to_numpy())
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)

# Train and evaluate SVM
clf = SVC()
yhat = cross_val_predict(clf, X, y, cv=5)

acc = np.mean(yhat == y)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# ✅ Save DVC metrics (for metrics diff)
with open("metrics.json", "w") as f:
    json.dump({
        "accuracy": float(acc),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity)
    }, f, indent=4)

# ✅ Save DVC plot data (for plots diff)
# For example: accuracy across folds or pseudo-epochs
plot_data = []
for i, val in enumerate(yhat):
    plot_data.append({"sample": int(i), "correct": int(yhat[i] == y[i])})

with open("accuracy.json", "w") as f:
    json.dump(plot_data, f, indent=4)

# ✅ Visualization by region (static PNG)
df["pred_accuracy"] = [int(a == b) for a, b in zip(yhat, y)]
sns.set_color_codes("dark")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette="Greens_d")
ax.set(xlabel="Region", ylabel="Model accuracy")
plt.savefig("by_region.png", dpi=80)
