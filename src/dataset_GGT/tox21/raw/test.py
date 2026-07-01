import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

labels = [
    "NR-AR","NR-AR-LBD","NR-ER","NR-ER-LBD",
    "NR-AhR","NR-Aromatase","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

df = pd.read_csv("tox21.csv")[labels]

corr = df.corr(method="pearson")

plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True
)
plt.title("Tox21 Label Correlation Matrix")
plt.tight_layout()
plt.show()