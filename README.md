# Penguin Clustering Using K-Means and DBSCAN

## 📌 Project Overview

Penguin Clustering is an Unsupervised Machine Learning project that explores patterns in penguin measurements and groups similar penguins using clustering algorithms.

The project uses the `penguins.csv` dataset and applies two clustering algorithms: **K-Means** and **DBSCAN**. It includes data cleaning, exploratory data analysis (EDA), feature encoding, cluster visualization, the Elbow Method, silhouette score evaluation, and cluster summary analysis.

The goal is to understand how unsupervised learning can discover groups in data without using predefined target labels.

## 🎯 Project Objectives

- Load and explore penguin measurement data.
- Handle missing and invalid values.
- Perform exploratory data analysis.
- Visualize distributions of penguin measurements.
- Encode categorical data for machine learning.
- Apply K-Means clustering.
- Use the Elbow Method to explore the number of clusters.
- Apply DBSCAN for density-based clustering.
- Evaluate clustering results using silhouette scores.
- Visualize cluster assignments using scatter plots.
- Compare the characteristics of the resulting clusters.

## 🛠️ Technologies and Libraries

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, clustering, and evaluation |
| StandardScaler | Feature standardization |
| LabelEncoder | Encoding categorical labels |
| K-Means | Centroid-based clustering |
| DBSCAN | Density-based clustering |
| Silhouette Score | Clustering evaluation |
| Jupyter Notebook | Interactive development |

## 📂 Dataset Description

The project loads a dataset named `penguins.csv`.

The notebook uses the following columns:

| Column | Description |
|---|---|
| `culmen_length_mm` | Penguin culmen (bill) length in millimeters |
| `culmen_depth_mm` | Penguin culmen (bill) depth in millimeters |
| `flipper_length_mm` | Flipper length in millimeters |
| `body_mass_g` | Body mass in grams |
| `sex` | Penguin sex, encoded numerically before clustering |

The notebook does not explicitly use a species label as the target for clustering. Instead, it explores patterns using the measurement features and encoded sex column.

**Note:** The CSV file must be available in the notebook's working directory. The dataset file itself was not included in the uploaded notebook.

## 🔄 Project Workflow

### 1. Importing Libraries

The notebook imports Pandas, NumPy, Matplotlib, Seaborn, and the required Scikit-learn classes for preprocessing, clustering, and evaluation.

### 2. Loading the Dataset

```python
import pandas as pd

df = pd.read_csv("penguins.csv")

df.head()
```

The dataset is loaded into a Pandas DataFrame for inspection and analysis.

### 3. Data Inspection and Cleaning

The notebook checks the DataFrame structure and missing values.

```python
df.info()
df.isna().sum()
```

Rows containing missing values are removed:

```python
df.dropna(inplace=True)
```

The notebook also checks for duplicate rows and filters out records where the `sex` value is `"."`.

```python
df = df[df["sex"] != "."]
```

These steps help prepare the dataset for further analysis.

### 4. Exploratory Data Analysis (EDA)

The notebook explores the dataset through several visualizations:

- Bar chart showing the distribution of penguin sex.
- Histograms showing culmen depth, flipper length, and body mass distributions.
- Box plot comparing body mass by sex.
- Descriptive statistics for numerical features.

These visualizations help reveal differences in the measurements and identify patterns within the dataset.

### 5. Encoding Categorical Data

The `sex` column contains categorical values. The notebook uses `LabelEncoder` to convert these categories into numerical labels.

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["sex"] = encoder.fit_transform(df["sex"])
```

This allows the encoded column to be used as a numerical input in the clustering experiments.

### 6. Feature Scaling

The notebook creates a standardized version of the DataFrame using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(df)
```

Standardization transforms numerical features to have approximately zero mean and unit variance.

This is particularly relevant for distance-based algorithms such as K-Means and DBSCAN.

**Implementation note:** Although the notebook creates the `scaled` array, the later K-Means and DBSCAN cells use the original DataFrame. Using a consistent, appropriately scaled feature matrix would improve the clustering workflow.

### 7. K-Means Clustering

K-Means groups data points into a specified number of clusters by assigning points to nearby cluster centroids.

The notebook uses the Elbow Method to explore cluster counts from 1 to 10.

```python
from sklearn.cluster import KMeans

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)
```

The inertia values are plotted to create an Elbow Curve. This helps explore a suitable number of clusters.

The notebook then fits K-Means with two clusters.

```python
kmeans = KMeans(
    n_clusters=2,
    random_state=400
)

df["KMeans_clusters"] = kmeans.fit_predict(df)
```

The resulting cluster labels are stored in the `KMeans_clusters` column.

### 8. Evaluating K-Means

The notebook calculates a silhouette score to evaluate the separation of the clusters.

```python
from sklearn.metrics import silhouette_score

KMeans_score = silhouette_score(
    df,
    kmeans.labels_
)

print("K-Means silhouette score:", KMeans_score)
```

The silhouette score ranges from -1 to 1:

- Values close to 1 generally indicate well-separated clusters.
- Values around 0 indicate overlapping clusters.
- Negative values can indicate that observations may have been assigned to inappropriate clusters.

**Important:** In the current notebook, the score calculation uses a DataFrame that already contains the `KMeans_clusters` column. This introduces the cluster labels into the feature data used for evaluation and can produce a misleading score. The score should be calculated using the original feature matrix without the cluster-label column.

### 9. Visualizing K-Means Clusters

The notebook creates scatter plots to visualize cluster assignments using different combinations of penguin measurements:

- Culmen length versus body mass.
- Culmen depth versus flipper length.
- Flipper length versus body mass.

The points are colored by `KMeans_clusters`, making it easier to compare the groups.

### 10. Cluster Summary

The notebook groups observations by their K-Means cluster and calculates the average measurements and number of observations in each cluster.

The summary includes:

- Average culmen length.
- Average culmen depth.
- Average encoded sex value.
- Average flipper length.
- Average body mass.
- Number of observations.

This helps describe the characteristics of the clusters produced by the model.

### 11. DBSCAN Clustering

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified in advance. It groups observations in dense regions and identifies isolated observations as noise.

The notebook uses the following configuration:

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=12,
    min_samples=5
)

df["DBSCAN_Cluster"] = dbscan.fit_predict(
    df.iloc[:, :-1]
)
```

Here:

- `eps=12` sets the neighborhood radius.
- `min_samples=5` sets the minimum number of points needed to form a dense region.
- `-1` represents noise points that DBSCAN does not assign to a regular cluster.

The chosen `eps` value should be reviewed after feature scaling because DBSCAN is sensitive to feature distances and scale.

### 12. Evaluating DBSCAN

The notebook separates noise points from the other cluster assignments and attempts to calculate a silhouette score when more than one non-noise cluster exists.

```python
dbscan_labels = df["DBSCAN_Cluster"]

valid_mask = dbscan_labels != -1
valid_features = df[valid_mask]
valid_labels = dbscan_labels[valid_mask]
```

The silhouette score is calculated only when more than one non-noise cluster is present.

For a valid evaluation, the features should exclude `DBSCAN_Cluster` and any other cluster-label columns. The current notebook's `valid_features` DataFrame contains the `DBSCAN_Cluster` column, which should be removed before scoring.

### 13. Visualizing DBSCAN Clusters

The notebook creates a scatter plot comparing culmen depth and body mass, with points colored by their DBSCAN cluster assignments.

This visualization helps identify dense groups and observations classified as noise.

### 14. DBSCAN Cluster Summary

The notebook calculates average penguin measurements and the number of observations for each DBSCAN cluster.

This includes the cluster labeled `-1`, which represents noise points.

## ✨ Key Features

- Penguin dataset inspection and cleaning.
- Missing-value and invalid-value handling.
- Exploratory data analysis.
- Histograms, bar charts, and box plots.
- Categorical encoding using LabelEncoder.
- Feature standardization using StandardScaler.
- K-Means clustering.
- Elbow Method for exploring cluster counts.
- Silhouette score evaluation.
- DBSCAN density-based clustering.
- Two-dimensional cluster visualization.
- Cluster-level statistical summaries.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version.
- Jupyter Notebook or JupyterLab.
- The `penguins.csv` dataset.

### 1. Clone the Repository

Replace the example URL with your actual GitHub repository URL.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install the Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Add the Dataset

Place `penguins.csv` in the project directory, or update the path in the notebook.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Penguine Clustering.ipynb` and execute the cells in order.

## 📁 Suggested Repository Structure

```text
penguin-clustering/
│
├── Penguine Clustering.ipynb
├── penguins.csv
├── images/
│   ├── penguin-distributions.png
│   ├── elbow-curve.png
│   ├── kmeans-clusters.png
│   └── dbscan-clusters.png
│
├── requirements.txt
└── README.md
```

This is a suggested structure, not a verified listing of your current repository files. Check the dataset's source, licensing, and file size before uploading it to GitHub.

## 💡 Applications

Clustering techniques can help with:

- Exploring patterns in biological measurements.
- Identifying groups with similar physical characteristics.
- Discovering structure in unlabeled datasets.
- Detecting unusual observations.
- Comparing centroid-based and density-based clustering.
- Learning unsupervised machine learning techniques.

These are potential applications of clustering; the notebook is an exploratory analysis rather than a deployed penguin classification application.

## 🔮 Future Improvements

- Use a consistent, scaled feature matrix for clustering.
- Exclude cluster-label columns from silhouette score calculations.
- Compare several K-Means cluster counts using silhouette scores.
- Tune DBSCAN's `eps` and `min_samples` parameters.
- Add a nearest-neighbor distance plot to help choose `eps`.
- Compare cluster characteristics using summary tables and visualizations.
- Evaluate clustering against species labels separately, if the dataset contains them and such an evaluation is appropriate.
- Improve cluster interpretation using biological context.
- Add screenshots of the Elbow Curve and clustering results.
- Save the fitted preprocessing objects and model settings for reproducibility.

## 👨‍💻 Author

**Anandhu P S**

- GitHub: https://github.com/Anandhu114

## 📚 Key Concepts Demonstrated

- Unsupervised Machine Learning
- K-Means Clustering
- DBSCAN
- Exploratory Data Analysis
- Data Cleaning
- Feature Encoding
- Feature Scaling
- Elbow Method
- Silhouette Score
- Noise Detection
- Cluster Visualization

## 📄 License

No license is specified in the notebook. Add an appropriate license if you want others to reuse, modify, or distribute your project.

---

If you find this project useful, feel free to star the repository!
