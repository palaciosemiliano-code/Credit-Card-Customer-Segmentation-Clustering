"""Clustering de clientes de tarjetas de crédito con KMeans, Agglomerative y DBSCAN."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(r"C:\Credit Card Clustering\CC GENERAL.csv")
OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def valid_silhouette(features, labels):
    """Calcula silhouette solo cuando hay al menos dos clusters válidos."""
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(unique_labels) == len(features):
        return None
    return silhouette_score(features, labels)


def main():
    # 1. Cargar y mostrar el DataFrame, como en el proyecto previo.
    raw_df = pd.read_csv(DATA_PATH)
    print("Primeras filas del DataFrame original:")
    print(raw_df.head())
    print(f"\nDimensiones originales: {raw_df.shape}")

    # CUST_ID es un identificador único; se conserva para el resultado, pero no es una variable de clustering.
    feature_df = raw_df.drop(columns="CUST_ID").copy()

    # 2. Reemplazar nulos por la mediana de cada columna numérica.
    print("\nNulos antes de imputar:")
    print(feature_df.isna().sum()[feature_df.isna().sum() > 0])
    imputer = SimpleImputer(strategy="median")
    feature_df = pd.DataFrame(
        imputer.fit_transform(feature_df),
        columns=feature_df.columns,
        index=feature_df.index,
    )
    print(f"\nNulos después de imputar: {feature_df.isna().sum().sum()}")

    # 3. Escalar para que importes monetarios, frecuencias y conteos tengan el mismo peso.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    # 4. Elegir KMeans mediante silhouette para k entre 2 y 8.
    candidates = []
    for n_clusters in range(2, 9):
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(features_scaled)
        candidates.append(
            {
                "n_clusters": n_clusters,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(features_scaled, labels),
            }
        )
    kmeans_selection = pd.DataFrame(candidates)
    best_k = int(kmeans_selection.loc[kmeans_selection["silhouette"].idxmax(), "n_clusters"])
    print("\nSelección de k para KMeans:")
    print(kmeans_selection.to_string(index=False))
    print(f"\nK seleccionado por mayor silhouette: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    kmeans_labels = kmeans.fit_predict(features_scaled)

    # 5. Usar el mismo número de grupos para una comparación directa con clustering jerárquico.
    agglomerative = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agglomerative_labels = agglomerative.fit_predict(features_scaled)

    # 6. Elegir eps para DBSCAN a partir del percentil 90 de la distancia al 5.º vecino.
    min_samples = 5
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(features_scaled)
    distances, _ = neighbors.kneighbors(features_scaled)
    eps = float(pd.Series(distances[:, -1]).quantile(0.90))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(features_scaled)

    non_noise = dbscan_labels != -1
    dbscan_cluster_sizes = pd.Series(dbscan_labels[non_noise]).value_counts()
    min_comparable_cluster_size = max(20, int(non_noise.sum() * 0.01))
    dbscan_silhouette = (
        valid_silhouette(features_scaled[non_noise], dbscan_labels[non_noise])
        if len(dbscan_cluster_sizes) > 1
        and dbscan_cluster_sizes.min() >= min_comparable_cluster_size
        else None
    )

    metrics = pd.DataFrame(
        [
            {
                "method": "KMeans",
                "clusters": len(set(kmeans_labels)),
                "noise_points": 0,
                "silhouette": valid_silhouette(features_scaled, kmeans_labels),
            },
            {
                "method": "AgglomerativeClustering",
                "clusters": len(set(agglomerative_labels)),
                "noise_points": 0,
                "silhouette": valid_silhouette(features_scaled, agglomerative_labels),
            },
            {
                "method": "DBSCAN",
                "clusters": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                "noise_points": int((dbscan_labels == -1).sum()),
                "silhouette": dbscan_silhouette,
            },
        ]
    )
    print(f"\nDBSCAN: eps={eps:.3f}, min_samples={min_samples}")
    if dbscan_silhouette is None:
        print(
            "El silhouette de DBSCAN no es comparable: al menos un cluster tiene menos "
            f"de {min_comparable_cluster_size} clientes."
        )
    print("\nComparación de métodos:")
    print(metrics.to_string(index=False))

    # 7. Guardar etiquetas para analizar perfiles de clientes después.
    clustered_df = raw_df[["CUST_ID"]].join(feature_df)
    clustered_df["cluster_kmeans"] = kmeans_labels
    clustered_df["cluster_agglomerative"] = agglomerative_labels
    clustered_df["cluster_dbscan"] = dbscan_labels
    clustered_df.to_csv(OUTPUT_DIR / "credit_card_clusters.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "clustering_metrics.csv", index=False)

    # 8. Visualización en dos componentes principales; PCA solo es para graficar.
    pca_features = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(features_scaled)
    methods = [
        ("KMeans", kmeans_labels),
        ("Agglomerative", agglomerative_labels),
        ("DBSCAN (-1 = ruido)", dbscan_labels),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for axis, (name, labels) in zip(axes, methods):
        scatter = axis.scatter(
            pca_features[:, 0], pca_features[:, 1], c=labels, cmap="tab10", s=10, alpha=0.65
        )
        axis.set_title(name)
        axis.set_xlabel("Componente principal 1")
        axis.set_ylabel("Componente principal 2")
        fig.colorbar(scatter, ax=axis, label="Cluster")
    fig.suptitle("Comparación visual de métodos de clustering")
    fig.savefig(OUTPUT_DIR / "clustering_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(kmeans_selection["n_clusters"], kmeans_selection["inertia"], marker="o")
    axes[0].set(title="Método del codo", xlabel="Número de clusters", ylabel="Inercia")
    axes[1].plot(kmeans_selection["n_clusters"], kmeans_selection["silhouette"], marker="o")
    axes[1].set(title="Silhouette de KMeans", xlabel="Número de clusters", ylabel="Silhouette")
    fig.savefig(OUTPUT_DIR / "kmeans_selection.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
