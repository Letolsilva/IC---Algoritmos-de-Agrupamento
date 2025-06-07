import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.io import loadmat
from collections import Counter
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_dataset_from_mat(path):
    mat = loadmat(path)
    data = mat["data"]
    X = data[:, :-1]
    y = data[:, -1].flatten()
    return X, y


def elbow_method(X, max_k=8):
    distortions = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    # Encontrar o cotovelo (pode ser ajustado)
    diff = np.diff(distortions)
    diff2 = np.diff(diff)
    n_clusters = diff2.argmin() + 3  # +3 porque diff2 começa no k=4
    return n_clusters


def cluster_supervised_classifier(X_train, y_train, X_test, clustering_model):
    if isinstance(clustering_model, AgglomerativeClustering):
        clustering_model.fit(X_train)
        cluster_labels = clustering_model.labels_
        # Calcule os centróides dos clusters
        centroids = np.array(
            [
                X_train[cluster_labels == i].mean(axis=0)
                for i in np.unique(cluster_labels)
            ]
        )
        # Para cada amostra de teste, encontre o cluster mais próximo
        from scipy.spatial.distance import cdist

        pred_clusters = np.argmin(cdist(X_test, centroids), axis=1)
    else:
        clustering_model.fit(X_train)
        if hasattr(clustering_model, "labels_"):
            cluster_labels = clustering_model.labels_
        else:
            cluster_labels = clustering_model.predict(X_train)
        pred_clusters = clustering_model.predict(X_test)
    cluster_class_map = {}
    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)[0]
        majority_label = Counter(y_train[indices]).most_common(1)[0][0]
        cluster_class_map[cluster] = majority_label
    print("Mapeamento cluster->classe:", cluster_class_map)
    y_pred = np.array([cluster_class_map.get(cluster, -1) for cluster in pred_clusters])
    return y_pred


def run_experiment(X, y, model_type="kmeans", return_accs=False):
    accs = []
    cms = []
    for s in range(1, 31):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=s, stratify=y
        )
        # Normalização dos dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Print da distribuição das classes
        print(f"Repetição {s} - Distribuição das classes no treino:", Counter(y_train))

        n_clusters = elbow_method(X_train)
        if model_type == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=s)
        elif model_type == "agglo":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif model_type == "gmm":
            model = GaussianMixture(n_components=n_clusters, random_state=s)
        else:
            raise ValueError("Modelo não suportado")
        y_pred = cluster_supervised_classifier(X_train, y_train, X_test, model)
        accs.append(accuracy_score(y_test, y_pred))
        cms.append(confusion_matrix(y_test, y_pred))
    print(f"Modelo: {model_type}")
    print(f"Acurácia média: {np.mean(accs):.4f} | Desvio padrão: {np.std(accs):.4f}")
    print(f"Matriz de confusão (última repetição):\n{cms[-1]}\n")
    if return_accs:
        return accs

if __name__ == "__main__":
    resultados = {}
    for dataset_path, dataset_name in [
        ("Adult.mat", "Adult"),
        ("Dry_bean.mat", "Dry Bean"),
    ]:
        print(f"==== Dataset: {dataset_name} ====")
        X, y = load_dataset_from_mat(dataset_path)
        resultados[dataset_name] = {}
        for model_type in ["kmeans", "agglo", "gmm"]:
            accs = run_experiment(X, y, model_type=model_type, return_accs=True)
            resultados[dataset_name][model_type] = accs

    # Gerar gráficos
    for dataset_name in resultados:
        medias = []
        desvios = []
        labels = []
        for model_type in ["kmeans", "agglo", "gmm"]:
            accs = resultados[dataset_name][model_type]
            medias.append(np.mean(accs))
            desvios.append(np.std(accs))
            labels.append(model_type)
        plt.figure()
        plt.bar(labels, medias, yerr=desvios, capsize=5)
        plt.ylabel('Acurácia Média')
        plt.title(f'Resultados no conjunto {dataset_name}')
        plt.savefig(f'grafico_{dataset_name.lower().replace(" ", "")}.png')
        plt.close()