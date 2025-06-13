import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.io import loadmat
from collections import Counter
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_dataset_from_mat(path):
    mat = loadmat(path)
    data = mat["data"]
    X = data[:, :-1]
    y = data[:, -1].flatten()
    return X, y

# Cotovelo utilizado com o k_kmeans
def elbow_method(X, max_k=8): 
    distortions = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    # Encontrar o cotovelo (pode ser ajustado)
    diff = np.diff(distortions)
    diff2 = np.diff(diff)
    n_clusters = diff2.argmin() + 3  # +3 porque diff2 começa no k=4
    return n_clusters

def plot_elbow_mean(X, seeds, max_k, dataset_name, model_type):
    all_distortions = []
    ks = list(range(2, max_k + 1))
    print(f"Calculando o método do cotovelo com sementes {seeds}")
    for s in seeds:
        distortions = []
        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=s, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
        all_distortions.append(distortions)
    all_distortions = np.array(all_distortions)
    mean_distortions = np.mean(all_distortions, axis=0)
    std_distortions = np.std(all_distortions, axis=0)

    plt.figure()
    plt.plot(ks, mean_distortions, marker='o', label='Média das distorções')
    plt.fill_between(ks, mean_distortions-std_distortions, mean_distortions+std_distortions, alpha=0.2, label='Desvio padrão')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.title(f'Método do Cotovelo (Média) - {dataset_name} - {model_type}')
    plt.legend()
    plt.savefig(f'elbow_mean_{dataset_name}_{model_type}.png')
    plt.close()

# Silhouette Score para AgglomerativeClustering.
def get_best_k_silhouette(X, seeds, max_k=8):
    print(f"Calculando o melhor k para AgglomerativeClustering com sementes {seeds}")
    ks = list(range(2, max_k + 1))
    mean_scores = []
    for k in ks:
        print(f"Calculando para k={k}")
        scores = []
        for s in seeds:
            print(f"  Semente {s}")
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X):
                print(f"  Calculando Silhouette Score para k={k} e semente {s}")
                score = silhouette_score(X, labels)
            else:
                print(f"  Silhouette Score não aplicável para k={k} e semente {s}")
                score = np.nan
            scores.append(score)
        mean_scores.append(np.nanmean(scores))
    best_k = ks[np.nanargmax(mean_scores)]
    return best_k

def plot_silhouette_mean(X, seeds, max_k, dataset_name, model_type):
    all_scores = []
    ks = list(range(2, max_k + 1))
    for s in seeds:
        scores = []
        for k in ks:
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
            # Silhouette só faz sentido se k > 1 e < n amostras
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X):
                score = silhouette_score(X, labels)
            else:
                score = np.nan
            scores.append(score)
        all_scores.append(scores)
    all_scores = np.array(all_scores)
    mean_scores = np.nanmean(all_scores, axis=0)
    std_scores = np.nanstd(all_scores, axis=0)

    plt.figure()
    plt.plot(ks, mean_scores, marker='o', label='Média do Silhouette')
    plt.fill_between(ks, mean_scores-std_scores, mean_scores+std_scores, alpha=0.2, label='Desvio padrão')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette (Média) - {dataset_name} - {model_type}')
    plt.legend()
    plt.savefig(f'silhouette_mean_{dataset_name}_{model_type}.png')
    plt.close()

# BIC para GaussianMixture.
def get_best_k_bic(X, seeds, max_k=8):
    ks = list(range(2, max_k + 1))
    mean_bics = []
    for k in ks:
        bics = []
        for s in seeds:
            model = GaussianMixture(n_components=k, random_state=s)
            model.fit(X)
            bics.append(model.bic(X))
        mean_bics.append(np.mean(bics))
    best_k = ks[np.argmin(mean_bics)]
    return best_k

def plot_bic_mean(X, seeds, max_k, dataset_name, model_type):
    all_bics = []
    ks = list(range(2, max_k + 1))
    for s in seeds:
        bics = []
        for k in ks:
            model = GaussianMixture(n_components=k, random_state=s)
            model.fit(X)
            bic = model.bic(X)
            bics.append(bic)
        all_bics.append(bics)
    all_bics = np.array(all_bics)
    mean_bics = np.mean(all_bics, axis=0)
    std_bics = np.std(all_bics, axis=0)

    plt.figure()
    plt.plot(ks, mean_bics, marker='o', label='Média do BIC')
    plt.fill_between(ks, mean_bics-std_bics, mean_bics+std_bics, alpha=0.2, label='Desvio padrão')
    plt.xlabel('Número de componentes (k)')
    plt.ylabel('BIC')
    plt.title(f'BIC (Média) - {dataset_name} - {model_type}')
    plt.legend()
    plt.savefig(f'bic_mean_{dataset_name}_{model_type}.png')
    plt.close()

def cluster_supervised_classifier(X_train, y_train, X_test, clustering_model):
    if isinstance(clustering_model, AgglomerativeClustering):
        clustering_model.fit(X_train)
        cluster_labels = clustering_model.labels_
        centroids = np.array(
            [
                X_train[cluster_labels == i].mean(axis=0)
                for i in np.unique(cluster_labels)
            ]
        )
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
    seeds = list(range(1, 3))
    max_k = 8
    for s in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=s, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model_type == "kmeans":
            n_clusters = elbow_method(X_train, max_k=max_k)
            model = KMeans(n_clusters=n_clusters, random_state=s, n_init=10)
        elif model_type == "agglo":
            print(f"Calculando o melhor k para AgglomerativeClustering com semente {s}")
            n_clusters = get_best_k_silhouette(X_train, seeds, max_k=max_k)
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif model_type == "gmm":
            n_clusters = get_best_k_bic(X_train, seeds, max_k=max_k)
            model = GaussianMixture(n_components=n_clusters, random_state=s)
        else:
            raise ValueError("Modelo não suportado")

        y_pred = cluster_supervised_classifier(X_train, y_train, X_test, model)
        accs.append(accuracy_score(y_test, y_pred))
        cms.append(confusion_matrix(y_test, y_pred))
        mean_cm = np.mean(cms, axis=0)
        mean_cm = np.round(mean_cm).astype(int)  # Arredonda para inteiro
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
    print(f"Modelo: {model_type}")
    print(f"Acurácia média: {mean_acc:.4f} | Desvio padrão: {std_acc:.4f}")
    print(f"Matriz de confusão média:\n{mean_cm}\n")
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
        seeds = list(range(1, 3))
        for model_type in ["kmeans", "agglo", "gmm"]:
            accs = run_experiment(X, y, model_type=model_type, return_accs=True)
            print(f"Resultados para {model_type} no dataset {dataset_name}: {accs}")
            resultados[dataset_name][model_type] = accs
            if model_type == "kmeans":
                print(f"Calculando o melhor k para KMeans no dataset {dataset_name}")
                plot_elbow_mean(X, seeds, max_k=8, dataset_name=dataset_name, model_type=model_type)
            elif model_type == "agglo":
                plot_silhouette_mean(X, seeds, max_k=8, dataset_name=dataset_name, model_type=model_type)
            elif model_type == "gmm":
                plot_bic_mean(X, seeds, max_k=8, dataset_name=dataset_name, model_type=model_type)
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