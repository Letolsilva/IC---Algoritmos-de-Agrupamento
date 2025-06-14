import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from scipy.io import loadmat
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset_from_mat(path):
    mat = loadmat(path)
    data = mat["data"]
    X = data[:, :-1]
    y = data[:, -1].flatten()
    return X, y


def plot_metric_vs_k(ks, means, stds, ylabel, title, filename):
    plt.figure()
    plt.plot(ks, means, marker="o", label="Média")
    plt.fill_between(ks, means - stds, means + stds, alpha=0.2, label="Erro padrão")
    plt.xlabel("Número de clusters")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix_normalized(cm, class_labels, title, filename):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_boxplot_accuracies(accs_dict, dataset_name):
    plt.figure()
    data = [accs_dict[mt] for mt in ["kmeans", "agglo", "gmm"]]
    plt.boxplot(data, tick_labels=["KMeans", "Agglo", "GMM"])
    plt.ylabel("Acurácia")
    plt.title(f"Boxplot das acurácias - {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"boxplot_acuracia_{dataset_name.lower().replace(' ', '')}.png")
    plt.close()


def plot_pca_clusters(X, labels, title, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def cluster_supervised_classifier(X_train, y_train, X_test, clustering_model):
    print("Treinando modelo de clustering...")
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
        print("Treinando modelo de clustering...")
        clustering_model.fit(X_train)
        if hasattr(clustering_model, "labels_"):
            cluster_labels = clustering_model.labels_
        else:
            cluster_labels = clustering_model.predict(X_train)
        pred_clusters = clustering_model.predict(X_test)
    cluster_class_map = {}
    for cluster in np.unique(cluster_labels):
        print(f"Mapeando cluster {cluster} para rótulo...")
        indices = np.where(cluster_labels == cluster)[0]
        majority_label = Counter(y_train[indices]).most_common(1)[0][0]
        cluster_class_map[cluster] = majority_label
    y_pred = np.array([cluster_class_map.get(cluster, -1) for cluster in pred_clusters])
    return y_pred, cluster_labels


if __name__ == "__main__":
    resultados = {}
    seeds = range(1, 31)  # aumente para range(1, 31) depois de testar
    max_k = 5
    for dataset_path, dataset_name in [
        ("Adult.mat", "Adult"),
        ("Dry_bean.mat", "Dry Bean"),
    ]:
        print(f"\n==== Dataset: {dataset_name} ====")
        X, y = load_dataset_from_mat(dataset_path)
        resultados[dataset_name] = {}

        print("Normalizando dados...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ks = list(range(2, max_k + 1))

        # Salvar resultados para cada método
        resultados_kmeans = []
        resultados_agglo = []
        resultados_gmm = []
        stds_kmeans = []
        stds_agglo = []
        stds_gmm = []
        # Para boxplot e PCA
        all_accs = {"kmeans": [], "agglo": [], "gmm": []}
        first_X_train = first_labels_dict = {}

        for k in ks:
            distortions = []
            silhouettes = []
            bics = []
            for s in seeds:
                # KMeans
                print(f"Executando KMeans para k={k}, seed={s}...")
                kmeans = KMeans(n_clusters=k, random_state=s, n_init=10)
                kmeans.fit(X_scaled)
                distortions.append(kmeans.inertia_)
                # Agglomerative
                print(f"Executando Agglomerative para k={k}, seed={s}...")
                agglo = AgglomerativeClustering(n_clusters=k)
                print("Ajustando AgglomerativeClustering...")
                labels = agglo.fit_predict(X_scaled)
                if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(
                    X_scaled
                ):
                    print("Calculando silhouette score...")
                    score = silhouette_score(X_scaled, labels)
                else:
                    print("Silhouette score não calculável, atribuindo NaN...")
                    score = np.nan
                silhouettes.append(score)
                # GMM
                print(f"Executando GMM para k={k}, seed={s}...")
                gmm = GaussianMixture(n_components=k, random_state=s)
                gmm.fit(X_scaled)
                bics.append(gmm.bic(X_scaled))
            resultados_kmeans.append(np.mean(distortions))
            resultados_agglo.append(np.nanmean(silhouettes))
            resultados_gmm.append(np.mean(bics))
            stds_kmeans.append(np.std(distortions))
            stds_agglo.append(np.nanstd(silhouettes))
            stds_gmm.append(np.std(bics))

        # Gráficos com erro padrão
        print("Plotando gráfico de distorção do KMeans...")
        plot_metric_vs_k(
            ks,
            np.array(resultados_kmeans),
            np.array(stds_kmeans),
            ylabel="Distorção (Inércia)",
            title=f"KMeans - Distorção média por número de clusters - {dataset_name}",
            filename=f"distorcao_kmeans_{dataset_name.lower().replace(' ', '')}.png",
        )
        print("Plotando gráfico de silhouette do Agglomerative...")
        plot_metric_vs_k(
            ks,
            np.array(resultados_agglo),
            np.array(stds_agglo),
            ylabel="Silhouette Média",
            title=f"Agglomerative - Silhouette média por número de clusters - {dataset_name}",
            filename=f"silhouette_agglo_{dataset_name.lower().replace(' ', '')}.png",
        )
        print("Plotando gráfico de BIC do GMM...")
        plot_metric_vs_k(
            ks,
            np.array(resultados_gmm),
            np.array(stds_gmm),
            ylabel="BIC Médio",
            title=f"GMM - BIC médio por número de clusters - {dataset_name}",
            filename=f"bic_gmm_{dataset_name.lower().replace(' ', '')}.png",
        )

        # Escolha o melhor k a partir dos resultados já calculados
        best_k_kmeans = ks[np.argmin(resultados_kmeans)]
        best_k_agglo = ks[np.nanargmax(resultados_agglo)]
        best_k_gmm = ks[np.argmin(resultados_gmm)]

        print(f"KMeans melhor k: {best_k_kmeans}")
        print(f"Agglomerative melhor k: {best_k_agglo}")
        print(f"GMM melhor k: {best_k_gmm}")

        # Experimentos só com o melhor k de cada método
        for model_type, best_k in zip(
            ["kmeans", "agglo", "gmm"], [best_k_kmeans, best_k_agglo, best_k_gmm]
        ):
            accs = []
            cms = []
            first_X_train = None
            first_labels = None
            for idx, s in enumerate(seeds):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=s, stratify=y
                )
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                if model_type == "kmeans":
                    model = KMeans(n_clusters=best_k, random_state=s, n_init=10)
                elif model_type == "agglo":
                    model = AgglomerativeClustering(n_clusters=best_k)
                elif model_type == "gmm":
                    model = GaussianMixture(n_components=best_k, random_state=s)
                else:
                    raise ValueError("Modelo não suportado")
                y_pred, cluster_labels = cluster_supervised_classifier(
                    X_train, y_train, X_test, model
                )
                accs.append(accuracy_score(y_test, y_pred))
                cms.append(confusion_matrix(y_test, y_pred))
                if idx == 0:
                    first_X_train = X_train
                    first_labels = cluster_labels
            mean_cm = np.mean(cms, axis=0)
            mean_cm = np.round(mean_cm, 2)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"Modelo: {model_type}")
            print(f"Acurácia média: {mean_acc:.4f} | Desvio padrão: {std_acc:.4f}")
            print(f"Matriz de confusão média:\n{mean_cm}\n")
            resultados[dataset_name][model_type] = accs

            # Matriz de confusão normalizada (heatmap)
            class_labels = [str(int(c)) for c in np.unique(y)]
            plot_confusion_matrix_normalized(
                mean_cm,
                class_labels,
                title=f"Matriz de Confusão Normalizada - {dataset_name} - {model_type}",
                filename=f"cm_normalizada_{dataset_name.lower().replace(' ', '')}_{model_type}.png",
            )

            # Clusters com PCA (usando primeira repetição)
            plot_pca_clusters(
                first_X_train,
                first_labels,
                title=f"Clusters com PCA - {dataset_name} - {model_type}",
                filename=f"pca_clusters_{dataset_name.lower().replace(' ', '')}_{model_type}.png",
            )

            # Salva acurácias para boxplot
            all_accs[model_type] = accs

        # Boxplot das acurácias
        plot_boxplot_accuracies(all_accs, dataset_name)
