import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, auc, average_precision_score, confusion_matrix, roc_auc_score, classification_report, plot_precision_recall_curve
from collections import Counter
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KernelDensity

def accuracy(y, y_hat):
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    tn = sum([1 for i in y_y_hat if i[0] == 0 and i[1] == 0])
    return (tp + tn) / float(len(y_y_hat))

def f1(y, y_hat):
    precision_score = precision(y, y_hat)
    recall_score = recall(y, y_hat)
    numerator = precision_score * recall_score
    denominator = precision_score + recall_score
    return 2 * (numerator / denominator)

def precision(y, y_hat):
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fp = sum([1 for i in y_y_hat if i[0] == 0 and i[1] == 1])
    return tp / float(tp + fp)

def recall(y, y_hat):
    # Your code here
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fn = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 0])
    return tp / float(tp + fn)

def auc(X, y, model):
    probs = model.predict_proba(X)[:,1] 
    return roc_auc_score(y, probs)

def auc2(X, y, model):
    probs = model.decision_function(X)
    return roc_auc_score(y, probs)

def aps(X, y, model):
    probs = model.predict_proba(X)[:,1]
    return average_precision_score(y, probs)

def aps2(X, y, model):
    probs = model.decision_function(X)
    return average_precision_score(y, probs)

def get_metric(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    ac_tr = accuracy_score(y_tr, y_pred_tr)
    ac_val= accuracy_score(y_val, y_pred_val)
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    au_tr = auc(X_tr, y_tr, model)
    au_val = auc(X_val, y_val, model)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    aps_tr = aps(X_tr, y_tr, model)
    aps_val = aps(X_val, y_val, model)

    print('Training Accuracy: ', ac_tr)
    print('Validation Accuracy: ', ac_val)
    print('Training F1 Score: ', f1_tr)
    print('Validation F1 Score: ', f1_val)
    print('Training AUC Score: ', au_tr)
    print('Validation AUC Score: ', au_val)
    print('Training Recall Score: ', rc_tr)
    print('Validation Recall Score: ', rc_val)
    print('Training Precision Score: ', pr_tr)
    print('Validation Precision Score: ', pr_val)
    print('Training Average Precision Score: ', aps_tr)
    print('Validation Average Precision Score: ', aps_val)

def get_metrics(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    ac_tr = accuracy_score(y_tr, y_pred_tr)
    ac_val= accuracy_score(y_val, y_pred_val)
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    au_tr = auc(X_tr, y_tr, model)
    au_val = auc(X_val, y_val, model)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    aps_tr = aps(X_tr, y_tr, model)
    aps_val = aps(X_val, y_val, model)

    print('Training Accuracy: ', ac_tr)
    print('Validation Accuracy: ', ac_val)
    print('Training F1 Score: ', f1_tr)
    print('Validation F1 Score: ', f1_val)
    print('Training AUC Score: ', au_tr)
    print('Validation AUC Score: ', au_val)
    print('Training Recall Score: ', rc_tr)
    print('Validation Recall Score: ', rc_val)
    print('Training Precision Score: ', pr_tr)
    print('Validation Precision Score: ', pr_val)
    print('Training Average Precision Score: ', aps_tr)
    print('Validation Average Precision Score: ', aps_val)
    print('')
    print("Training Classification Report: ")
    print(classification_report(y_tr, y_pred_tr))
    print("")
    print("Validation Classification Report: ")
    print(classification_report(y_val, y_pred_val))
    
    cnf = confusion_matrix(y_val, y_pred_val)
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf.flatten()/np.sum(cnf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cnf, annot=labels, fmt='', cmap='Blues', annot_kws={'size':16})

def get_metric_2(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    ac_tr = accuracy_score(y_tr, y_pred_tr)
    ac_val= accuracy_score(y_val, y_pred_val)
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    au_tr = auc2(X_tr, y_tr, model)
    au_val = auc2(X_val, y_val, model)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    aps_tr = aps2(X_tr, y_tr, model)
    aps_val = aps2(X_val, y_val, model)

    print('Training Accuracy: ', ac_tr)
    print('Validation Accuracy: ', ac_val)
    print('Training F1 Score: ', f1_tr)
    print('Validation F1 Score: ', f1_val)
    print('Training AUC Score: ', au_tr)
    print('Validation AUC Score: ', au_val)
    print('Training Recall Score: ', rc_tr)
    print('Validation Recall Score: ', rc_val)
    print('Training Precision Score: ', pr_tr)
    print('Validation Precision Score: ', pr_val)
    print('Training Average Precision Score: ', aps_tr)
    print('Validation Average Precision Score: ', aps_val)

def get_metrics_2(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    ac_tr = accuracy_score(y_tr, y_pred_tr)
    ac_val= accuracy_score(y_val, y_pred_val)
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    au_tr = auc2(X_tr, y_tr, model)
    au_val = auc2(X_val, y_val, model)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    aps_tr = aps2(X_tr, y_tr, model)
    aps_val = aps2(X_val, y_val, model)

    print('Training Accuracy: ', ac_tr)
    print('Validation Accuracy: ', ac_val)
    print('Training F1 Score: ', f1_tr)
    print('Validation F1 Score: ', f1_val)
    print('Training AUC Score: ', au_tr)
    print('Validation AUC Score: ', au_val)
    print('Training Recall Score: ', rc_tr)
    print('Validation Recall Score: ', rc_val)
    print('Training Precision Score: ', pr_tr)
    print('Validation Precision Score: ', pr_val)
    print('Training Average Precision Score: ', aps_tr)
    print('Validation Average Precision Score: ', aps_val)
    print('')
    print("Training Classification Report: ")
    print(classification_report(y_tr, y_pred_tr))
    print("")
    print("Validation Classification Report: ")
    print(classification_report(y_val, y_pred_val))
    
    cnf = confusion_matrix(y_val, y_pred_val)
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf.flatten()/np.sum(cnf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cnf, annot=labels, fmt='', cmap='Blues', annot_kws={'size':16})

def plot_feature_importances(X, model):
    n_features = X.shape[1]
    plt.figure(figsize=(8, 8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns.values)
    plt.xlabel("Feature Importance")
    plt.ylabel('Feature')


def pr_curve(X, y, model):
    y_score = model.decision_function(X) 
    ap = average_precision_score(y, y_score)
    disp = plot_precision_recall_curve(model, X, y)
    disp.ax_.set_title('Precision-Recall Curve: AP={0:0.2f}'.format(ap))

def pr_curve2(X, y, model):
    y_score = model.predict_proba(X)[:,1]
    ap = average_precision_score(y, y_score)
    disp = plot_precision_recall_curve(model, X, y)
    disp.ax_.set_title('Precision-Recall Curve: AP={0:0.2f}'.format(ap))


def sampling(X_tr, y_tr, X_val, y_val, model, best_model):
    X_tr_res, y_tr_res = model.fit_resample(X_tr, y_tr)
    print("Training Count: ", Counter(y_tr_res))
    fit = best_model.fit(X_tr_res, y_tr_res)
    y_pred_tr = fit.predict(X_tr_res)
    y_pred_val = fit.predict(X_val)
    get_metric(X_tr_res, y_tr_res, X_val, y_val, y_pred_tr, y_pred_val, fit)

def sampling2(X_tr, y_tr, X_val, y_val, model, best_model):
    X_tr_res, y_tr_res = model.sample(X_tr, y_tr)
    print("Training Count: ", Counter(y_tr_res))
    fit = best_model.fit(X_tr_res, y_tr_res)
    y_pred_tr = fit.predict(X_tr_res)
    y_pred_val = fit.predict(X_val)
    get_metric(X_tr_res, y_tr_res, X_val, y_val, y_pred_tr, y_pred_val, fit)

def plot_agglomerative_algorithm():
    # generate synthetic two-dimensional data
    X, y = make_blobs(random_state=0, n_samples=12)

    agg = AgglomerativeClustering(n_clusters=X.shape[0], compute_full_tree=True).fit(X)

    fig, axes = plt.subplots(X.shape[0] // 5, 5, subplot_kw={'xticks': (),
                                                             'yticks': ()},
                             figsize=(20, 8))

    eps = X.std() / 2

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    gridpoints = np.c_[xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)]

    for i, ax in enumerate(axes.ravel()):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        agg.n_clusters = X.shape[0] - i
        agg.fit(X)
        ax.set_title("Step %d" % i)
        ax.scatter(X[:, 0], X[:, 1], s=60, c='grey')
        bins = np.bincount(agg.labels_)
        for cluster in range(agg.n_clusters):
            if bins[cluster] > 1:
                points = X[agg.labels_ == cluster]
                other_points = X[agg.labels_ != cluster]

                kde = KernelDensity(bandwidth=.5).fit(points)
                scores = kde.score_samples(gridpoints)
                score_inside = np.min(kde.score_samples(points))
                score_outside = np.max(kde.score_samples(other_points))
                levels = .8 * score_inside + .2 * score_outside
                ax.contour(xx, yy, scores.reshape(100, 100), levels=[levels],
                           colors='k', linestyles='solid', linewidths=2)

    axes[0, 0].set_title("Initialization")


def plot_agglomerative():
    X, y = make_blobs(random_state=0, n_samples=12)
    agg = AgglomerativeClustering(n_clusters=3)

    eps = X.std() / 2.

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    gridpoints = np.c_[xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)]

    ax = plt.gca()
    for i, x in enumerate(X):
        ax.text(x[0] + .1, x[1], "%d" % i, horizontalalignment='left', verticalalignment='center')

    ax.scatter(X[:, 0], X[:, 1], s=60, c='grey')
    ax.set_xticks(())
    ax.set_yticks(())

    for i in range(11):
        agg.n_clusters = X.shape[0] - i
        agg.fit(X)

        bins = np.bincount(agg.labels_)
        for cluster in range(agg.n_clusters):
            if bins[cluster] > 1:
                points = X[agg.labels_ == cluster]
                other_points = X[agg.labels_ != cluster]

                kde = KernelDensity(bandwidth=.5).fit(points)
                scores = kde.score_samples(gridpoints)
                score_inside = np.min(kde.score_samples(points))
                score_outside = np.max(kde.score_samples(other_points))
                levels = .8 * score_inside + .2 * score_outside
                ax.contour(xx, yy, scores.reshape(100, 100), levels=[levels],
                           colors='k', linestyles='solid', linewidths=1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)