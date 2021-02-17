import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, classification_report, 
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc, average_precision_score


def recall(y, y_hat):
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fn = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 0])
    return tp / float(tp + fn)

def auc(X, y, model):
    probs = model.predict_proba(X)[:,1] 
    return roc_auc_score(y, probs)

def aps(X, y, model):
    probs = model.predict_proba(X)[:,1]
    return average_precision_score(y, probs)

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

    print('Training Accuracy: ', )
    print('Validation Accuracy: ', )
    print('Training F1 Score: ', )
    print('Validation F1 Score: ', )
    print('Training AUC Score: ', )
    print('Validation AUC Score: ', )
    print('Training Recall Score: ', )
    print('Validation Recall Score: ', )
    print('Training Precision Score: ', )
    print('Validation Precision Score: ', )
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