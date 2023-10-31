import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, ConfusionMatrixDisplay


def report(y_true, y_pred, y_proba, model):
    print('%s:\nAccuracy: %.4f\nF1 macro: %.4f\nLogLoss: %.4f' %
          (type(model).__name__,
           accuracy_score(y_true.iloc[:, 1], y_pred),
           f1_score(y_true.iloc[:, 1], y_pred, average = 'macro'),
           log_loss(y_true.iloc[:, 1], y_proba)))


def real_report(y_true, y_proba, model, crown, le):
    tmp_prob = pd.DataFrame(columns = ['crown_id'] + model.classes_.tolist())
    tmp_prob['crown_id'] = crown
    tmp_prob.iloc[:, 1:] = y_proba

    y_prob = tmp_prob.groupby('crown_id').mean()
    y_pred = y_prob.idxmax(axis = 1).values.tolist()
    y_prob = y_prob.values
    
    print('Accuracy: %.4f\nF1 macro: %.4f\nLogLoss: %.4f' %
          (accuracy_score(y_true.tolist(), y_pred),
           f1_score(y_true.tolist(), y_pred, average = 'macro'),
           log_loss(y_true, y_prob)))
    cm = confusion_matrix(y_true.tolist(), y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(model.classes_))
    disp.plot()
    plt.show()


def real_log_loss(y_true, y_proba, model, crown):
    tmp_prob = pd.DataFrame(columns = ['crown_id'] + model.classes_.tolist())
    tmp_prob['crown_id'] = crown
    tmp_prob.iloc[:, 1:] = y_proba
    y_prob = tmp_prob.groupby('crown_id').mean().values
    return log_loss(y_true, y_prob)


def real_accuracy(y_true, y_proba, model, crown):
    tmp = pd.DataFrame(columns = ['crown_id'] + model.classes_.tolist())
    tmp['crown_id'] = crown
    tmp.iloc[:, 1:] = y_proba
    y_prob = tmp.groupby('crown_id').mean()
    y_pred = y_prob.idxmax(axis = 1).values.tolist()
    return accuracy_score(y_true.tolist(), y_pred)


def real_f1(y_true, y_proba, model, crown):
    tmp = pd.DataFrame(columns = ['crown_id'] + model.classes_.tolist())
    tmp['crown_id'] = crown
    tmp.iloc[:, 1:] = y_proba
    y_prob = tmp.groupby('crown_id').mean()
    y_pred = y_prob.idxmax(axis = 1).values.tolist()
    return f1_score(y_true.tolist(), y_pred, average = 'macro')


def real_accuracy_xgb(y_true, y_proba, model, encoder):
    y_true_xgb = y_true.copy()
    y_true_xgb[['species_id']] = encoder.transform(y_true_xgb[['species_id']])
    tmp = pd.DataFrame(columns = ['crown_id'] + model.classes_.tolist())
    tmp['crown_id'] = y_true_xgb['crown_id']
    tmp.iloc[:, 1:] = y_proba
    y_tr= y_true_xgb.groupby('crown_id').first()
    y_prob = tmp.groupby('crown_id').mean()
    y_pred = y_prob.idxmax(axis = 1)
    return accuracy_score(y_tr, y_pred)


def real_f1_xgb(y_true, y_proba, model, encoder):
    y_true_xgb = y_true.copy()
    y_true_xgb[['species_id']] = encoder.transform(y_true_xgb[['species_id']])
    tmp = pd.DataFrame(columns = ['crown_id'] + model.classes_.tolist())
    tmp['crown_id'] = y_true_xgb['crown_id']
    tmp.iloc[:, 1:] = y_proba
    y_tr= y_true_xgb.groupby('crown_id').first()
    y_prob = tmp.groupby('crown_id').mean()
    y_pred = y_prob.idxmax(axis = 1)
    return f1_score(y_tr, y_pred,  average = 'macro')


def real_report_xgb(y_true, y_proba, model, encoder):
    y_true_xgb = y_true.copy()
    y_true_xgb[['species_id']] = encoder.transform(y_true_xgb[['species_id']])
    tmp = pd.DataFrame(columns = ['crown_id'] + model.classes_.tolist())
    tmp['crown_id'] = y_true_xgb['crown_id']
    tmp.iloc[:, 1:] = y_proba
    y_tr= y_true_xgb.groupby('crown_id').first()
    y_prob = tmp.groupby('crown_id').mean()
    y_pred = y_prob.idxmax(axis = 1)
    
    print('%s:\nAccuracy: %.4f\nF1 macro: %.4f\nLogLoss: %.4f' %
          (type(model).__name__,
           accuracy_score(y_tr.iloc[:,0].values.tolist(), y_pred.to_list()),
           f1_score(y_tr.iloc[:,0].values.tolist(), y_pred.to_list(), average = 'macro'),
           log_loss(y_tr.iloc[:,0].values.tolist(), y_prob)))
    cm = confusion_matrix(y_tr.iloc[:,0].values.tolist(), y_pred.to_list(), labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()