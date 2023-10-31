import warnings
warnings.filterwarnings('ignore')

from preprocess import *
from reports import *

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer
neg_log_loss_scoring = make_scorer(log_loss, needs_proba=True, labels=np.arange(9), greater_is_better=False)
log_loss_scoring = make_scorer(log_loss, needs_proba=True, labels=np.arange(9), greater_is_better=True)

class RF:
    def __init__(self, params = False, verbose = True, encoder = None):
        self.model = RandomForestClassifier(n_jobs = -1, verbose = 0)
        self.verbose = verbose
        self.enocder = encoder
        self.params = params
        if self.params:
            self.model.set_params(**params)

    def fit(self, train_bands, train_id, bands):
        X_train, y_train, X_calib, y_calib, X_test, y_test,\
            test_cr, test_id, train_cv = prepare_training_data(train_bands, train_id, bands)
        
        # if parameters are set
        if self.params:
            self.model.fit(X_train, y_train)
        # else search for best parameters
        else:
            grid = {
                'n_estimators': [10, 100, 200, 500, 1000],
                'max_depth': [10, 50, 100, None],
                'min_samples_split': [2, 10, 20],
                'min_impurity_decrease': [0, 0.001, 0.01]
            }
            gs = GridSearchCV(self.model, grid, verbose = 10, scoring=neg_log_loss_scoring, n_jobs=-1, cv = train_cv)
            gs.fit(X_train, y_train)
            self.params = gs.best_params_
            self.model = gs.best_estimator_
            if self.verbose:
                print('Best hyperparameters:', gs.best_params_)
                print('Best LogLoss:', -gs.best_score_)

        # asses model with caliration
        self.model_calibrated = CalibratedClassifierCV(estimator=self.model, n_jobs=-1, cv = 'prefit').\
            fit(X_calib, y_calib)
        
        y_proba_calibrated = self.model_calibrated.predict_proba(X_test)
        calibrated_f1 = real_f1(test_id, y_proba_calibrated, self.model, test_cr)

        if self.verbose:
            print('Calibrated Random Forest Classifier')
            real_report(
                test_id, y_proba_calibrated, self.model_calibrated, test_cr, self.enocder
            )

        # asses model without calibration
        self.model.fit(
            np.vstack([X_train, X_calib]),
            np.hstack([y_train, y_calib])
        )

        y_proba = self.model.predict_proba(X_test)
        f1 = real_f1(test_id, y_proba, self.model, test_cr)

        if self.verbose:
            print('Uncalibrated Random Forest Classifier')
            real_report(
                test_id, y_proba, self.model, test_cr, self.enocder
            )

        # choose the best model of the two and refit it
        self.calibrated = f1 <= calibrated_f1
        
        # if caliberated is better fit on train, calibrate on calib, test
        if self.calibrated:
            self.model.fit(X_train, y_train)
            self.model = CalibratedClassifierCV(estimator=self.model, n_jobs=-1, cv = 'prefit').\
                fit(
                    np.vstack([X_calib, X_test]),
                    np.hstack([y_calib, y_test])
                )
        # else fit on whole data
        else:
            self.model.fit(
                np.vstack([X_train, X_calib, X_test]),
                np.hstack([y_train, y_calib, y_test])
            )
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
        
        

from catboost import CatBoostClassifier
import optuna

class CB:
    def __init__(self, params = False, verbose = True, encoder = None):
        self.model = CatBoostClassifier(
            loss_function='MultiClass',
            task_type='CPU',
            thread_count=8,
            silent=True,
            bootstrap_type='MVS'
        )
        self.verbose = verbose
        self.enocder = encoder
        self.params = params
        if self.params:
            self.model.set_params(**params)

    def fit(self, train_bands, train_id, bands):
        X_train, y_train, X_calib, y_calib, X_test, y_test,\
            test_cr, test_id, train_cv = prepare_training_data(train_bands, train_id, bands)
        
        # if parameters are set
        if self.params:
            self.model.fit(X_train, y_train)
        # else search for best parameters
        else:
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 50, log = True),
                    "learning_rate": trial.suggest_float("learning_rate", 5e-2, 0.1, log=True),
                    "depth": trial.suggest_int("depth", 5, 10),
                    "subsample": trial.suggest_float("subsample", 0.4, 0.6),
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.7),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
                    "loss_function": "MultiClass",
                    "task_type": "CPU",
                    "thread_count": 8,
                    "silent": True,
                    "bootstrap_type": "MVS"
                }
                model = CatBoostClassifier(**params)
                return cross_val_score(model, X_train, y_train, scoring=log_loss_scoring, cv = train_cv).mean()

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=30)

            print('Best hyperparameters:', study.best_params)
            print('Best LogLoss:', study.best_value)
            
            self.model = CatBoostClassifier(**study.best_params,
                                            loss_function='MultiClass',
                                            task_type='CPU',
                                            thread_count=8,
                                            silent=True,
                                            bootstrap_type='MVS')
            self.model.fit(X_train, y_train)
            self.params = self.model.get_params()


        # asses model with caliration
        self.model_calibrated = CalibratedClassifierCV(estimator=self.model, n_jobs=-1, cv = 'prefit').\
            fit(X_calib, y_calib)
        
        y_proba_calibrated = self.model_calibrated.predict_proba(X_test)
        calibrated_f1 = real_f1(test_id, y_proba_calibrated, self.model, test_cr)

        if self.verbose:
            print('Calibrated CatBoost Classifier')
            real_report(
                test_id, y_proba_calibrated, self.model_calibrated, test_cr, self.enocder
            )

        # asses model without calibration
        self.model.fit(
            np.vstack([X_train, X_calib]),
            np.hstack([y_train, y_calib])
        )

        y_proba = self.model.predict_proba(X_test)
        f1 = real_f1(test_id, y_proba, self.model, test_cr)

        if self.verbose:
            print('Uncalibrated CatBoost Classifier')
            real_report(
                test_id, y_proba, self.model, test_cr, self.enocder
            )

        # choose the best model of the two and refit it
        self.calibrated = f1 <= calibrated_f1
        
        # if caliberated is better fit on train, calibrate on calib, test
        if self.calibrated:
            self.model.fit(X_train, y_train)
            self.model = CalibratedClassifierCV(estimator=self.model, n_jobs=-1, cv = 'prefit').\
                fit(
                    np.vstack([X_calib, X_test]),
                    np.hstack([y_calib, y_test])
                )
        # else fit on whole data
        else:
            self.model.fit(
                np.vstack([X_train, X_calib, X_test]),
                np.hstack([y_train, y_calib, y_test])
            )
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    

from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope


class XGB:
    def __init__(self, params = False, verbose = True, encoder = None):
        self.model = XGBClassifier(
            nthread = 8,
            objective = 'multi:softmax',
            num_class = 9
        )
        self.verbose = verbose
        self.enocder = encoder
        self.params = params
        if self.params:
            self.model.set_params(**params)

    def fit(self, train_bands, train_id, bands):
        X_train, y_train, X_calib, y_calib, X_test, y_test,\
            test_cr, test_id, train_cv = prepare_training_data(train_bands, train_id, bands)
        
        # if parameters are set
        if self.params:
            self.model.fit(X_train, y_train)
        # else search for best parameters
        else:
            space = {
                'n_estimators': scope.int(hp.qnormal('n_estimators', 1500, 200, 1)),
                'eta': hp.loguniform('eta', 0.02, 0.3),
                'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
                'gamma': hp.loguniform('gamma', 0, 5),
                'subsample': hp.uniform('subsample', 0.2, 0.8),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 0.6),
                'nthread': 8,
                'objective': 'multi:softmax',
                'num_class': 9
            }

            def objective(space):
                model = XGBClassifier(**space)
                score = cross_val_score(model, X_train, y_train, scoring=log_loss_scoring, cv = train_cv).mean()
                print('SCORE', score)
                return {'loss': score, 'status': STATUS_OK}

            trials = Trials()

            best_hyperparams = fmin(fn = objective,
                                    space = space,
                                    algo = tpe.suggest,
                                    max_evals = 30,
                                    trials = trials)

            print('Best hyperparameters:', best_hyperparams)
            print('Best LogLoss:', trials.best_trial['result']['loss'])

            self.model = XGBClassifier(**best_hyperparams)
            self.params = best_hyperparams
            self.model.set_params(nthread = 8, objective = 'multi:softmax', num_class = 9)
            self.model.fit(X_train, y_train)


        # asses model with caliration
        self.model_calibrated = CalibratedClassifierCV(estimator=self.model, n_jobs=-1, cv = 'prefit').\
            fit(X_calib, y_calib)
        
        y_proba_calibrated = self.model_calibrated.predict_proba(X_test)
        calibrated_f1 = real_f1(test_id, y_proba_calibrated, self.model, test_cr)

        if self.verbose:
            print('Calibrated XGBoost Classifier')
            real_report(
                test_id, y_proba_calibrated, self.model_calibrated, test_cr, self.enocder
            )

        # asses model without calibration
        self.model.fit(
            np.vstack([X_train, X_calib]),
            np.hstack([y_train, y_calib])
        )

        y_proba = self.model.predict_proba(X_test)
        f1 = real_f1(test_id, y_proba, self.model, test_cr)

        if self.verbose:
            print('Uncalibrated XGBoost Classifier')
            real_report(
                test_id, y_proba, self.model, test_cr, self.enocder
            )

        # choose the best model of the two and refit it
        self.calibrated = f1 <= calibrated_f1
        
        # if caliberated is better fit on train, calibrate on calib, test
        if self.calibrated:
            self.model.fit(X_train, y_train)
            self.model = CalibratedClassifierCV(estimator=self.model, n_jobs=-1, cv = 'prefit').\
                fit(
                    np.vstack([X_calib, X_test]),
                    np.hstack([y_calib, y_test])
                )
        # else fit on whole data
        else:
            self.model.fit(
                np.vstack([X_train, X_calib, X_test]),
                np.hstack([y_train, y_calib, y_test])
            )
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)