import numpy as np
import pandas as pd
from joblib import dump
import statsmodels, argparse, warnings, os
from statsmodels.discrete.discrete_model import Logit
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("-o", dest='model_dir', type=str, required=True, help="Directory with data to fit the model with")
parser.add_argument("--penalty", dest='penalty', default='none', type=str, help="String for the regularization type")
parser.add_argument("--score", dest='score', default='F1', type=str, help='Metric to maximize when selecting a binarization threshold.')

cmd_line_args = parser.parse_args()
model_dir = cmd_line_args.model_dir
penalty = cmd_line_args.penalty
score = cmd_line_args.score

df_train = pd.read_csv(f"{model_dir}/training_data.csv")
df_val = pd.read_csv(f"{model_dir}/validation_data.csv")

out_dir = os.path.join(model_dir, f"penalty_{penalty}")

os.makedirs(out_dir, exist_ok=True)

predictors = ['Mean_BQ_ALT_allele', 'NEG_COV_RATIO', 'POS_COV_RATIO', 'DISCORDANT_READS_RATIO', 'CLIPPED_BASES_RATIO', 'SAF_prop_deviation_from_half', 'VariantSupportMedianIndex', 'Soft_Clipped_Read_Support']

# max(1 - COV_RATIO) and max(COV_RATIO - 1). None means to not perform clipping on that edge
df_train['NEG_COV_RATIO'] = np.clip(1-df_train['COV_RATIO'], 0, None)
df_train['POS_COV_RATIO'] = np.clip(df_train['COV_RATIO']-1, 0, None)

# max(1 - COV_RATIO) and max(COV_RATIO - 1). None means to not perform clipping on that edge
df_val['NEG_COV_RATIO'] = np.clip(1-df_val['COV_RATIO'], 0, None)
df_val['POS_COV_RATIO'] = np.clip(df_val['COV_RATIO']-1, 0, None)

X_train = df_train[predictors]
y_train = df_train['Real']

X_val = df_val[predictors]

X_train_mean = X_train.mean(axis=0)
X_train_sd = X_train.std(axis=0)

# save these
np.save(f"{model_dir}/X_train_mean.npy", X_train_mean)
np.save(f"{model_dir}/X_train_sd.npy", X_train_sd)

X_train_scaled = (X_train - X_train_mean) / X_train_sd
X_val_scaled = (X_val - X_train_mean) / X_train_sd

# add constant (intercept) because statsmodels doesn't do that by default
X_train_scaled_with_constant = statsmodels.tools.add_constant(X_train_scaled)
X_val_scaled_with_constant = statsmodels.tools.add_constant(X_val_scaled)

# statsmodels logit doesn't have L2 penalty, so just implement it for unpenalized and L1 penalty
def fit_LASSO_logit(X, y, alphas=np.logspace(-5, 5, 11), n_splits=5, maxiter=1000):
    """
    Fit LASSO logistic regression with CV using statsmodels.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Design matrix WITH constant column.
    y : array-like (n_samples,)
        Binary response.
    l1_wt : float
        Elastic net mixing parameter (0=ridge, 1=lasso).
    alphas : iterable
        Regularization strengths to try.
    n_splits : int
        CV folds.
    """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    losses = []
    aucs = []

    for i, alpha in enumerate(alphas):
        
        fold_losses = []
        fold_aucs = []

        for train_idx, val_idx in cv.split(X, y):

            X_cv_train, X_cv_val = X.values[train_idx], X.values[val_idx]
            y_cv_train, y_cv_val = y.values[train_idx], y.values[val_idx]

            cv_model = Logit(y_cv_train, X_cv_train).fit_regularized(method="l1", alpha=alpha, maxiter=maxiter, disp=False)

            y_cv_hat = cv_model.predict(X_cv_val)

            fold_losses.append(sklearn.metrics.log_loss(y_cv_val, y_cv_hat))
            fold_aucs.append(sklearn.metrics.roc_auc_score(y_cv_val, y_cv_hat))
           
        losses.append(np.mean(fold_losses))
        aucs.append(np.mean(fold_aucs))
        print(f"Mean AUC for α = {alpha}: {np.mean(fold_aucs)}")

    # Best hyperparameters
    best_alpha = alphas[np.argmin(losses)]

    # Refit on full data
    final_model = Logit(y, X).fit_regularized(method="l1", alpha=best_alpha, maxiter=maxiter)

    print(f"Best model α = {best_alpha}, Neg. LL = {np.min(losses)}")
    
    return final_model



def fit_elastic_net_logit(X, y, penalty='elasticnet', l1_wts=np.linspace(0, 1, 11), alphas=np.logspace(-5, 5, 11), n_splits=5, maxiter=1000):
    """
    Fit Elastic Net logistic regression with CV using sklearn.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Design matrix WITH constant column.
    y : array-like (n_samples,)
        Binary response.
    l1_wt : float
        Elastic net mixing parameter (0=ridge, 1=lasso).
    alphas : iterable
        Regularization strengths to try.
    n_splits : int
        CV folds.
        
    By default, LogisticRegression and LogisticRegressionCV fit an intercept
    """
    
    assert penalty in ['elasticnet', 'l2']
    
    if penalty == 'elasticnet':
        solver = 'saga'
    else:
        solver = 'liblinear'
        
    # defaults to using Stratified k-fold generator
    model = LogisticRegressionCV(Cs=1/alphas, l1_ratios=l1_wts, cv=n_splits, penalty=penalty, scoring='neg_log_loss', class_weight='balanced', solver=solver).fit(X, y)

    # Best hyperparameters
    if penalty == 'elasticnet':
        print(f"Best model α = {1/model.C_[0]}, L1 ratio = {model.l1_ratio_[0]}, Neg. LL = {model.score(X, y)}")
    else:
        print(f"Best model α = {1/model.C_[0]}, Neg. LL = {model.score(X, y)}")
    
    final_model = LogisticRegression(C=model.C_[0], l1_ratio=model.l1_ratio_[0], penalty=penalty, class_weight='balanced', solver=solver).fit(X, y)
    
    return final_model



if penalty == 'none':
    model = Logit(y_train, X_train_scaled_with_constant).fit()
    
    y_hat = model.predict(X_train_scaled_with_constant).values
    y_val_hat = model.predict(X_val_scaled_with_constant).values
    
elif penalty == 'l1':
    model = fit_LASSO_logit(X_train_scaled_with_constant, y_train)
    
    y_hat = model.predict(X_train_scaled_with_constant).values
    y_val_hat = model.predict(X_val_scaled_with_constant).values
    
else:
    # elastic net model and need to perform cross-validation to get 
    model = fit_elastic_net_logit(X_train_scaled, y_train, penalty=penalty)
    y_hat = model.predict_proba(X_train_scaled)
    y_val_hat = model.predict_proba(X_val_scaled)
    
    pos_class_idx = np.argmax(model.classes_)
    y_hat = y_hat[:, pos_class_idx]
    y_val_hat = y_val_hat[:, pos_class_idx]
    
    

def get_classification_metrics(y_hat, y_true, threshold=0.5):
    
    y_pred = (y_hat >= threshold).astype(int)
    
    df = pd.DataFrame({'predicted': y_pred, 'true': y_true})
    
    TP = len(df.query("predicted==1 and true==1"))
    TN = len(df.query("predicted==0 and true==0"))
    FP = len(df.query("predicted==1 and true==0"))
    FN = len(df.query("predicted==0 and true==1"))
    
    F1 = sklearn.metrics.f1_score(y_true, y_pred)
        
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    
    # compute harmonic mean of sens and spec for mixed samples because TPs dominate, so lots of FPs might not actually change the precision much, but will change spec
    sens_spec_harmonic_mean = 2 * sens * spec / (sens + spec)
    
    # this happens if there are no positives, which occurs when the threshold becomes very high
    try:
        prec = TP / (TP + FP)
    except:
        prec = np.nan
    
    return pd.DataFrame({'Thresh': threshold, 'F1': F1, 'Sens': sens, 'Spec': spec, 'Prec': prec, 'Sens_Spec_Harmonic_Mean': sens_spec_harmonic_mean}, index=[0])


test_thresholds = np.linspace(0, 0.99, 100)
df_metrics_thresholds = []

for thresh in test_thresholds:
    df_metrics_thresholds.append(get_classification_metrics(y_hat, y_train, threshold=thresh))
    
df_metrics_thresholds = pd.concat(df_metrics_thresholds).reset_index(drop=True)
auc = sklearn.metrics.roc_auc_score(y_train, y_hat)
df_metrics_thresholds['AUC'] = auc

best_thresh = df_metrics_thresholds.sort_values(score, ascending=False).Thresh.values[0]
best_metrics = df_metrics_thresholds.sort_values(score, ascending=False).iloc[[0], :]

# save the best classification metrics
best_metrics.to_csv(f"{out_dir}/classification_metrics.csv", index=False)

print(f"AUC = {auc}")
print(f"Best threshold: {best_thresh}\nF1 = {best_metrics.F1.values[0]}\nSens/Spec Harmonic Mean: {best_metrics.Sens_Spec_Harmonic_Mean.values[0]}\nPrecision = {best_metrics.Prec.values[0]}\nRecall = {best_metrics.Sens.values[0]}\nSpecificity = {best_metrics.Spec.values[0]}")


# use the regularization parameter determined above
def perform_permutation_test(model, X, y, num_reps=10000, penalty='elasticnet', progress_bar=False):
    
    assert penalty in ['elasticnet', 'l2']
    
    reg_strength = model.C
    
    if penalty == 'elasticnet':
        solver = 'saga'
        l1_ratio = model.l1_ratio
    else:
        solver = 'liblinear'
        l1_ratio = 0
                    
    coefs = []
    
    for i in range(num_reps):

        if i == 0:
            print(f"Fitting permuted models using {penalty} penalty, L1 ratio = {l1_ratio}, and α = {reg_strength}")
            
        # shuffle phenotypes. np.random.shuffle works in-place
        y_permute = y.copy()
        np.random.shuffle(y_permute)

        rep_model = LogisticRegression(C=reg_strength, l1_ratio=l1_ratio, penalty=penalty, class_weight='balanced', solver=solver)

        rep_model.fit(X, y_permute)
        coefs.append(np.squeeze(rep_model.coef_))
        
        if progress_bar:
            if i % int(num_reps / 10) == 0:
                print(i)
        
    return pd.DataFrame(coefs)



# use the regularization parameter determined above
def perform_bootstrapping(model, X, y, num_bootstrap=10000, penalty='elasticnet', progress_bar=False):
    
    assert penalty in ['elasticnet', 'l2']
    
    reg_strength = model.C
    
    if penalty == 'elasticnet':
        solver = 'saga'
        l1_ratio = model.l1_ratio
    else:
        solver = 'liblinear'
        l1_ratio = 0
                    
    coefs = []
    
    for i in range(num_bootstrap):
        
        if i == 0:
            print(f"Fitting bootstrapped models using {penalty} penalty, L1 ratio = {l1_ratio}, and α = {reg_strength}")

        # randomly draw sample indices
        sample_idx = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)

        bs_model = LogisticRegression(C=reg_strength, l1_ratio=l1_ratio, penalty=penalty, class_weight='balanced', solver=solver)

        # fit the model on the supperset of data
        bs_model.fit(X.values[sample_idx, :], y.values[sample_idx])
        
        coefs.append(np.squeeze(bs_model.coef_))

        if progress_bar:
            if i % int(num_bootstrap / 10) == 0:
                print(i)
        
    return pd.DataFrame(coefs)



def get_coef_and_confidence_intervals(coef_df, permute_df, bootstrap_df, alpha=0.05):
    '''
    Alpha is only used to determine the confidence level of the confidence intervals. Permutation test significance is not determined at this stage. 
    '''
    
    permute_df.columns = coef_df['covariate'].values
    bootstrap_df.columns = coef_df['covariate'].values
    
    # add confidence intervals for the coefficients for all mutation. first check ordering of mutations
    ci = (1-alpha)*100
    diff = (100-ci)/2

    lower, upper = np.percentile(bootstrap_df, axis=0, q=(diff, 100-diff))
    coef_df["Coef_lower"] = lower
    coef_df["Coef_upper"] = upper

    # assess significance using the results of the permutation test
    for i, row in coef_df.iterrows():
        # p-value is the proportion of permutation coefficients that are AT LEAST AS EXTREME as the test statistic
        # ONE-SIDED because we are interested in the sign of the coefficient
        if row["Coef"] > 0:
            coef_df.loc[i, "pval"] = np.mean(permute_df[row["covariate"]] >= row["Coef"])
        else:
            coef_df.loc[i, "pval"] = np.mean(permute_df[row["covariate"]] <= row["Coef"])

    # convert to odds ratios
    coef_df["OR"] = np.exp(coef_df["Coef"])
    coef_df["OR_lower"] = np.exp(coef_df["Coef_lower"])
    coef_df["OR_upper"] = np.exp(coef_df["Coef_upper"])

    return coef_df



# save the model results -- betas, confidence intervals, and p-values
if penalty not in ['l2', 'elasticnet']:
    
    # also save the fitted model
    model.save(f"{out_dir}/logistic_model.pkl")

    model_results = model.conf_int(alpha=0.05, cols=None).reset_index()
    model_results.columns = ['covariate', 'Coef_lower', 'Coef_upper']
    model_results['Coef'] = model.params.values
    model_results['pval'] = model.pvalues.values
    
    for col in ['Coef', 'Coef_lower', 'Coef_upper']:
        model_results[col.replace('Coef', 'OR')] = np.exp(model_results[col])
        
else:

    # also save the fitted model
    dump(model, f"{out_dir}/logistic_model.joblib")
    model_results = pd.DataFrame({'covariate': predictors, 'Coef': np.squeeze(model.coef_)})
    
    permutation_results = perform_permutation_test(model, X_train_scaled, y_train, penalty=penalty)
    bootstrapping_results = perform_bootstrapping(model, X_train_scaled, y_train, penalty=penalty)
    model_results = get_coef_and_confidence_intervals(model_results, permutation_results, bootstrapping_results, alpha=0.05)

    
model_results.to_csv(f"{out_dir}/model_results.csv", index=False)


# add predictions to the original dataframes and save
df_train['predicted'] = y_hat
df_train['pred_class'] = (y_hat >= best_thresh).astype(int)

df_val['predicted'] = y_val_hat
df_val['pred_class'] = (y_val_hat >= best_thresh).astype(int)

df_train.to_csv(f"{out_dir}/training_data_with_predictions.csv", index=False)
df_val.to_csv(f"{out_dir}/validation_data_with_predictions.csv", index=False)