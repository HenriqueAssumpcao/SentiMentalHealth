import numpy as np 
def huber_approx_obj(dtrain, preds):
    """
    NOT USED IN THE PAPER
    """
    d = preds - dtrain 
    h = .5  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess

""" https://en.wikipedia.org/wiki/Huber_loss """

def huber_loss_sklearn(y_true,y_pred):
    """
    NOT USED IN THE PAPER
    """
    h = .5
    abs_error = np.abs(y_true-y_pred)
    return np.mean((abs_error <= h) * (.5 * abs_error**2) +
                 (abs_error >  h) * (h * abs_error - .5 * h**2) )

def huber_approx_sklearn(y_true,y_pred):
    """
    NOT USED IN THE PAPER
    """
    delta = .5
    a = np.abs(y_true-y_pred)
    return np.mean(delta**2 * (np.sqrt(1+(a/delta)**2)-1))

def huber_approx_xgb(y_pred, y_true):
    """
    NOT USED IN THE PAPER
    """
    return 'huber_approx', huber_approx_sklearn(y_true.get_label(),y_pred)

def huber_loss_xgb(y_pred, y_true):
    """
    NOT USED IN THE PAPER
    """
    return 'huber', huber_loss_sklearn(y_true.get_label(),y_pred)
  
def weightedhuber_approx_obj(dtrain, preds):
    """
    NOT USED IN THE PAPER
    """
    weights = bin_weights[(np.floor((dtrain - MIN_VALUE)/BIN_WIDTH)).astype(int)]
    d = preds - dtrain#.get_labels() #remove .get_labels() for sklearn
    h = .5  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = weights*(d / scale_sqrt)
    hess = weights*(1 / scale / scale_sqrt)
    return grad, hess

def weightedhuber_approx_sklearn(y_true,y_pred):
    """
    NOT USED IN THE PAPER
    """
    weights = bin_weights[np.floor((y_true - MIN_VALUE)/BIN_WIDTH).astype(int)]
    delta = .5
    a = np.abs(y_true-y_pred)
    return np.mean(weights*(delta**2 * (np.sqrt(1+(a/delta)**2)-1)))

def weightedhuber_approx_xgb(y_pred, y_true):
    """
    NOT USED IN THE PAPER
    """
    return 'weightedhuber_approx', weightedhuber_approx_sklearn(y_true.get_label(),y_pred)

def weightedl1_obj(y_true, predt):
    weights = bin_weights[(np.floor((y_true - MIN_VALUE)/BIN_WIDTH)).astype(int)]
    hess = np.zeros((predt.shape))
    grad = np.zeros((predt.shape))
    grad = (((y_true < predt) * 2) - 1) * weights 

    grad /= len(predt)
    return grad,hess

def weightedl1_loss_sklearn(y_true,y_pred):
    losses = np.abs(np.subtract(y_true,y_pred))
    weights = bin_weights[np.floor((y_true - MIN_VALUE)/BIN_WIDTH).astype(int)]
    return np.mean(losses*weights)

def weightedl1_loss_xgb(y_pred,y_true):
    return 'WeightedL1', weightedl1_loss_sklearn(y_true.get_label(),y_pred)


def plot_model_error(model,model_name):
    
    results =  model.evals_result()
    epochs = len(results['validation_0'][model_name])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots(figsize=(15,10))
    ax.plot(x_axis, results['validation_0'][model_name], label='Train')
    ax.plot(x_axis, results['validation_1'][model_name], label='Test')
    ax.plot(x_axis, results['validation_2'][model_name], label='Validation')
    ax.legend()
    plt.ylabel(model_name)
    plt.title('XGBoost ' + model_name)
    plt.show()


def hyperParameterTuning_xgb(model,param_tuning,X_train, y_train,X_val=None,y_val=None,early_stop=None,eval_metric=None):
    gsearch = GridSearchCV(estimator = model,
                           param_grid = param_tuning,                        
                           cv = [(slice(None), slice(None))],
                           scoring = scorer_xgb,
                           verbose = 5,
                           n_jobs = 4)
    if early_stop == None:
        gsearch.fit(X_train,y_train)
        return gsearch.best_params_
    else:
        eval_set = [(X_val,y_val)]
        gsearch.fit(X=X_train,y=y_train,eval_set = eval_set,early_stopping_rounds = early_stop,verbose=False,eval_metric = eval_metric)
        return gsearch.best_params_

def getKeysByValue(dictOfElements, valueToFind):
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            return item[0]
    

def grid_search_xgb(param_grid,early_stop,X_train,y_train,X_val,y_val,n_jobs,MODELSDIR):
    params = ['learning_rate','max_depth','min_child_weight','subsample','colsample_bytree','n_estimators','objective']

    #obj to int
    objective2int = {}
    aux = 1
    for objective in param_grid['objective']:
        objective2int[objective] = aux
        aux += 1
    #calculating num_fits
    num_param_list = list()
    for key in param_grid:
        num_param_list.append(len(param_grid[key]))
    num_fits = np.prod(num_param_list)

    print(f"Initializing grid search...")  
    print("-----------------------------")  
    #creating dataframes
    print("Looking for checkpoints...")
    print("-----------------------------")
    try:
        param_df = pd.read_csv(MODELSDIR + "param_df.csv")
        val_error_df = pd.read_csv(MODELSDIR + "val_error_df.csv")
        print("Checkpoints successfully loaded.")
    except:
        param_df = pd.DataFrame(columns=params)
        val_error_df = pd.DataFrame(columns=['val_error'])
        print("No checkpoints were found, initializing grid search from start.")
    print("-----------------------------")
    #initializing variables
    curr_params_dict = {'learning_rate': 0,
        'max_depth': 0,
        'min_child_weight': 0,
        'subsample': 0,
        'colsample_bytree': 0,
        'n_estimators' : 0,
        'objective': 0}
    curr_params = np.zeros((1,len(params)))

    eval_set = [(X_train,y_train),(X_test,y_test),(X_val,y_val)] 
    counter = 0
    
    print(f"Computing {num_fits} fits with {n_jobs} concurrent workers...")

    for l_rate in param_grid['learning_rate']:
        curr_params_dict['learning_rate'] = l_rate
        curr_params[0,0] = l_rate
        

        for depth in param_grid['max_depth']:
            curr_params_dict['max_depth'] = depth
            curr_params[0,1] = depth
           

            for m_child_weight in param_grid['min_child_weight']:
                curr_params_dict['min_child_weight'] = m_child_weight
                curr_params[0,2] = m_child_weight
                

                for subs in param_grid['subsample']:
                    curr_params_dict['subsample'] = subs
                    curr_params[0,3] = subs
                    

                    for colsample in param_grid['colsample_bytree']:
                        curr_params_dict['colsample_bytree'] = colsample
                        curr_params[0,4] = colsample
                        

                        for n_est in param_grid['n_estimators']:
                            curr_params_dict['n_estimators'] = n_est
                            curr_params[0,5] = n_est
                            

                            for obj in param_grid['objective']:
                                curr_params_dict['objective'] = obj
                                curr_params[0,6] = objective2int[obj]
                                ilocs = np.where(np.all(curr_params == param_df.to_numpy(),axis=1))[0]
                                if len(ilocs) == 0:
                                    
                                    xgb_model = XGBRegressor(**curr_params_dict,n_jobs = n_jobs)
                                    xgb_model.fit(X_train,y_train,eval_set=eval_set,early_stopping_rounds=early_stop,verbose=False)

                                    val_error_df = val_error_df.append({'val_error': mean_squared_error(y_val,xgb_model.predict(X_val))},ignore_index=True)
                                    curr_params_dict['objective'] = getKeysByValue(objective2int, curr_params_dict['objective'])
                                    param_df = param_df.append(curr_params_dict,ignore_index=True)
                                    counter += 1

                                    param_df.to_csv(MODELSDIR + "param_df.csv",index=False)
                                    val_error_df.to_csv(MODELSDIR + "val_error_df.csv",index=False)

                                else:
                                    counter += 1

                                if counter % 10 == 0:
                                    print(f"{counter} fits done out of {num_fits}")
                                    print(f"Progress: {np.round((counter/num_fits)*100,2)}%")

                                if counter % 50 == 0:
                                    param_df.to_csv(MODELSDIR + f"param_df_backup{counter}.csv",index=False)
                                    val_error_df.to_csv(MODELSDIR + f"val_error_backup{counter}_df.csv",index=False)

    best_model_index = val_error_df.idxmin().values[0]
    best_parameters = dict(param_df.iloc[best_model_index])

    best_parameters['max_depth'] = int(best_parameters['max_depth'])
    best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
    best_parameters['objective'] = getKeysByValue(objective2int, best_parameters['objective'])

    print("Grid search finished!")
    print(f"Smallest validation error was {np.round(float(val_error_df.min()),4)} at the {best_model_index}th iteration.")
    print(f"The best parameters found were: \n{best_parameters}")
    pd.concat([param_df,val_error_df],1).to_csv(MODELSDIR + f"grid_search_results_{counter}.csv",index=False)
    os.remove('param_df.csv')
    os.remove('val_error_df.csv')
    
    return best_parameters, best_model_index