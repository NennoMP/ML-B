FINAL MODEL:
The used final model is an Stacking Regressor Ensemble (Ridge, alpha=30), which is trained on the predictions of three selected models: (1) a Neural Network with SGD, (2) a Neural Network with ADAM, and (3) a Support Vector Regressor. The ensemble takes the predictions of such models and learns from them. Note that, since neural network are not deterministic in their predictions (unlike the SVR), we take the mean of the predictions over five trials, to have more robustness and generalization.

VALIDATION PROCEDURE:
The three selected models have been selected through a KFold (k=5) cross-validation on a 90% train/val split of the development set, according to the mean validation MEE across the splits. This model selection step also included a coarse grid-search and a fine-tune random search to identify the best hyper-parameter configuration for each architecture. Other models such as RandomForest and KNN have not been taken into account for the ensemble since they had a very high mean validation MEE w.r.t. the other three, and we believed they would negatively affect the results. Finally, we have compared and validated several ensemble approaches, as well as the individual constituent models, with Hold-out cross-validation on a further 20% validation split taken from the 90% train/val set. The final model (i.e., Stacking Regressor Ensemble with Ridge) has been selected based on the lowest hold-out validation MEE.