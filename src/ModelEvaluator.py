class ModelEvaluator:
    """
    evaluation class for classification models
    """
    
    
    def __init__(self, models, X_train, X_test, y_train, y_test, feature_names=None):
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.predictions = {}
        self.probabilities = {}
        self.metrics = {}
    
    
    def generate_predictions(self):
        """
        predictions and probabilities for all models
        """
        
        print("="*50)
        print("INFERRING PREDICTIONS FROM ALL MODELS")
        print("="*50)
        
        for name, model in self.models.items():
            # predict the class
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # probability of positive class
            
            self.predictions[name] = y_pred
            self.probabilities[name] = y_pred_proba
            
            print(f"✓ inferred predictions for {name}")
        
        print("predictions generated\n")
    
    def calculate_metrics(self):
        """
        comprehensive metrics for all models
        """
        
        print("="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        
        metrics_data = []
        
        for name in self.models.keys():
            y_pred = self.predictions[name]
            y_pred_proba = self.probabilities[name]
            
            # model quality metrics
            accuracy = accuracy_score(self.y_test, y_pred)                 # do not use
            precision = precision_score(self.y_test, y_pred)               # how many predicted to purchase, actually did purchase?
            recall = recall_score(self.y_test, y_pred)                     # how many actually purchasing were predicted to do so?
            f1 = f1_score(self.y_test, y_pred)                             # a balance of precision and recall
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)             # area under receiver-operator curve
            auc_pr = average_precision_score(self.y_test, y_pred_proba)    # average precision score
            
            metrics_data.append({
                'model': name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'ROC-AUC': auc_roc,
                'PR-AUC': auc_pr
            })
            
            self.metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': auc_roc,
                'pr_auc': auc_pr
            }
        
        self.metrics_df = pd.DataFrame(metrics_data)
        
        print("all metrics calculated\n")
        return self.metrics_df
    
    def cross_validation_scores(self, cv_folds=5):
        """
        cross-validation score for models
        """
        
        print("="*50)
        print("CROSS-VALIDATION")
        print("="*50)
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # combine train and test sets for CV
        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.hstack([self.y_train, self.y_test])
        
        for name, model in self.models.items():
            # perform cross-validation for multiple metrics
            cv_accuracy = cross_val_score(model, X_full, y_full, cv=skf, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_full, y_full, cv=skf, scoring='f1')
            cv_roc_auc = cross_val_score(model, X_full, y_full, cv=skf, scoring='roc_auc')
            
            cv_results[name] = {
                'accuracy_mean': cv_accuracy.mean(),
                'accuracy_std': cv_accuracy.std(),
                'f1_mean': cv_f1.mean(),
                'f1_std': cv_f1.std(),
                'roc_auc_mean': cv_roc_auc.mean(),
                'roc_auc_std': cv_roc_auc.std()
            }
            
            print(f"{name}:")
            print(f"  accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std():.4f})")
            print(f"  f1-Score: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
            print(f"  ROC-AUC:  {cv_roc_auc.mean():.4f} (±{cv_roc_auc.std():.4f})")

        
        self.cv_results = cv_results
        return cv_results
    
    def plot_confusion_matrices(self):
        """
        confusion matrices for all models
        """
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('confusion matrices comparison', fontsize=16, fontweight='bold')
        
        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['no purchase', 'purchase'],
                       yticklabels=['no purchase', 'purchase'],
                       ax=axes[idx])
            
            axes[idx].set_title(f'{name.replace("_", " ").title()}')
            axes[idx].set_xlabel('predicted')
            axes[idx].set_ylabel('actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self):
        """
        plot ROC curves
        """
        
        plt.figure(figsize=(10, 8))
        
        for name, y_pred_proba in self.probabilities.items():
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('false positive rate', fontsize=12)
        plt.ylabel('true positive rate', fontsize=12)
        plt.title('ROC comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self):
        """
        plotting precision-recall curves
        """
        
        plt.figure(figsize=(10, 8))
        
        for name, y_pred_proba in self.probabilities.items():
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            avg_precision = average_precision_score(self.y_test, y_pred_proba)
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{name.replace("_", " ").title()} (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = (self.y_test == 1).mean()
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'random classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('recall', fontsize=12)
        plt.ylabel('precision', fontsize=12)
        plt.title('precision-recall curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance_comparison(self):
        """
        compare feature importance across tree-based models
        """
        
        tree_models = ['random_forest', 'xgboost']
        available_models = [name for name in tree_models if name in self.models.keys()]
        
        if len(available_models) < 2:
            print("need at least 2 tree-based models for comparison")
            return
        
        fig, axes = plt.subplots(1, len(available_models), figsize=(6*len(available_models), 8))
        if len(available_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(available_models):
            model = self.models[model_name]
            importance = model.feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Plot horizontal bar chart
            axes[idx].barh(range(len(feature_importance_df)), 
                          feature_importance_df['importance'])
            axes[idx].set_yticks(range(len(feature_importance_df)))
            axes[idx].set_yticklabels(feature_importance_df['feature'])
            axes[idx].set_xlabel('feature importance')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nfeature importance')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_prediction_confidence(self):
        """
        analyze prediction confidence distributions
        """
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('prediction confidence distributions', fontsize=16, fontweight='bold')
        
        for idx, (name, probabilities) in enumerate(self.probabilities.items()):
            # separate probabilities by actual class
            prob_no_purchase = probabilities[self.y_test == 0]
            prob_purchase = probabilities[self.y_test == 1]
            
            axes[idx].hist(prob_no_purchase, bins=30, alpha=0.7, 
                          label='no purchase (actual)', color='lightcoral')
            axes[idx].hist(prob_purchase, bins=30, alpha=0.7, 
                          label='purchase (actual)', color='lightblue')
            
            axes[idx].set_xlabel('predicted probability of purchase')
            axes[idx].set_ylabel('frequency')
            axes[idx].set_title(f'{name.replace("_", " ").title()}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_detailed_report(self):
        """
        model evaluation report
        """
        
        print("\n" + "="*50)
        print("MODEL EVALUATION REPORT")
        print("="*50)
        
        # metrics comparison
        print("\n1. PERFORMANCE METRICS COMPARISON")
        print("-"*50)
        print(self.metrics_df.round(4).to_string(index=False))
        
        # Best model identification
        print(f"\n2. BEST MODEL BY METRIC")
        print("-"*50)
        for metric in ['accuracy', 'precision', 'recall', 'f1-score', 'ROC-AUC', 'PR-AUC']:
            best_model = self.metrics_df.loc[self.metrics_df[metric].idxmax(), 'model']
            best_score = self.metrics_df[metric].max()
            print(f"{metric:12}: {best_model:15} ({best_score:.4f})")
        
        # cross-validation results
        if hasattr(self, 'cv_results'):
            print(f"\n3. CROSS-VALIDATION RESULTS ({5}-FOLD)")
            print("-"*50)
            for name, results in self.cv_results.items():
                print(f"\n{name.replace('_', ' ').title()}:")
                print(f"  accuracy: {results['accuracy_mean']:.4f} (±{results['accuracy_std']:.4f})")
                print(f"  f1-score: {results['f1_mean']:.4f} (±{results['f1_std']:.4f})")
                print(f"  ROC-AUC:  {results['roc_auc_mean']:.4f} (±{results['roc_auc_std']:.4f})")
        
        # feature importance insights for tree models
        print(f"\n4. KEY INSIGHTS FROM FEATURE IMPORTANCE")
        print("-"*50)
        
        tree_models = ['random_forest', 'xgboost']
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                importance = model.feature_importances_
                
                # Get top 3 features
                top_indices = np.argsort(importance)[-3:][::-1]
                
                print(f"\n{model_name.replace('_', ' ').title()} - top 3 features:")
                for i, idx in enumerate(top_indices, 1):
                    print(f"  {i}. {self.feature_names[idx]}: {importance[idx]:.4f}")
        
        # model recommendations
        print(f"\n5. MODEL RECOMMENDATIONS")
        print("-"*50)
        
        # Find overall best model (weighted average of key metrics)
        weights = {'f1-score': 0.3, 'ROC-AUC': 0.3, 'PR-AUC': 0.25, 'precision': 0.15}
        weighted_scores = {}
        
        for _, row in self.metrics_df.iterrows():
            model_name = row['model']
            weighted_score = sum(row[metric] * weight for metric, weight in weights.items())
            weighted_scores[model_name] = weighted_score
        
        best_overall = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
        
        print(f"\noverall best model: {best_overall.replace('_', ' ').title()}")
        print(f"weighted score: {weighted_scores[best_overall]:.4f}")
        
        print(f"\nmodel-specific insights:")
        print(f"• logistic regression: baseline, interpretable coefficients")
        print(f"• random forest: handles feature interactions, feature type diversity, robust to outliers")
        print(f"• XGBoost: often best performance, handles imbalanced data well")
        
        print("\n"+"="*50)
    
    def run_complete_evaluation(self):
        """run the complete evaluation pipeline."""
        
        print("="*50)
        print("STARTING MODEL EVALUATION")
        print("="*50)
        
        # generate predictions
        self.generate_predictions()
        
        # compute metrics
        self.calculate_metrics()
        
        # cross-validation
        self.cross_validation_scores()
        
        # generate plots
        print("VISUALIZATION PLOTS")
        print("="*50)
        
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        # self.plot_feature_importance_comparison()
        self.analyze_prediction_confidence()
        
        # generate comprehensive report
        self.generate_detailed_report()
        
        return self.metrics_df