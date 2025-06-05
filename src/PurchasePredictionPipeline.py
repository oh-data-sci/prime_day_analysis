class PurchasePredictionPipeline:
    """
    a pipeline for predicting, from mixed categorical and numerical data, whether user made a purchase on prime day.
    """
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = None
        
    def create_preprocessor(self, X):
        """
        a preprocessing pipeline for mixed data types
        """
        
        # identify and separate out numerical and categorical columns
        numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist() # none at present
        
        print(f"numerical features: {numerical_features}")
        print(f"categorical features: {categorical_features}")
        
        # preprocessing steps
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        # combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor


    def _get_feature_names(self, X):
        """
        feature names after preprocessing
        """
        feature_names = []
        
        # numerical features
        numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
        feature_names.extend(numerical_features)
        
        # categorical features (after one-hot encoding)
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            unique_values = X[col].unique()
            # OneHotEncoder drops first category
            for val in sorted(unique_values)[1:]:
                feature_names.append(f"{col}_{val}")
        
        return feature_names

    def prepare_data(self, df, target_col='is_purchaser', test_size=0.2, random_state=42):
        """
        convert dataframe into training data shape, and split up testing/training data
        """
        
        # separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # separate out training data and testing data with random selection
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # preprocessor on all data 
        # (note the harmless leakage! test data is allowed to influence scaler ranges)
        self.preprocessor = self.create_preprocessor(X)
        
        # transform the data in scaled 
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # store feature names in object data for later use
        self.feature_names = self._get_feature_names(X)
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    
    def build_logistic_regression(self, X_train, y_train):
        """
        baseline model: logistic regression
        """
        
        print("\n" + "="*50)
        print("LOGISTIC REGRESSION")
        print("="*50)
        
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # class imbalance
        )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        print("✓ logistic regression model trained")
        return model
    
    def build_random_forest(self, X_train, y_train):
        """
        comparison model no 1:  random forest
        """
        
        print("\n" + "="*50)
        print("RANDOM FOREST")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        print("✓ random forest model trained")
        return model
    
    def build_xgboost(self, X_train, y_train):
        """
        comparison model no 2: XGBoost
        """
        
        print("\n" + "="*50)
        print("XGBOOST")
        print("="*50)
        
        # calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        print("✓ XGBoost model trained")
        return model
    
    def get_feature_importance(self, model_name):
        """
        feature importance reporting for tree-based models
        """
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        
        if model_name == 'random_forest':
            importance = model.feature_importances_
        elif model_name == 'xgboost':
            importance = model.feature_importances_
        else:
            print(f"Feature importance not available for {model_name}")
            return None
        
        # feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def train_all_models(self, df):
        """
        run pipeline and train all models
        """
        
        print("STARTING TRAINING PIPELINE")
        print("="*50)
        
        # prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        print(f"\nshape of training data: {X_train.shape}")
        print(f"shape of test data: {X_test.shape}")
        print(f"number of features after preprocessing: {X_train.shape[1]}")
        
        # train all models
        self.build_logistic_regression(X_train, y_train)
        self.build_random_forest(X_train, y_train)
        self.build_xgboost(X_train, y_train)
        
        print(f"\n" + "="*50)
        print("ALL MODELS TRAINED")
        print(f"models available: {list(self.models.keys())}")
        print("="*50)
        
        return X_train, X_test, y_train, y_test
