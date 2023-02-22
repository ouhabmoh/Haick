# Oversampling using SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Shape of X_resampled after SMOTE:", X_resampled.shape)
print("Shape of y_resampled after SMOTE:", y_resampled.shape)

# Undersampling using Tomek links
from imblearn.under_sampling import TomekLinks

tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X, y)
print("Shape of X_resampled after Tomek links:", X_resampled.shape)
print("Shape of y_resampled after Tomek links:", y_resampled.shape)

# Gradient boosting with XGBoost
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}
num_round = 100
bst = xgb.train(param, dtrain, num_round)
y_pred = bst.predict(dtest)
y_pred_binary = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy for XGBoost: {:.3f}".format(accuracy))

# Gradient boosting with LightGBM
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}
num_round = 100
gbm = lgb.train(params, lgb_train, num_round, valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=5)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_binary = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy for LightGBM: {:.3f}".format(accuracy))

# Convolutional neural network (CNN) for image classification
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=10, batch_size=32)

# Recurrent neural network (RNN) for sequence classification
from keras.layers import LSTM, Embedding

# Define the RNN architecture
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid
