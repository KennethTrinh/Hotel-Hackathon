import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, optimizers, losses, metrics, regularizers
#import models, regularizers, layers, optimizers, losses, metrics
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical



os.chdir( os.path.join( os.getcwd(), 'Desktop'))

data = pd.read_csv('mimic3d.csv')
data_full = data.drop('hadm_id', 1)

y = data_full['LOSgroupNum']
X = data_full.drop('LOSgroupNum', 1)
X = X.drop('LOSdays', 1)
X = X.drop('ExpiredHospital', 1)
X = X.drop('AdmitDiagnosis', 1)
X = X.drop('AdmitProcedure', 1)
X = X.drop('marital_status', 1)
X = X.drop('ethnicity', 1)
X = X.drop('religion', 1)
X = X.drop('insurance', 1)

categorical_columns = [
                    'gender',                     
                    'admit_type',
                    'admit_location'
                      ]

for col in categorical_columns:
    #if the original column is present replace it with a one-hot
    if col in X.columns:
        one_hot_encoded = pd.get_dummies(X[col])
        X = X.drop(col, axis=1)
        X = X.join(one_hot_encoded, lsuffix='_left', rsuffix='_right')

XnotNorm = X.copy()

x = XnotNorm.values #returns a numpy array
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)
XNorm = pd.DataFrame(x_scaled, columns=XnotNorm.columns)

X_train, X_test, y_train, y_test = train_test_split(XNorm, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(30,)),
    Dropout(0.5),
    Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

yTrain = to_categorical(y_train)
yVal = to_categorical(y_test)


model.fit(X_train, yTrain, epochs=100, batch_size=32, validation_data=(X_test, yVal))

model.save('./model')


# write a function to plot a straight line
"""
def plot_line(x1, x2):
    # plot the line
    plt.plot([x1, x2], [m*x1 + b, m*x2 + b], 'r')
"""
# translate that code to R and plot the line
