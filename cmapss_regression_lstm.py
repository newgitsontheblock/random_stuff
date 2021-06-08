import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
import keras.backend as K

#########################
# load data
#########################

datafolder = '../../datasets/predictive_maintenance/NASA/CMAPSS/'

colnames = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
            's15', 's16', 's17', 's18', 's19', 's20', 's21']

train_df = pd.read_csv(datafolder + 'train_FD001.txt', sep='\s+', names=colnames)
test_df = pd.read_csv(datafolder + 'test_FD001.txt', sep='\s+', names=colnames)
truth_df = pd.read_csv(datafolder + 'RUL_FD001.txt', header=None)

#########################
# labels training
#########################
"""
column RUL, label1, label2 for regression, binary and multi-class classification respectively
"""

# prepare regression target column - training data

rul = train_df.groupby('id', as_index=False)['cycle'].max()  # get maximum no. of cycles per engine
rul.columns = ['id', 'max']

train_df = train_df.merge(rul, on='id', how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']  # invert cycles to get no. of cycles left till engine failure
train_df.drop('max', axis=1, inplace=True)

# prepare classification target columns (binary, multi-class) - training data
"""
# class w0 means failure is happening in 15 cycles or less
# class w1 means failure is happening in 30 cycles or less
# only label2 uses both w0 and w1 for multi-class classification
"""

w0 = 15
w1 = 30

train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

#########################
# normalization training
#########################

min_max_scaler = MinMaxScaler()

# normalize all features - training data

train_df['cycle_norm'] = train_df['cycle']

cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])

norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize, index=train_df.index)

train_join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)

train_df = train_join_df.reindex(columns=train_df.columns)  # resets order of columns

#########################
# normalization test
#########################

# normalize all features - test data

test_df['cycle_norm'] = test_df['cycle']

cols_normalize = test_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])

norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(test_df[cols_normalize]),
                            columns=cols_normalize, index=test_df.index)

test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)

test_df = test_join_df.reindex(columns=test_df.columns)

# reset sequential row index

test_df = test_df.reset_index(drop=True)

#########################
# labels test
#########################

# prepare ground truth labels

rul = test_df.groupby('id', as_index=False)['cycle'].max()  # get max no. cycles per engine
rul.columns = ['id', 'max']

truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = truth_df['more'] + rul['max']  # get the total no. of cycles till failure per engine
truth_df.drop('more', axis=1, inplace=True)

# prepare regression target column - test data

test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']  # invert cycles to get no. of cycles left till engine failure
test_df.drop('max', axis=1, inplace=True)

# prepare classification target columns (binary, multi-class) - test data

test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

#########################
# reshape training
#########################
"""
only the last observation of every window is relevant for overall window label
if the 'label1' value of last observation is 1 then the engine will fail in predetermined time k (if 0 then not)
each window will have a sequence_length of 50, meaning it will learn how to use the last 50 observations to make
prediction
"""

# reshape input data

sequence_length = 50
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# generator function (returns generator object)
# creates multiples

def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values  # create feature array
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# generator expression (returns generator object)

seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())

seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

# generate regression target (RUL)
# get the last 'label1' value for each window within each id (starting from sequence_length till last observation)

def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['RUL'])
             for id in train_df['id'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)

#########################
# LSTM architecture
#########################

# rmse function

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))

# parameters

nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(LSTM(input_shape=(sequence_length, nb_features), units=100, return_sequences=True, name='lstm_0'))
model.add(Dropout(0.2, name='dropout_0'))

model.add(LSTM(units=50, return_sequences=True, name='lstm_1'))
model.add(Dropout(0.2, name='dropout_1'))

model.add(LSTM(units=25, return_sequences=False, name='lstm_2'))
model.add(Dropout(0.2, name='dropout_2'))

model.add(Dense(units=nb_out, activation='linear', name='output_layer'))

model.compile(loss='mse', optimizer='rmsprop', metrics=[rmse, 'mae'])

# fit

history = model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')])

#########################
# evaluate training
#########################

# evaluate - training data

model.evaluate(seq_array, label_array, verbose=1, batch_size=200)

# predict - training data

y_pred = model.predict(seq_array, verbose=1, batch_size=200)
y_true = label_array

#########################
# reshape test
#########################
"""
take last window (50 timesteps) of each id for prediction
"""

seq_array_test_last = [test_df[test_df['id'] == id][sequence_cols].values[-sequence_length:] for id in
                       test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

y_mask = [len(test_df[test_df['id'] == id]) >= sequence_length for id in test_df['id'].unique()]

label_array_test_last = test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
label_array_test_last = np.expand_dims(label_array_test_last, axis=1).astype(np.float32)

#########################
# evaluate test
#########################

# evaluate - test data

model.evaluate(seq_array_test_last, label_array_test_last, verbose=1, batch_size=200)

# predict - test data

y_pred_test = model.predict(seq_array_test_last, verbose=1, batch_size=200)
y_true_test = label_array_test_last

#########################
# plots
#########################

# plot results

x = np.arange(1, y_true_test.shape[0] + 1)

ax = plt.subplot()
ax.bar(x-0.5, y_pred_test.squeeze(), color='b', width=0.5)
ax.bar(x, y_true_test.squeeze(), color='r', width=0.5)

# plot the last 20 cycles before failure of engine 3 for the first 9 measurements (s1-s9)

engine1 = test_df[test_df['id'] == 3]
engine1 = engine1[engine1['RUL'] <= engine1['RUL'].min() + 20]
engine1 = engine1.filter(regex='s[0-9]$', axis=1)
engine1.plot(subplots=True, sharex=True)

# training plots

plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.legend(['train', 'test'], loc='upper right')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'], loc='upper right')