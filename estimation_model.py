import os
import numpy as np
from numpy import array
from numpy import hstack
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense
from ast import literal_eval
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, RMSprop
import traceback

LOGNAME = 'logs/log.csv'
MODELSFOLDER = 'models/'

INF = 9999999999999999
MACHINES = [["riggeti_aspen8", "riggeti_aspen9", "ionq"],["dwave_advantage","dwave_dw2000"]] #[gate machines, annealing machines+]
TYPES = ["gate", "annealing"]
N_STEPS = 5

# split a multivariate sequence into samples
def split_sequences(data,n_steps):
    sequences_X = []
    sequences_y = []

    sequence_X = []
    sequence_y = []

    data = data.sort_values(['machine', 'timestamp'], ascending=[True, True])
    data['machine'] = pd.Categorical(data['machine'], categories=["riggeti_aspen8", "riggeti_aspen9", "ionq","dwave_advantage","dwave_dw2000"])
    data['day'] = pd.Categorical(data['day'], categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    data_new = pd.get_dummies(data)
    data_new = data_new.join(data["machine"])
    data = data_new

    print(data.columns)

    data.drop("timestamp", axis=1, inplace=True)

    #generate sequences
    actual_machine = data.iloc[0]["machine"]
    for i in range(0, data.shape[0]):
        if actual_machine != data.iloc[i]["machine"]: #machine changes
            sequence_X = []
            sequence_y = []
            actual_machine = data.iloc[i]["machine"]

        sequence_X.append(data.iloc[i].drop(["execution_time","machine"]))
        sequence_y.append(data.iloc[i]["execution_time"])
        if len(sequence_X)==n_steps:
            sequences_X.append(sequence_X.copy())
            sequences_y.append([sequence_y[n_steps-1]])
            sequence_X.pop(0)
            sequence_y.pop(0)

    return array(sequences_X), array(sequences_y)


def define_model(n_steps, n_features):
    opt = Adam(learning_rate=0.0002)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(5,  activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return model

def train_lstm(model, epochs,  X_train, X_test, y_train, y_test):
    # training the model
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        shuffle=False)
    return model, history

def generate_model(data, n_steps=N_STEPS):
    X, y = split_sequences(data, n_steps)
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    n_features = X.shape[2]
    model = define_model(n_steps, n_features)
    model.summary()


    epochs = 400
    model, history = train_lstm(model, epochs,  X_train, X_test, y_train, y_test)

    print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    print("epochs:", epochs)
    print("acc:", acc)
    print("val_acc:", val_acc)
    print("loss:", loss)
    print("val_loss:", val_loss)

    return model


def save_model(name, model):
    model.save(os.path.join(MODELSFOLDER, name))

def load_model_(name):
    return load_model(os.path.join(MODELSFOLDER, name))


def generate_all_models():

    try:
        data = pd.read_csv(LOGNAME)

        filter_gate = (data["machine"].isin(MACHINES[0]))
        filter_annealing = (data["machine"].isin(MACHINES[1]))

        data_gate = data[filter_gate]
        data_annealing = data[filter_annealing]
        data_annealing.drop("qubits", axis=1, inplace=True)

        data_gate = data_gate.sort_values(['machine', 'timestamp'], ascending=[True, True])

        if data_gate.shape[0] > 0:
            # actual_machine = data_gate.iloc[0]["machine"]
            # previous_time = 0
            # for index in range(0, data_gate.shape[0]):
            #     if data_gate.iloc[index]["machine"] != actual_machine:
            #         actual_machine = data_gate.iloc[index]["machine"]
            #         previous_time = 0
            #     actual_time = data_gate.iloc[index]["execution_time"]
            #     data_gate.iloc[index]["execution_time"] = previous_time
            #     previous_time = actual_time

            model = generate_model(data_gate)
            save_model("gate.h5", model)

        if data_annealing.shape[0] > 0:
            # actual_machine = data_annealing.iloc[0]["machine"]
            # previous_time = 0
            # for index in range(0, data_annealing.shape[0]):
            #     if data_annealing.iloc[index]["machine"] != actual_machine:
            #         actual_machine = data_annealing.iloc[index]["machine"]
            #         previous_time = 0
            #     actual_time = data_annealing.iloc[index]["execution_time"]
            #     data_annealing.iloc[index]["execution_time"] = previous_time
            #     previous_time = actual_time

            model = generate_model(data_annealing)
            save_model("annealing.h5", model)

        return True

    except Exception as e:
        traceback.print_exc()
        print(e)
        return False




def get_previous_execution_time(machine):
    try:
        data = pd.read_csv(LOGNAME)
    except:
        return 0
    filter_machine = (data["machine"] == machine)
    data_machine = data[filter_machine]

    if data_machine.shape[0] == 0:  # there are not data
        return 0

    data_machine = data_machine.sort_values(['timestamp'], ascending=[False])
    return data_machine.iloc[0]["execution_time"]




def write_feedback(machine, qubits, shots, time, day, execution_time):
    try:
        ts = datetime.datetime.now().timestamp()
        with open(LOGNAME, 'a') as f:
            if not os.path.exists("./"+str(LOGNAME)):
                print("·no existe")
                f.write("machine,qubits,shots,time,day,previous_execution_time,execution_time,timestamp\n")
            f.write(str(machine)+","+str(qubits)+","+str(shots)+","+str(time)+","+str(day)+","+str(get_previous_execution_time(machine))+","+str(execution_time)+","+str(ts)+"\n")
            f.close()
    except Exception as e:
        print(e)
        f.close()
        raise Exception



def get_data_predict(machine, qubits, shots, time, week_day):
    data = pd.read_csv(LOGNAME)
    filter_machine = (data["machine"] == machine)
    data_machine = data[filter_machine]

    if data_machine.shape[0] < 5:  # there are not sufficient data
        return INF


    row = {}
    row["machine"]= machine
    row["qubits"] = qubits
    row["shots"] = shots
    row["time"] = time
    row["day"] = week_day
    row["previous_execution_time"] = get_previous_execution_time(machine)
    row["timestamp"] = datetime.datetime.now().timestamp()
    row["execution_time"] = None

    data_machine.append(row, ignore_index=True)

    if qubits == None:
        data_machine.drop("qubits", axis=1, inplace=True)

    data_machine = data_machine.sort_values(['timestamp'], ascending=[False])

    data_machine.drop("timestamp", axis=1, inplace=True)
    data_machine.drop("execution_time", axis=1, inplace=True)

    data_machine['machine'] = pd.Categorical(data_machine['machine'], categories=["riggeti_aspen8", "riggeti_aspen9", "ionq","dwave_advantage","dwave_dw2000"])
    data_machine['day'] = pd.Categorical(data_machine['day'], categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    data_machine = pd.get_dummies(data_machine)

    print(data_machine.columns)
    Xs = []
    for i in reversed(range(5)):
        Xs.append(data_machine.iloc[i])

    return Xs



def execute_gate_prediction(model, machine, qubits, shots, time, week_day):

    X = array(get_data_predict(machine, qubits, shots, time, week_day))

    X = X.reshape((1, X.shape[0], X.shape[1]))
    yhat = model.predict(X, verbose=0)
    print(yhat)
    return yhat


def execute_annealing_prediction(model, machine, shots, time, week_day):

    x = array(get_data_predict(machine, None, shots, time, week_day))

    x = x.reshape((1, x.shape[0],  x.shape[1]))
    yhat = model.predict(x, verbose=0)
    print(yhat)
    return yhat

def predict_time(type_machine, machine, qubits, shots, time, week_day):

    try:
        if type_machine == "gate":
            model = load_model_("gate.h5")
            prediction = execute_gate_prediction(model, machine, qubits, shots, time, week_day)

        else:
            model = load_model_("annealing.h5")
            prediction = execute_annealing_prediction(model, machine, shots, time, week_day)

        return prediction[0][0]
    except Exception as e:
        print(e)
        return INF
















# #%%
#
# # multivariate output stacked lstm example
#
# #%%
#
#
#
#
# #%%
#
# data["functions"] = data["functions"].apply(literal_eval)
# data["structures"] = data["structures"].apply(literal_eval)
# data["ambiental factors"] = data["ambiental factors"].apply(literal_eval)
#
# #%% md
#
# **Define variables**
#
# #%%
#
# n_steps = 2 #number of evaluations to predict next
# n_features = 3
# epochs = 400
#
# #%% md
#
# **Data to format LSTM need**
#
# #%%
#
# X = np.array([]).reshape(0,n_steps, n_features)
# y = np.array([]).reshape(0,n_features)
# number_evaluations = 0
# number_patients = 0
# for index, row in data.iterrows():
#     number_patients= number_patients+1
#     functions = array(row["functions"])
#     structures = array(row["structures"])
#     ambiental = array(row["ambiental factors"])
#     print("Functional length:",len(functions))
#     number_evaluations = number_evaluations + len(functions)
#     # convert to [rows, columns] structure
#     functions = functions.reshape((len(functions), 1))
#     structures = structures.reshape((len(structures), 1))
#     ambiental = ambiental.reshape((len(ambiental), 1))
#     # horizontally stack columns
#     dataset = hstack((functions, structures, ambiental))
#     # convert into input/output
#     X_one_row, y_one_row = split_sequences(dataset, n_steps)
#     print(X_one_row.shape)
#     print(y_one_row.shape)
#     if len(X_one_row.shape)>=3:
#         X = np.concatenate((X,X_one_row),axis=0)
#         y = np.concatenate((y,y_one_row),axis=0)
#
# #%% md
#
# **Split data into train and test sets**
#
# #%%
#
# from sklearn.model_selection import train_test_split
# print(X.shape)
# print(y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# #X_train, X_test, y_train, y_test = X, X, y, y
#
# #%% md
#
# **Define the model**
#
# #%%
#
# # define model
# from keras.optimizers import Adam, RMSprop
#
# def define_model(n_features):
#     opt = Adam(learning_rate=0.0002)
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
#     model.add(Dense(n_features))
#     model.compile(optimizer=opt, loss='mse', metrics = ['accuracy'])
#     return model
#
# #%%
#
# n_features = X.shape[2]
# model = define_model(n_features)
# model.summary()
#
# #%% md
#
# **Fit the model**
#
# #%%
#
# def train_lstm(model, epochs):
#     #training the model
#     history = model.fit(X_train, y_train,
#               epochs=epochs,
#               verbose=1,
#               validation_data=(X_test, y_test),
#               shuffle=False)
#     return model, history
#
# #%%
#
# model, history = train_lstm(model,epochs)
#
# #%%
#
# import matplotlib.pyplot as plt
#
# print(history.history.keys())
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# epochs = range(1, len(acc) + 1)
#
# print(epochs)
#
# plt.plot(epochs, acc, 'r', label = 'Training acc')
# plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()
#
# #%%
#
# import matplotlib.pyplot as plt
#
# print(history.history.keys())
# acc = history.history['loss']
# val_acc = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# print(epochs)
#
# plt.plot(epochs, acc, 'k', label = 'Training loss')
# plt.plot(epochs, val_acc, 'k--', label = 'Validation loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
#
# #%% md
#
# **Test the model**
#
# #%%
#
# # demonstrate prediction
# #x_input = array([[12, 52, 27] , [12, 50, 27], [ 33, 55, 53]])
# x_input = array([ [12, 50, 27], [ 33, 55, 53]])
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)