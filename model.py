import keras as k
import pandas as pd
from keras.layers import Dense #Activation, Dropout, Conv2D, LSTM, MaxPooling2D, Flatten, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

data_frame = pd.read_csv("titanic.csv")
input_names = ["Age", "Sex", "Pclass"]
output_names = ["Survived"]

#raw_input_data = data_frame[input_names]
#raw_output_data = data_frame[output_names]

max_age = 100
encoders = {"Age": lambda age: [age/max_age],
            "Sex": lambda gen: {"male": [0], "female": [1]}.get(gen),
            "Pclass": lambda pclass: {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}.get(pclass),
            "Survived": lambda s_value: []}

def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column] = values
    return result

def make_supervised(df):
    raw_input_data = data_frame[input_names]
    raw_output_data = data_frame[output_names]
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data)}
def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted
    #print(vectors)

supervised = make_supervised(data_frame)
encoded_inputs = np.array(encode(supervised["inputs"]))
encoded_outputs = np.array(encode(supervised["outputs"]))

train_x = encoded_inputs[:600]
train_y = encoded_outputs[:600]

test_x = encoded_inputs[600:]
test_y = encoded_outputs[600:]

#model of neuron web
model = k.Sequential()
model.add(k.layers.Dense(units=5, activation="relu"))
model.add(k.layers.Dense(units=1, activation="sigmoid"))
model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])#metrics=["acc"]
model.load_weights("weights.h5")#trying to load trained web
#fit_result = model.fit(x=train_x, y=train_y, epochs=10, validation_split=0.2)

#starting using matplotlib with import matplotlib.pyplot as plt for data-visualisation
"""plt.title("Losses train validation")
plt.plot(fit_result.history["loss"], label="Train")#visualisator of curved, training loss
plt.plot(fit_result.history["val_loss"], label="Validation")#validation loss
plt.legend()#show on shedule
plt.show()"""

"""plt.title("Accuracies train validation")
plt.plot(fit_result.history["accuracy"], label="Train")#visualisator of curved, training loss
plt.plot(fit_result.history["val_accuracy"], label="Validation")#validation loss
plt.legend()#show on shedule
plt.show()"""

predicted_test = model.predict(test_x)
real_data = data_frame.iloc[600:][input_names+output_names]
real_data["PSurvived"] = predicted_test
print(real_data)

#to save previous results
model.save_weights("weights.h5")
#print(encoded_inputs)
#print(encoded_outputs)


#print(supervised)
#print(data_frame["Sex"].unique())
#print(data_frame["Age"].max())