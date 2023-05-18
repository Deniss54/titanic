import pandas as pd
import keras
from keras.optimizers import Adam
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

data_frame = pd.read_csv('titanic.csv')

input_name = ['Age', 'Sex', 'Pclass']
output_name = ['Survived']

max_age = 80 # максимальный возраст в дф 80 лет
# encoders  будем использовать для нормализации данных
# входные данные [возраст(int), пол(str), класс(int)]
# необходимо преобразовать в след вид возраст число от 0 до 1, пол 1 или 0, а класс в список (пример 2ой класс должен
# быть в виде [0, 1, 0])

encoders = {
    'Age': lambda age: [age / max_age],
    'Sex': lambda gen: {'male': [0], 'female': [1]}.get(gen),
    'Pclass': lambda pclass: {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}.get(pclass),
    'Survived': lambda x: [x]
}


def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column] = values
    return result


def make_supervised(df):
    raw_input_data = data_frame[input_name]
    raw_output_data = data_frame[output_name]
    return {
        'inputs': dataframe_to_dict(raw_input_data),
        'outputs': dataframe_to_dict(raw_output_data)
    }


def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for elem in vector_raw:
            for i in elem:
                vector.append(i)
        formatted.append(vector)
    return formatted


supervised = make_supervised(data_frame)
encoded_inputs = np.array(encode(supervised['inputs']))  # [[0.22, 0, 0, 0, 1], [0.38, 1, 1, 0, 0]....
encoded_outputs = np.array(encode(supervised['outputs']))  # [[0], [1] ....

train_x = encoded_inputs[:700]
train_y = encoded_outputs[:700]

test_x = encoded_inputs[700:]
test_y = encoded_outputs[700:]

model = keras.Sequential()
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mse',optimizer=Adam(0.001), metrics=['accuracy'])
log = model.fit(train_x, train_y, epochs=50, validation_split=0.2)



plt.title('Losses Train/Validation')
plt.plot(log.history['loss'], label='Train')
plt.plot(log.history['val_loss'], label='Validation')
plt.legend()
plt.show()

plt.title('Accuracy Train/Validation')
plt.plot(log.history['accuracy'], label='Train')
plt.plot(log.history['val_accuracy'], label='Validation')
plt.legend()
plt.show()

# строим датафрэйм с тестовой выборкой и выводим результат
predict_test = model.predict(test_x)
real_data = data_frame[700:][input_name + output_name]
real_data['Predict_survived'] = predict_test
print(real_data.to_string())

# можно подставить свои данные
you_age = int(input('Введите ваш возраст (целое число):')) / max_age
you_sex = {'male': [0], 'female': [1]}.get(input('Введите ваш пол male/female:'))
you_class = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}.get(int(input('Введите класс от 1 до 3:')))

you_vector = []
you_vector.append(you_age)
you_vector.append(*you_sex)
you_vector += you_class
you_predict = model.predict([you_vector])[0][0] * 100

print(f'Вероятность вашего выживания {round(you_predict, 2)}%')
