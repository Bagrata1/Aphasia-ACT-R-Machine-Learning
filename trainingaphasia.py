#Training model for aphasia/ACT-R generated data
#Load libraries
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Store the data set
df = pd.read_csv('combined.csv')
#Look at first 7 rows of data
df.head(7)

#Show the shape (number of rows & columns)
df.shape

#Checking for duplicates and removing them
df.drop_duplicates(inplace = True)

#Show the shape to see if any rows were dropped 
df.shape

#Convert the data into an array
dataset = df.values
dataset


# Get all of the rows from the first eight columns of the dataset
X = dataset[:,0:4] 
# Get all of the rows from the last column
y = dataset[:,4]

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3, random_state = 4)

model = Sequential([
    Dense(6, activation='relu', input_shape=( 4 ,)),
    Dense(2, activation='relu'),
    Dense(1, activation='sigmoid')
]) 

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, y_train,
          batch_size=57, epochs=100, validation_split=0.3)

#visualize the training loss and the validation loss to see if the model is overfitting
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


#visualize the training accuracy and the validation accuracy to see if the model is overfitting
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#Make a prediction & print the actual values
prediction = model.predict(X_test)
prediction  = [1 if y>=0.5 else 0 for y in prediction] #Threshold
print(prediction)
print(y_test)

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = model.predict(X_train)
pred  = [1 if y>=0.5 else 0 for y in pred] #Threshold
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))
print()



