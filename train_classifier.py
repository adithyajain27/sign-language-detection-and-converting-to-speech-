'''import pickle
# import sklearn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)
#0.3 means 30% for test and 70% for train

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))
print(f" {score*100}% of samples were classified correctly!")

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()




#-------------------------------------------------------------------------------------------------------------------------


'''
'''
# import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load your data (similar to your previous code)
# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# # Create a CNN model
# model = keras.Sequential([
#     # Convolutional layers with pooling
#     # keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, num_channels)),
#     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 3)),
#     # For example, if you are working with 100x100-pixel RGB images, you would set image_width to 100, image_height to 100, and num_channels to 3. If you're working with grayscale images, num_channels would be 1.
    
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
    
#     # Flatten the output for the fully connected layers
#     keras.layers.Flatten(),
    
#     # Fully connected layers
#     keras.layers.Dense(128, activation='relu'),
#     # keras.layers.Dense(number_of_classes, activation='softmax') 
#     keras.layers.Dense(4, activation='softmax') 
    
#     # . For example, if you're working with an image classification problem where you need to distinguish between different types of animals (e.g., cats, dogs, birds, etc.), number_of_classes would be set to the total number of animal classes, such as 3 for the given example.
#     # You should replace the number_of_classes variable with the actual number of classes in your specific dataset.
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Evaluate the model on the test data
# y_predict = model.predict(x_test)
# y_predict_labels = np.argmax(y_predict, axis=1)

# score = accuracy_score(y_test, y_predict_labels)
# print(f"{score*100}% of samples were classified correctly!")

# # Save the model
# model.save('cnn_model.h5')







#---------------------------------------------------------------------------------------------
# from sklearn.base import accuracy_score
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # from sklearn import *

# # Load your data (similar to your previous code)
# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# # Create a CNN model
# model = keras.Sequential([
#   # Convolutional layers with pooling
#   keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 3)),
#   keras.layers.MaxPooling2D((2, 2)),
#   keras.layers.Conv2D(64, (3, 3), activation='relu'),
#   keras.layers.MaxPooling2D((2, 2)),

#   # Flatten the output for the fully connected layers
#   keras.layers.Flatten(),

#   # Fully connected layers
#   keras.layers.Dense(128, activation='relu'),
#   keras.layers.Dense(4, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Evaluate the model on the test data
# y_predict = model.predict(x_test)
# y_predict_labels = np.argmax(y_predict, axis=1)

# score = accuracy_score(y_test, y_predict_labels)
# print(f"{score*100}% of samples were classified correctly!")

# # Save the model
# model.save('cnn_model.h5')

'''
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Ensure all elements in 'data' are treated as strings
data = [str(doc) if not isinstance(doc, str) else doc for doc in data]

# Create a TfidfVectorizer and transform your data
vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f"{score * 100}% of samples were classified correctly!")

f = open('model.p', 'wb')
pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
f.close()
