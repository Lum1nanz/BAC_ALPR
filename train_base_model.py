import numpy as np
from sklearn.model_selection import train_test_split
from helper_functions import load_data
import models
from tensorflow.keras.optimizers import Adam

# this module trains the R-CNN, so there is a model created that can be used to detect licence plates via the R-CNN

def main():
    X,y = load_data('use_data_with_eu')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) #use fit so will use validation_split

    model = models.get_model_1(input_shape=(128,128,3))
    optimizer=Adam(lr=.0005)
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    history = model.fit(X_train,y_train,validation_split=0.15,epochs=15)

    print(model.evaluate(X_test,y_test))
    model.save('model_with_eu_15epochs.h5')


main()
