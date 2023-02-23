from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Define the model architecture
model = Sequential()
model.add(SimpleRNN(32, input_shape=(None, 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size=32)

# Save the model
model.save('my_rnn_model.h5')

#
# In this example, we are building a simple RNN model with one layer of 32 units.
# The input shape is set to (None, 1) which means the model can take in sequences of
# any length with one feature. We then add a dense layer with one output unit and a sigmoid
# activation function to produce a binary classification output.
#
# We compile the model with binary crossentropy loss, the Adam optimizer, and accuracy as
# the evaluation metric. We then train the model on the training data for
# 10 epochs with a batch size of 32, and validate on the validation data.
#
# Finally, we evaluate the model on the test data and save the model to a file named 'my_rnn_model.h5'.