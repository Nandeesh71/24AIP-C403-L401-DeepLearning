

# CONVERTED: tf.keras.datasets.mnist -> torchvision.datasets.MNIST
# transforms.ToTensor() already scales pixel values to [0, 1],
# which is the same normalization the original code did manually with x / 255.0


# CONVERTED: validation_split=0.2 in model.fit() -> manual 80/20 split with random_split



# CONVERTED: Sequential([Flatten, Dense(64, relu), Dense(10, softmax)]) -> nn.Module
# OVERFITTING HANDLING (kept simple, as requested):
#   - Dropout(0.3) after the hidden layer: randomly zeroes 30% of activations
#     during training so the network can't over-rely on specific neurons.
#     This is the same Dropout the original script imported but never used.
#   - weight_decay=1e-4 on the optimizer (L2 regularization): penalizes large
#     weights, which is a standard, low-complexity way to reduce overfitting.
# Softmax is left out of the model itself because nn.CrossEntropyLoss applies
# log-softmax internally (this is the standard PyTorch equivalent of Keras'
# 'sparse_categorical_crossentropy' + softmax output).





# CONVERTED: model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
 # weight_decay = L2 regularization





# CONVERTED: model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)




# CONVERTED: model.evaluate(x_test, y_test)



