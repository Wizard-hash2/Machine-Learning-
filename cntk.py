# 1. Import the required libraries.
import cntk as C          # CNTK is the main deep learning library.
import numpy as np         # NumPy is used for numerical operations.

# 2. Prepare the dataset.
# Here we define the XOR dataset.
# 'data' contains the input samples (2 features per example).
# 'labels' contains the corresponding binary labels.
data = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]], dtype=np.float32)
labels = np.array([[0],
                   [1],
                   [1],
                   [0]], dtype=np.float32)

# 3. Define the input and label variables.
# 'input_var' is a placeholder for the input data with 2 features.
# 'label_var' is a placeholder for the corresponding labels (scalar per sample).
input_var = C.input_variable(2)
label_var = C.input_variable(1)

# 4. Build the neural network model.
# Here we define a simple feedforward network with one hidden layer.
def create_model(input_var):
    # Dense layer with 4 units and sigmoid activation for hidden processing.
    hidden = C.layers.Dense(4, activation=C.sigmoid)(input_var)
    # Output Dense layer with 1 unit and sigmoid activation to produce a probability.
    output = C.layers.Dense(1, activation=C.sigmoid)(hidden)
    return output

# Create an instance of the model.
model = create_model(input_var)

# 5. Define the loss function and evaluation metric.
# For binary classification, binary cross-entropy loss is appropriate.
loss = C.binary_cross_entropy(model, label_var)
# Classification error computes the rate of misclassification.
error = C.classification_error(model, label_var)

# 6. Choose an optimization algorithm (learner) and initialize the trainer.
# We use stochastic gradient descent (SGD) with a fixed learning rate.
learning_rate = 0.1
learner = C.sgd(model.parameters, lr=learning_rate)
# Trainer ties the model, loss, error, and learner together.
trainer = C.Trainer(model, (loss, error), [learner])

# 7. Train the network.
# Here we run the training loop for a fixed number of epochs (iterations over the data).
num_epochs = 2000
for epoch in range(num_epochs):
    # Train on the entire dataset in each epoch.
    trainer.train_minibatch({input_var: data, label_var: labels})
    
    # Every 500 epochs, print the current loss and error for monitoring.
    if epoch % 500 == 0:
        train_loss = trainer.previous_minibatch_loss_average
        train_error = trainer.previous_minibatch_evaluation_average
        print("Epoch %d: Loss = %.4f, Error = %.4f" % (epoch, train_loss, train_error))

# 8. Test the trained model on the training samples.
# Since the XOR dataset is very simple, we use the same data for testing.
for i in range(len(data)):
    # Evaluate the model for a single sample.
    output = model.eval({input_var: [data[i]]})
    # The output is a probability, so we threshold at 0.5 to get a binary prediction.
    predicted = int(output[0][0] > 0.5)
    print("Input:", data[i], " Expected:", int(labels[i][0]), " Predicted:", predicted)
