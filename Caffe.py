# --------------------------------------------------------------------
# 1. Import necessary libraries.
# --------------------------------------------------------------------
import caffe                 # Main Caffe library providing access to the Caffe framework.
import numpy as np           # NumPy for numerical operations.
import matplotlib.pyplot as plt  # Optional: used to display the image.

# --------------------------------------------------------------------
# 2. Set Caffe to run in CPU or GPU mode.
#    For demonstration, we'll use CPU mode; to use GPU mode, call caffe.set_mode_gpu() 
#    and set the appropriate device ID.
# --------------------------------------------------------------------
caffe.set_mode_cpu()

# --------------------------------------------------------------------
# 3. Load the pre-trained network.
#
# Define the paths to the model definition and the trained weights.
# Replace 'path/to/...' with the actual paths on your system.
# - The deploy.prototxt file defines the network architecture in test (inference) mode.
# - The caffemodel file contains the trained parameters from the training process.
# --------------------------------------------------------------------
model_def = 'path/to/deploy.prototxt'
model_weights = 'path/to/lenet_iter_10000.caffemodel'

# Load the net in "test" phase so that dropout layers, if any, behave in inference mode.
net = caffe.Net(model_def,      # Network architecture.
                model_weights,  # Pre-trained weights.
                caffe.TEST)     # Set mode to test/inference.

# --------------------------------------------------------------------
# 4. Set up the preprocessing transformer for the input data.
#
# The network expects input data in a specific shape and format.
# Here, an instance of caffe.io.Transformer is created to:
#   - Rearrange the dimensions to [channels, height, width].
#   - Scale raw pixel values to the expected range.
#   - Optionally swap channels if necessary (e.g., RGB to BGR).
# For MNIST, the images are single channel (grayscale), but the transformer still handles the shape properly.
# --------------------------------------------------------------------
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# The network typically expects the color channel as the first dimension;
# set_transpose changes the order from (height, width, channels) to (channels, height, width).
transformer.set_transpose('data', (2, 0, 1))
# Raw scale - if the network was trained with pixel values scaled up (e.g., 0-255), ensure consistency.
transformer.set_raw_scale('data', 255)
# For MNIST (grayscale), there is no channel swap needed, but if using color images uncomment:
# transformer.set_channel_swap('data', (2, 1, 0))

# --------------------------------------------------------------------
# 5. Load an example image.
#
# For demonstration, we assume you have a sample MNIST digit stored as an image file (e.g., PNG format).
# Replace 'path/to/mnist_image.png' with the path to an MNIST image.
# Note: Caffe's image loader (caffe.io.load_image) expects a color image by default.
# Since MNIST is grayscale, we pass color=False.
# --------------------------------------------------------------------
image_path = 'path/to/mnist_image.png'
input_image = caffe.io.load_image(image_path, color=False)

# --------------------------------------------------------------------
# 6. Preprocess the image and load it into the network.
#
# The network's first layer (named 'data') expects a batch of images.
# Here we preprocess the single image using the transformer created earlier,
# then assign it to the net's data blob.
# --------------------------------------------------------------------
net.blobs['data'].data[...] = transformer.preprocess('data', input_image)

# --------------------------------------------------------------------
# 7. Perform a forward pass.
#
# This computes the outputs of the network given the input image.
# The output is a dictionary where the key 'prob' (for probability) holds the classification results.
# --------------------------------------------------------------------
output = net.forward()

# --------------------------------------------------------------------
# 8. Extract and display the prediction.
#
# The output dictionary has a key 'prob' which is typically an array of probabilities for each class.
# The predicted class is the index with the maximum probability.
# --------------------------------------------------------------------
predicted_class = output['prob'][0].argmax()
print("Predicted class:", predicted_class)

# --------------------------------------------------------------------
# 9. (Optional) Display the input image.
#
# Use matplotlib to display the MNIST image.
# This can help verify the input to the network.
# --------------------------------------------------------------------
plt.imshow(input_image, cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.show()
