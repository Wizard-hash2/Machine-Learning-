import cv2
# OpenCV library for computer vision.
import matplotlib.pyplot as plt  # For plotting images using matplotlib.

# Load a sample image from disk.
# Replace 'sample.jpg' with the path to an actual image file.
image = cv2.imread('AVALANCHE2.jpg')
# imread loads the image in BGR (Blue, Green, Red) format.

# Check if the image was correctly loaded.
if image is None:
    print("Error: Unable to load image file.")
else:
    # Convert image to grayscale.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cvtColor converts the image from BGR to Grayscale.

    # Apply Canny edge detection.
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    # Canny detects edges by finding areas of strong intensity gradients.

    # Display the original and edge-detected images side by side.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )  # Convert BGR to RGB for correct color display.
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')

    plt.show()