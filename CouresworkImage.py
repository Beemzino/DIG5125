import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image_and_histogram(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:  # If the image fails to load, display an error and exit
        print(f"Failed to load the image: {image_path}")
        return

    # Convert the image from BGR (OpenCV default) to RGB for display using matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prompt the user to choose a histogram type
    print("\nChoose histogram type:")
    print("1. Luminance Histogram")  # Grayscale intensity histogram
    print("2. RGB Histogram")       # Separate histograms for Red, Green, and Blue channels
    print("3. Apply Histogram Normalization")  # Enhance contrast and display luminance histogram
    choice = input("Enter your choice (1/2/3): ").strip()  # Get user's choice as a string

    # Prepare a matplotlib figure to display the image and histogram side-by-side
    plt.figure(figsize=(12, 6))
    
    if choice == "1":  # Luminance Histogram
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Display the grayscale image
        plt.subplot(1, 2, 1)  # Create the first subplot for the image
        plt.imshow(gray, cmap='gray')  # Show the grayscale image
        plt.title("Grayscale Image")  # Add a title
        plt.axis('off')  # Remove axes for better visualization
        
        # Plot the grayscale intensity histogram
        plt.subplot(1, 2, 2)  # Create the second subplot for the histogram
        plt.hist(gray.ravel(), bins=256, range=(0, 256), color='black')  # Flatten image and calculate histogram
        plt.title("Luminance Histogram")  # Add a title
        plt.xlabel("Pixel Intensity")  # Label the x-axis
        plt.ylabel("Frequency")  # Label the y-axis

    elif choice == "2":  # RGB Histogram
        # Display the original image
        plt.subplot(1, 2, 1)  # Create the first subplot for the image
        plt.imshow(image_rgb)  # Show the image in RGB format
        plt.title("Original Image")  # Add a title
        plt.axis('off')  # Remove axes for better visualization
        
        # Plot the RGB channel histograms
        plt.subplot(1, 2, 2)  # Create the second subplot for the histograms
        colors = ['red', 'green', 'blue']  # Define the colors for each channel
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])  # Calculate histogram for each channel
            plt.plot(hist, color=color)  # Plot the histogram with the corresponding color
        plt.title("RGB Histogram")  # Add a title
        plt.xlabel("Pixel Intensity")  # Label the x-axis
        plt.ylabel("Frequency")  # Label the y-axis
        plt.legend(colors)  # Add a legend to distinguish channels

    elif choice == "3":  # Histogram Normalization
        # Apply normalization to enhance the image's contrast
        normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized_image_rgb = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        
        # Display the normalized image
        plt.subplot(1, 2, 1)  # Create the first subplot for the image
        plt.imshow(normalized_image_rgb)  # Show the normalized image
        plt.title("Normalized Image")  # Add a title
        plt.axis('off')  # Remove axes for better visualization
        
        # Convert normalized image to grayscale and plot the luminance histogram
        gray_normalized = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
        plt.subplot(1, 2, 2)  # Create the second subplot for the histogram
        plt.hist(gray_normalized.ravel(), bins=256, range=(0, 256), color='black')  # Calculate histogram
        plt.title("Luminance Histogram (Normalized)")  # Add a title
        plt.xlabel("Pixel Intensity")  # Label the x-axis
        plt.ylabel("Frequency")  # Label the y-axis

    else:
        # Handle invalid user input
        print("Invalid choice.")
        return

    # Adjust layout and display the plots
    plt.tight_layout()  # Adjust spacing between subplots for better visibility
    plt.show()  # Display the figure

# Specify the path to the image file
if __name__ == "__main__":
    image_path = "mario.jpg"  # Replace with the actual path to mario.jpg
    display_image_and_histogram(image_path)