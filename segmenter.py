import cv2, heaps, labeler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from labeler import Labeler


class colorSegmenter:

    def __init__(self):
        """ Initialize the segmenter with predefined colors and sizes."""
        self.colors = {
            'Red': redSegment(),
            'Green': greenSegment(),
            'Blue': blueSegment()
        }

        self.labeler = Labeler() # Initialize the labeler

    def segment(self, image_path, plot = False):
        """
        Segment the image based on predefined colors and sizes.
        Args:
            image_path (str): Path to the input image.
            plot (bool): Whether to plot the segmented images.
            
        Returns:
            None
        """
        print(f"Segmenting image: {image_path}")
        image_filename = image_path.split("/")[-1] # Get the image filename
        image = cv2.imread(image_path) # Read the image

        # initialize a median heap
        for color, segment in zip(self.colors.keys(), self.colors.values()):# Iterate through each color
            color_mask = cv2.inRange(image, segment.lower_color, segment.upper_color) # Color mask

            if plot:# PLot mask
                cv2.imshow("Mask", color_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            datum_space = self.find_corners(color_mask, color) # Find edges
            print(f"Segment corners detected: {datum_space}")

            # Define the size of the image
            height, width = segment.size

            # Define the points in the datum space
            output_space = np.array([
                [0, height - 1],  # Top left
                [width - 1, height - 1], # Top right
                [width - 1, 0], # Bottom right
                [0, 0] # Bottom left
            ], dtype=np.float32)
            
            # Define the transformation matrix
            transformation = cv2.getPerspectiveTransform(datum_space, output_space)

            # Warp image
            warped = cv2.warpPerspective(image, transformation, (width, height))
            if plot: # Plot the warped image
                cv2.imwrite("extracted_frame.jpg", warped)
                cv2.imshow("Extracted", warped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Do color specific post-processing
            processed_image = segment.process(warped, image_filename)
            if plot: # Plot the processed image
                cv2.imshow(f"Processed {color}", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            print(f"Processed {color} segment.")
        return
    
    def find_corners(self, mask, color):
        """
        Find the corners of a square in the image based on the mask.
        
        Args:
            image (np.ndarray): The input image.
            mask (np.ndarray): The binary mask of the square.
            color (str): The color of the square.
            tunnel (bool): Whether to use tunnel detection logic.
        Returns:
            list: A list of points representing the corners of the square.
        """
        min_heap = heaps.MinHeap(content=[]) # Initialize a min heap to store the points

        # Threshold to get binary image
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area, take the largest
        contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Find the centroid
        x_avg = 0
        y_avg = 0
        for point in approx:
            x_avg += point[0][0]
            y_avg += point[0][1]
        
        x_avg = int(x_avg / len(approx))
        y_avg = int(y_avg / len(approx))
        
        # Calculate distances from the centroid to each point
        points = {}
        for point in approx:
            distance = np.linalg.norm(point - np.array([x_avg, y_avg]))
            points[str(distance)] = [int(point[0, 0]), int(point[0, 1])]
            min_heap.insert(distance)
        print(min_heap)

        # Extract the four corners based on the closest distances
        closest_points = np.zeros((4, 2), dtype = np.float32)  # Initialize a list to hold the closest points for each corner
        while closest_points[0, 0] == 0 or closest_points[1, 0] == 0 or closest_points[2, 0] == 0 or closest_points[3, 0] == 0:
            distance = min_heap.extract_min()
            point = points[str(distance)]
            # If upper left corner
            if point[0] < x_avg and point[1] < y_avg:
                if closest_points[3, 0] != 0:
                    print("More than one point detected in upper left corner")
                    continue
                closest_points[3] = point
            # If upper right corner
            elif point[0] > x_avg and point[1] < y_avg:
                if closest_points[2, 0] != 0:
                    print("More than one point detected in upper right corner")
                    continue
                closest_points[2] = point
            # If lower right corner
            elif point[0] > x_avg and point[1] > y_avg:
                if closest_points[1, 0] != 0:
                    print("More than one point detected in lower right corner")
                    continue
                closest_points[1] = point
            # If lower left corner
            elif point[0] < x_avg and point[1] > y_avg:
                if closest_points[0, 0] != 0:
                    print("More than one point detected in lower left corner")
                    continue
                closest_points[0] = point
            else:
                ValueError("Point does not belong to any corner")
            
            if min_heap.size() == 0:
                print(f"No more points left to process for color {color}.")
                break
        print(f"Closest points: {closest_points}")

        # Draw the corners
        for row in range(closest_points.shape[0]):
            x, y = int(closest_points[row, 0]), int(closest_points[row, 1])
            print(f"Detected corner at ({x}, {y}) for color {color}")
            cv2.circle(mask, (x, y), radius=50, color=(203, 192, 255), thickness=-1)

        # Convert to RGB for matplotlib display
        img_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Show the result
        plt.imshow(img_rgb)
        plt.title("Detected Square Corners (Pink)")
        plt.axis("off")
        plt.show()

        return closest_points

    
class redSegment:
    """
    Segmenter for red color segments.
    """
    
    def __init__(self):
        self.color = 'Red'
        self.size = (400, 500)
        self.lower_color = np.array([0, 0, 30])
        self.upper_color = np.array([45, 35, 140])

    def process(self, image, image_filename, data_directory = None, plot = False):
        """
        Process the red segment. This method can be extended to include 
        specific processing for red segments. Segment the digits in the image.
        
        Args:
            image (np.ndarray): The input image.
        Returns:
            list: A list of segmented digits.
        """
        self.save(image, image_filename, data_directory)  # Save the image for further processing
    
    def save(self, image, image_filename, data_directory = None):
        """
        Save the red segment image for further processing.
        
        Args:
            image (np.ndarray): The input image.
            image_filename (str): The name of the original image file.
        """
        if data_directory is None:
            data_directory = f"/Users/dennyschaedig/Scripts/AvalancheAI/snow-profiles/cores/"
        image_filename = f"{data_directory}{image_filename}" # Create a new filename for the red segment
        cv2.imwrite(f"{image_filename}", image)
        print(f"Red segment saved in {image_filename}")

class greenSegment:
    """
    Segmenter for green color segments.
    """
    
    def __init__(self):
        self.color = 'Green'
        self.size = (200, 500)
        self.lower_color = np.array([20, 15, 0])
        self.upper_color = np.array([50, 60, 52])
    
    def process(self, image, image_filename, data_directory = None, plot = False):
        """
        Process the green segment analyzing the hand-written label segment
        using the labeler and saving the image with a name to help label
        relavent images.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 20 and w > 10:  # Filter small noise
                digit = thresh[y:y+h, x:x+w]
                digit = cv2.resize(digit, (28, 28))  # Match input size of CNN
                predicted_digit = self.classify_digit(digit)  # Classify the digit

                # Show image with label
                if plot:
                    cv2.imshow(f"Digit {predicted_digit}", digit)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                digits.append(digit)
        if not digits:
            print("No digits found in the green segment.")
        else:
            print(f"Found {len(digits)} digits in the green segment - {digits}")
            
        value_split = image_filename.split('.')
        image_filename = value_split[0] + "-".join(digits) + value_split[1]

        image_filename = f"{data_directory}{image_filename}" # Create a new filename for the green segment
        self.save(image, image_filename, data_directory)  # Save the image for further processing

    def classify_digit(self, image):
        """
        Classify a single digit image using a pre-trained model.
        
        Args:
            digit_image (np.ndarray): The segmented digit image.
        Returns:
            int: The predicted digit.
        """
        predicted_digit = self.labeler.predict(image)
        print(f"Predicted digit: {predicted_digit}")
        return predicted_digit

    def save(self, image, image_filename, data_directory = None):
        """
        Save the green segment image for further processing.
        
        Args:
            image (np.ndarray): The input image.
            image_filename (str): The name of the original image file.
        """
        if data_directory is None:
            data_directory = f"/Users/dennyschaedig/Scripts/AvalancheAI/snow-profiles/labels/"
        image_filename = f"{data_directory}{image_filename}" # Create a new filename for the red segment
        cv2.imwrite(image_filename, image)
        print(f"Green segment saved in {image_filename}")

class blueSegment:
    """
    Segmenter for blue color segments.
    """
    
    def __init__(self):
        self.color = 'Blue'
        self.size = (300, 500)
        self.lower_color = np.array([78, 15, 23])
        self.upper_color = np.array([85, 22, 30])

    def process(self, image, image_filename, data_directory = None, plot = False):
        """
        Process the blue segment. This method can be extended to include 
        specific processing for blue segments. Segment the digits in the image.
        
        Args:
            image (np.ndarray): The input image.
            image_filename (str): The name of the original image file.
        Returns:
            None
        """
        self.save(image, image_filename, data_directory)  # Save the image for further processing
    
    def save(self, image, image_filename, data_directory):
        """
        Save the blue segment image for further processing.
        
        Args:
            image (np.ndarray): The input image.
            image_filename (str): The name of the original image file.
            data_directory (str): Directory to save the image.
        """
        if data_directory is None:
            data_directory = f"/Users/dennyschaedig/Scripts/AvalancheAI/snow-profiles/profiles/"
        image_filename = f"{data_directory}{image_filename}" # Create a new filename for the blue segment
        cv2.imwrite(image_filename, image)
        print(f"Blue segment saved in {image_filename}")

if __name__ == "__main__":
    segmenter = colorSegmenter()
    segmenter.segment("../snow-profiles/raw/IMG_2649.JPG")