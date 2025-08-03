import cv2, os, heaps
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

class colorSegmenter:

    def __init__(self, dataset_dir):
        """ Initialize the segmenter with predefined colors and sizes."""
        self.colors = {
            'Red': redSegment(),
            'Green': greenSegment(),
            'Blue': blueSegment()
        }

        self.dataset_dir = dataset_dir

    def segment(self, image_path, plot = False, overwrite = False):
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

        # Create variable for storing corners
        corners = []

        # initialize a median heap
        for color, segment in self.colors.items():# Iterate through each color
            if os.path.exists(f"{self.dataset_dir}{segment.output_dir}{image_filename}"):
                FileExistsError(f"File {image_filename} for {segment.color.lower()} segment exists")
                if overwrite == False:
                    print(f"Overwriting set to False, skipping image {image_filename}...")
                    continue

            print(f"Image: {image.shape}")
            color_mask = cv2.inRange(image, segment.lower_color, segment.upper_color) # Color mask
            color_count = cv2.countNonZero(color_mask)
            print(f"{color} count - {color_count}")
            if color_count < 25000:
                print(f"Color segmentation for {color} failed, skipping image {image_filename}...")
                return None

            if plot:# PLot mask
                cv2.imshow("Mask", color_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            data_space = self.find_corners(color_mask, color, segment, corners, plot) # Find edges
            print(f"Segment corners detected: {data_space}")
            if data_space is None:
                return None

            # Add data space to corners
            corners.append(data_space)

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
            transformation = cv2.getPerspectiveTransform(data_space, output_space)

            # Warp image
            warped = cv2.warpPerspective(image, transformation, (width, height))
            if plot: # Plot the warped image
                cv2.imwrite("extracted_frame.jpg", warped)
                cv2.imshow("Extracted", warped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Do color specific post-processing
            processed_image = segment.process(warped, image_filename, self.dataset_dir)
            if plot: # Plot the processed image
                cv2.imshow(f"Processed {color}", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            print(f"Processed {color} segment.")

            segment_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # Create a black mask
            cv2.fillPoly(segment_mask, [data_space.astype(np.int32)], 255)  # Fill the polygon with white (255)

            # Apply the mask to a copy of the original image
            image[segment_mask == 255] = [255, 255, 255]  # Set region to white
        return True
    
    def find_corners(self, mask, color, segment, corners, plot):
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

        if color == 'Blue':
            closest_points = self.estimate_corners(corners)
            print(f"Estimated core space...\n {closest_points}")
        
        else:
            heap = segment.heap(content=[]) # Initialize a min heap to store the points

            # Threshold to get binary image
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort by area, take the largest
            contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if approx is None or len(approx) == 0:
                print("Skipping: No approximated contour.")
                return None  # or return, pass, etc.

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
                heap.insert(distance)
            print(heap)

            if len(heap.heap) < 4: # If we didn't find 4 corners
                return None

            # Extract the four corners based on the closest distances
            closest_points = np.zeros((4, 2), dtype = np.float32)  # Initialize a list to hold the closest points for each corner
            while closest_points[0, 0] == 0 or closest_points[1, 0] == 0 or closest_points[2, 0] == 0 or closest_points[3, 0] == 0:
                if heap.size() == 0:
                    print(f"No more points left to process for color {color}, failed to find 4 corners in seperate quadrants")
                    break
                
                distance = heap.extract()
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
                
                
        print(f"Closest points: {closest_points}")

        if plot:
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

    def estimate_corners(self, corners):

        # Estimate bottom left point using red space
        p1 = corners[0][3]
        p2 = corners[0][0]

        p3 = p1 + 2 * (p2 - p1)

        # Estimate bottom right
        p1 = corners[0][2]
        p2 = corners[0][1]
        p4 = p1 + 2 * (p2 - p1)

        # Use bottom green corners for top of blue
        p1 = corners[1][0]
        p2 = corners[1][1]

        estimated_space = np.array([
            p3, # Top right
            p4, # Top left
            p2, # Bottom left
            p1 # Bottom right
            ], dtype = np.float32)
        return estimated_space
    
class redSegment:
    """
    Segmenter for red color segments.
    """
    
    def __init__(self):
        self.color = 'Red'
        self.size = (400, 500)
        self.lower_color = np.array([0, 0, 30])
        self.upper_color = np.array([59, 43, 195])

        self.heap = heaps.MinHeap

        self.output_dir = "preprocessed/profiles/"

    def process(self, image, image_filename, dataset_dir):
        """
        Process the red segment. This method can be extended to include 
        specific processing for red segments. Segment the digits in the image.
        
        Args:
            image (np.ndarray): The input image.
        Returns:
            list: A list of segmented digits.
        """
        self.save(image, image_filename, dataset_dir)  # Save the image for further processing
        return image
    
    def save(self, image, image_filename, dataset_dir):
        """
        Save the red segment image for further processing.
        
        Args:
            image (np.ndarray): The input image.
            image_filename (str): The name of the original image file.
        """
        image_filename = f"{dataset_dir}{self.output_dir}{image_filename}" # Create a new filename for the red segment
        cv2.imwrite(image_filename, image)
        print(f"Red segment saved in {image_filename}")

class greenSegment:
    """
    Segmenter for green color segments.
    """
    
    def __init__(self):
        self.color = 'Green'
        self.size = (200, 500)
        self.lower_color = np.array([10, 10, 0])
        self.upper_color = np.array([75, 75, 58])

        self.heap = heaps.MinHeap

        self.output_dir = "preprocessed/written_labels/"
    
    def process(self, image, image_filename, dataset_dir):
        """
        Process the green segment analyzing the hand-written label segment
        using the labeler and saving the image with a name to help label
        relavent images.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        thresh = cv2.adaptiveThreshold(enhanced, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 
                               11, 2)

        self.save(image, image_filename, dataset_dir)  # Save the image for further processing
        
        return image

    def save(self, image, image_filename, dataset_dir):
        """
        Save the green segment image for further processing.
        
        Args:
            image (np.ndarray): The input image.
            image_filename (str): The name of the original image file.
        """
        image_filename = f"{dataset_dir}{self.output_dir}{image_filename}" # Create a new filename for the red segment
        cv2.imwrite(image_filename, image)
        print(f"Green segment saved in {image_filename}")

class blueSegment:
    """
    Segmenter for blue color segments.
    """
    
    def __init__(self):
        self.color = 'Blue'
        self.size = (300, 500)
        self.lower_color = np.array([65, 5, 2])
        self.upper_color = np.array([160, 90, 36])

        self.heap = heaps.MaxHeap

        self.output_dir = "preprocessed/cores/"

    def process(self, image, image_filename, dataset_dir):
        """
        Process the blue segment. This method can be extended to include 
        specific processing for blue segments. Segment the digits in the image.
        
        Args:
            image (np.ndarray): The input image.
            image_filename (str): The name of the original image file.
        Returns:
            None
        """
        self.save(image, image_filename, dataset_dir)  # Save the image for further processing
        return image

    def save(self, image, image_filename, dataset_dir):
        """
        Save the blue segment image for further processing.
        
        Args:
            image (np.ndarray): The input image.
            image_filename (str): The name of the original image file.
            data_directory (str): Directory to save the image.
        """
        image_filename = f"{dataset_dir}{self.output_dir}{image_filename}" # Create a new filename for the blue segment
        cv2.imwrite(image_filename, image)
        print(f"Blue segment saved in {image_filename}")
