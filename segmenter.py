import cv2
import numpy as np

class segmenter:

    def __init__(self):
        self.colors = [
            'Red',
            'Green',
            'Blue'
        ]
        self.sizes = [
            (400, 500),
            (200, 500),
            (300, 500),
        ]

        # Red
        self.lower_colors = [
            np.array([160, 100, 70]), # Red
            np.array([36, 50, 20]), # Green
            np.array([86, 100, 70]), # Blue
        ]

        self.upper_colors = [
            np.array([180, 255, 255]), # Red
            np.array([85, 255, 200]), # Green
            np.array([125, 255, 255]), # Blue
        ]

    def segment(self, image_path):
        image = cv2.imread(image_path) # Read the image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for color, size, lower, upper in zip(self.colors, self.sizes, self.lower_colors, self.upper_colors):# Iterate through each color
            color_mask = cv2.inRange(hsv, lower, upper) # Color mask

            # Define the size of the image
            height, width = size

            datum_space = self.find_corners(image, color_mask, color) # Find edges

            # Define the points in the datum space
            output_space = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1] ]# Define output space  
            )     
            
            # Define the transformation matrix
            transformation = cv2.getPerspectiveTransform(datum_space, output_space)

            # Warp image
            warped = cv2.warpPerspective(image, transformation, (width, height))
            cv2.imwrite("extracted_frame.jpg", warped)
            cv2.imshow("Extracted", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return
    
    def find_corners(self, image, mask, color):
        width_midpoint = int(round(mask.shape[0]/2))# Find midpoint
        height_midpoint = int(round(mask.shape[1]/2))# Find midpoint

        pointer = (width_midpoint, 0)# Find top of segment
        while mask[pointer[0], pointer[1]] == 0 and pointer[1] < mask.shape[1]:
            pointer = (pointer[0], pointer[1] + 1)
        
        if color != 'Blue': # Find inside of 
            while mask[pointer[0], pointer[1]] == 1 and pointer[1] < mask.shape[1]:
                pointer = (pointer[0], pointer[1] + 1)
        elif color == 'Blue':
            interest = 0

        # Top right corner   

        # Bottom right corner

        # Bottom left corner

        # Top left corner

        # Format

        return