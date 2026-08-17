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
            # Check for the original filename or a renamed (labeled) variant like image_N_site_col_core.png
            stem = image_filename.split('.png')[0]
            if glob(f"{self.dataset_dir}{segment.output_dir}{stem}.*") or glob(f"{self.dataset_dir}{segment.output_dir}{stem}_*"):
                if overwrite == False:
                    print(f"Overwriting set to False, skipping image {image_filename}...")
                    continue

            print(f"Image: {image.shape}")

            if color == 'Green':
                # The green frame is dark and weakly saturated — its apparent color swings from
                # teal (shade) to olive (direct sun) and snow breaks its outline, so detecting it
                # by color is unreliable. The red frame is vivid on every card; derive the label
                # space from its corners instead.
                data_space = self.estimate_label_space(corners[0], image)
                print(f"Green corners estimated from red frame geometry:\n {data_space}")
            else:
                # Build the color mask in HSV space (hue isolates each frame color far more
                # reliably than BGR boxes; red needs two ranges because its hue wraps around 0)
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for lower_color, upper_color in segment.hsv_ranges:
                    color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower_color, upper_color))
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

            # Close small breaks (snow on the frame, glare) so the ring reads as connected;
            # kept small so nearby red features (the strip's red edge-line) don't merge in
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Threshold to get binary image
            _, thresh = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)

            # Find contours with hierarchy: the frame is a hollow ring, so prefer the outer
            # contour enclosing the largest hole — a solid red blob elsewhere in the photo
            # (clothing, gear) can out-area the frame but has no hole
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None or len(contours) == 0:
                print("Skipping: No contours found.")
                return None
            chosen, best_hole = None, 0
            for ind in range(len(contours)):
                if hierarchy[0][ind][3] != -1:
                    continue # Only consider outer contours, not the holes themselves
                hole_area = 0
                child = hierarchy[0][ind][2]
                while child != -1:
                    hole_area = max(hole_area, cv2.contourArea(contours[child]))
                    child = hierarchy[0][child][0]
                if hole_area > best_hole:
                    chosen, best_hole = ind, hole_area

            hole_thickness = None
            if chosen is not None and best_hole >= 25000:
                # Take corners from the ring's HOLE (the grid interior): red features touching
                # the frame's outer edge (the strip's red edge-line, gear) can drag outer-hull
                # corners away, but nothing can touch the hole. Corners get pushed back out by
                # the frame thickness afterwards.
                contour = contours[chosen]
                hole, hole_area = None, 0
                child = hierarchy[0][chosen][2]
                while child != -1:
                    if cv2.contourArea(contours[child]) > hole_area:
                        hole, hole_area = child, cv2.contourArea(contours[child])
                    child = hierarchy[0][child][0]
                _, _, ow, oh = cv2.boundingRect(contour)
                _, _, hw, hh = cv2.boundingRect(contours[hole])
                # Attached slivers inflate one outer dimension, so trust the smaller estimate
                hole_thickness = min((ow - hw) / 2, (oh - hh) / 2)
                approx = cv2.convexHull(contours[hole])
            else:
                # No ring-like contour: fall back to the largest by area, merging back any
                # frame fragments a break split off, and take corners from the outer hull
                chosen = max(range(len(contours)), key=lambda ind: cv2.contourArea(contours[ind]))
                contour = contours[chosen]
                x, y, w, h = cv2.boundingRect(contour)
                margin_x, margin_y = int(0.05 * w), int(0.05 * h)
                pieces = [contour.reshape(-1, 2)]
                for ind in range(len(contours)):
                    if ind == chosen or hierarchy[0][ind][3] != -1 or cv2.contourArea(contours[ind]) < 2000:
                        continue
                    cx, cy, cw, ch = cv2.boundingRect(contours[ind])
                    if cx > x - margin_x and cy > y - margin_y and cx + cw < x + w + margin_x and cy + ch < y + h + margin_y:
                        pieces.append(contours[ind].reshape(-1, 2))
                approx = cv2.convexHull(np.vstack(pieces))

            if approx is None or len(approx) < 4:
                print("Skipping: No usable frame outline.")
                return None

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

            # A corner left at its zero placeholder means the search exhausted the heap
            # without covering all four quadrants — fail instead of warping with garbage
            if any(closest_points[row, 0] == 0 for row in range(4)):
                print(f"Failed to find 4 valid corners for color {color}, aborting segmentation")
                return None

            if hole_thickness is not None:
                # Corners came from the hole — push them outward to the frame's outer edge
                for row in range(4):
                    closest_points[row, 0] += hole_thickness if closest_points[row, 0] > x_avg else -hole_thickness
                    closest_points[row, 1] += hole_thickness if closest_points[row, 1] > y_avg else -hole_thickness

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

    def estimate_label_space(self, red_corners, image):
        """
        Estimate the green label space from the red frame corners. The label frame
        sits directly below the red box, but its extent varies (~0.36-0.49 red
        heights across cards), so the bottom edge is refined by walking down each
        side and stopping just above where the blue board begins. If no blue is
        found (board fully snow-covered) fall back to a generous 0.45.

        Corner row order matches find_corners: 0 lower-left, 1 lower-right,
        2 upper-right, 3 upper-left.
        """
        left_down = red_corners[0] - red_corners[3]   # red left edge, top to bottom
        right_down = red_corners[1] - red_corners[2]  # red right edge, top to bottom

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        steps = np.linspace(0.08, 0.55, 48)
        step_size = steps[1] - steps[0]

        def board_onset(origin, down):
            """Walk down the frame's side bar until the board begins: either blue,
            or — when snow hides the blue — a sustained bright (snow/card) run once
            we are deep enough that it cannot be the label strip itself."""
            run_blue = run_bright = 0
            for t in steps:
                x, y = (origin + t * down).astype(int)
                if x < 4 or y < 4 or x >= width - 4 or y >= height - 4:
                    break
                patch = hsv[y-4:y+5, x-4:x+5].reshape(-1, 3)
                med = np.median(patch, axis=0)
                if 104 <= med[0] <= 133 and med[1] > 60 and med[2] > 40:
                    run_blue += 1
                    if run_blue >= 3: # Sustained blue - back off to just above where it started
                        return t - 3 * step_size - 0.02
                else:
                    run_blue = 0
                if med[2] > 190 and med[1] < 50:
                    run_bright += 1
                    if run_bright >= 5 and t - 5 * step_size >= 0.28:
                        return t - 5 * step_size - 0.01
                else:
                    run_bright = 0
            return None # Board not visible on this side

        onsets = [board_onset(red_corners[0], left_down), board_onset(red_corners[1], right_down)]
        found = [t for t in onsets if t is not None]
        # Snow often hides the board on one side — borrow the measured side's depth
        # rather than defaulting deep into the snow
        default = float(np.median(found)) if found else 0.45
        t_left = float(np.clip(onsets[0] if onsets[0] is not None else default, 0.28, 0.5))
        t_right = float(np.clip(onsets[1] if onsets[1] is not None else default, 0.28, 0.5))

        estimated_space = np.array([
            red_corners[0] + t_left * left_down,    # lower left
            red_corners[1] + t_right * right_down,  # lower right
            red_corners[1],                         # upper right = red lower right
            red_corners[0],                         # upper left = red lower left
        ], dtype=np.float32)
        return estimated_space

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
        # Red hue wraps around 0, so two HSV ranges (measured H ~0-8 and ~160-179 on cards)
        self.hsv_ranges = [
            (np.array([0, 90, 50]), np.array([8, 255, 255])),
            (np.array([160, 90, 50]), np.array([180, 255, 255])),
        ]

        # Corners are the FARTHEST points from the contour centroid in each quadrant;
        # a min heap let extra mid-edge vertices from approxPolyDP win over true corners
        self.heap = heaps.MaxHeap

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
        # Dark teal-green frame: hue sits tightly at ~87-93, well below the grid/board
        # blues at ~106-129; saturation floor rejects snow, shadow and the white strip
        self.hsv_ranges = [
            (np.array([82, 80, 25]), np.array([101, 255, 170])),
        ]

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
        # Blue board (H ~113-129); the navy grid falls in range too, which is fine —
        # blue corners are estimated from red/green and this mask only feeds the count check
        self.hsv_ranges = [
            (np.array([104, 40, 40]), np.array([133, 255, 220])),
        ]

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
