import os, cv2, atexit, json, random
import pandas as pd
from glob import glob
from segmenter import colorSegmenter


class valve:
    """
    This script is used for intaking new samples for the Rocky Mountain
    snowpack dataset and orchestrating the segmentation and labeling process.

    Class Atributes:
        - dataset_dir (str) - Directory storing the Rocky Mountain snowpack dataset

    Class Functions:
        - orient() - Orients the intake valve to the current state of the folder
        - segment() - 

    """
    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir
        if self.dataset_dir[-1] != '/':
            self.dataset_dir += '/'

        # Orient to the dataset dir
        self.sites = glob(f"{self.dataset_dir}intake/site*/")

        # Grab all raw images already intaken
        self.raw_images = glob(f"{self.dataset_dir}raw/magnified_profiles/*") + glob(f"{self.dataset_dir}raw/crystal_cards/*")
        self.image_count = len(self.raw_images) # Assess the count
        print(f"Current image count: {self.image_count}")

        # Open the image manifest
        self.manifest = pd.read_csv(f"{self.dataset_dir}intake/image_manifest.csv")

        # Load site data
        self.sites = pd.read_csv(f"{self.dataset_dir}intake/site_logs.csv")

        # Load temperature data
        self.temps = pd.read_csv(f"{self.dataset_dir}intake/site_temps.csv")

        atexit.register(self.save_state)

    def intake(self, site_folder):
        """
        Copy data from intake to raw folder and rename them with standardized name
        and image numbering that follows the last image saved. Save the name conversion
        within the manifest file

        Arguments:
            - site_folder (str) - Subfolder to intake data from
        """
        # Check if site has been processed
        site = int(site_folder.split('_')[-1].split('/')[0])

        if site in self.sites['site'].values:
            print(f"Site already intaken, canceling intake...")

        # Add intake parent folder if not specified
        if site_folder[:6] != 'intake':
            print(f"Parent folder intake/ not properly added, adding parent folder to path...")
            site_folder = 'intake/' + site_folder

        if site_folder[-1] != '/': # Add a final / if needed
            site_folder += '/'

        # Check if folder exists
        if os.path.exists(f"{self.dataset_dir}{site_folder}") == False:
            print(f"Site folder {self.dataset_dir}{site_folder} not found...")
            return
        
        # Grab site specific data
        intake_site = pd.read_csv(f"{self.dataset_dir}{site_folder}site_logs.csv")

        new_site = { # Create entry for site
            'site': site,
            'ascending_mountain': intake_site['ascending_mountain'][0],
            'city_state_country': intake_site['city_state_country'][0],
            'collector': intake_site['collector'][0],
            'coordinates': intake_site['coordinates'][0],
            'date': intake_site['date'][0],
            'time': intake_site['time'][0],
            'snowpack_depth': intake_site['snowpack_depth'][0],
            'slope_face': intake_site['slope_face'][0],
            'slope_gradient': intake_site['slope_gradient'][0],
            'air_temperature': intake_site['air_temperature'][0],
            'avalanches_spotted': intake_site['avalanches_spotted'][0],
            'wind_loading': intake_site['wind_loading'][0],
            'notes': intake_site['notes'][0],
        }

        self.sites = pd.concat([self.sites, pd.DataFrame([new_site])], ignore_index=True)

        # Grab core data
        intake_temps = pd.read_csv(f"{self.dataset_dir}{site_folder}site_temps.csv")
        for record in intake_temps.iterrows():
            new_temp = { # Create entry for core temperature
                'site': site,
                'column': record[1]['column'],
                'core': record[1]['core'],
                'core_temperature': record[1]['core_temperature'],
            }
            self.temps = pd.concat([self.temps, pd.DataFrame([new_temp])], ignore_index=True)
            
        
        # Construct directory
        data_dir = f"{self.dataset_dir}{site_folder}/*/*"

        # Grab all intake files and sort
        image_files = glob(data_dir)
        print(f"Image file count: {len(image_files)}")

        # Sort images - works for Apple and Android 
        image_numbers = [int(''.join(os.path.basename(file).split('.')[0].split('_')[1:])) for file in image_files]
        zipper = zip(image_numbers, image_files)
        zipper = sorted(zipper)
        image_numbers, image_files = zip(*zipper)

        # Iterative prepare and copy them to raw
        for image_filepath in image_files:
            path_split = image_filepath.split('/')
            image_filename = path_split[-1]
            image_type = path_split[-2]

            # Load images
            image = cv2.imread(image_filepath)

            # Rotate if needed
            h, w = image.shape[:2]
            if image_type == "magnified_profiles" and w > h:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Figure out image number in dataset
            new_filename = f"image_{self.image_count + 1}.png"

            # Double check image filename doesn't exist
            if os.path.exists(f"{self.dataset_dir}raw/{image_type}/{new_filename}"):
                FileExistsError(f"Image {new_filename} already exists in raw folder")
        
            # Save image to raw folder
            cv2.imwrite(f"{self.dataset_dir}raw/{image_type}/{new_filename}", image)

            if image_type == 'magnified_profiles': # Save magnified image to preprocessed
                cv2.imwrite(f"{self.dataset_dir}preprocessed/{image_type}/{new_filename}", image)

            # Add new info to manifest
            new_entry = { # Create new entry
                'image': new_filename,
                'original_filename': image_filename,
                'image_type': image_type[:-1]
            }
            self.manifest = pd.concat([self.manifest, pd.DataFrame([new_entry])], ignore_index=True)

            # Increment image count
            self.image_count += 1
        
        
    def segment_cards(self):
        """
        Segment crystal card images using the color segmenter class
        """
        # Initialize crystal card segmenter
        segmenter = colorSegmenter(self.dataset_dir)
        
        # Grab all crystal card images
        snow_images = glob(f"{self.dataset_dir}raw/crystal_cards/*.png")
        for snow_image in snow_images:
            results = segmenter.segment(snow_image, False)
            if results:
                print(f"Segmentation successful")
            else:
                print(f"Segmentation failed")

    def update_metadata(self):
        """
        Update metadata from manually labeled crystal card segments and copy labels
        back to the intake folder for archiving
        """

        # Define runtime parameters
        label_dir = 'preprocessed/written_labels/'
        preproc_dirs = ['preprocessed/cores/', 'preprocessed/magnified_profiles/', 'preprocessed/profiles/']
        raw_dirs = ['raw/crystal_cards/', 'raw/magnified_profiles/']

        extract_number = lambda filename : int(filename.split('image_')[1].split('_')[0].split('.png')[0])

        # Grab all label files
        label_files = glob(f"{self.dataset_dir}{label_dir}*")

        # Grab the label photo numbers
        label_filenumbers = [os.path.basename(file) for file in label_files] # Remove the image paths and leave just the filenames

        # Sort files in ascending order
        filenumbers = [extract_number(file) for file in label_filenumbers]
        zipper = sorted(zip(filenumbers, label_files))
        filenumbers, label_files = zip(*zipper)

        # Iterate through preprocessing states
        for data_dirs, processing_state in zip([raw_dirs, preproc_dirs], ['raw', 'preprocessed']):
            jsonl_data = []

            for data_dir in data_dirs:
                datatype_images = glob(f"{self.dataset_dir}{data_dir}*") # Grab all segmented images

                # Grab the label photo numbers
                datatype_filenumbers = [os.path.basename(file) for file in datatype_images] # Remove the image paths and leave just the filenames

                # Sort files in ascending order
                filenumbers = [extract_number(file) for file in datatype_filenumbers]
                zipper = sorted(zip(filenumbers, datatype_images))
                filenumbers, datatype_images = zip(*zipper)

                # Iterate through each label file
                for label_ind, label_file in enumerate(label_files):

                    # Grab the label for the image
                    label = [int(datum) for datum in label_file.split('/')[-1].split('.png')[0].split('_')[2:]]
                    if len(label) == 3:
                        label.append(None)
                    label_image_number = extract_number(label_file)

                    if len(label_files) == label_ind + 1: # If there is no next image
                        next_image_number = 99999999999  # large dummy value
                    
                    else: # Grab next label
                        next_label_file = label_files[label_ind + 1]
                        next_image_number = extract_number(next_label_file)

                    # Find the site info
                    site_mask = self.sites['site'] == label[0]

                    # Find core temp
                    temp_mask = (self.temps['site'] == label[0]) & (self.temps['column'] == label[1]) & (self.temps['core'] == label[2])
                    if temp_mask.any():
                        core_temp = self.temps.loc[temp_mask, 'temperature'].iloc[0]
                    else:
                        core_temp = None
                    print(f"Temp mask for labels {label}: {temp_mask}")

                    # Iterate through all raw files
                    for image_filepath in datatype_images:
                        image_filename = os.path.basename(image_filepath)
                        image_number = extract_number(image_filename) 
                        # If the file is between the current and next label
                        if image_number >= label_image_number and image_number < next_image_number:
                            # Assess core depth
                            core_depth = label[2] * 10.0

                            # Decide what split to put it in
                            flip = random.random()
                            if flip <= 0.8:
                                split = 'train'
                            elif flip > 0.8 and flip <= 0.9:
                                split = 'test'
                            else:
                                split = 'validation'

                            # Create metadata entry
                            new_entry = {
                                'image': f"{data_dir}{image_filename}",
                                'datatype': data_dir.split('/')[-2][:-1],
                                'site': label[0],
                                'column': label[1],
                                'core': label[2],
                                'segment': label[3],
                                'core_temperature': core_temp,
                                'air_temperature': self.sites.loc[site_mask, 'air_temperature'].iloc[0],
                                'ascending_mountain': self.sites.loc[site_mask, 'ascending_mountain'].iloc[0],
                                'city_state_country': self.sites.loc[site_mask, 'city_state_country'].iloc[0],
                                'collector': self.sites.loc[site_mask, 'collector'].iloc[0],
                                'coordinates': [float(coord) for coord in self.sites.loc[site_mask, 'coordinates'].iloc[0].split(', ')],
                                'date': self.sites.loc[site_mask, 'date'].iloc[0],
                                'time': self.sites.loc[site_mask, 'time'].iloc[0],
                                'snowpack_depth': self.sites.loc[site_mask, 'snowpack_depth'].iloc[0],
                                'core_depth': core_depth,
                                'slope_face': self.sites.loc[site_mask, 'slope_face'].iloc[0],
                                'slope_gradient': self.sites.loc[site_mask, 'slope_gradient'].iloc[0],
                                'avalanches_spotted': self.sites.loc[site_mask, 'avalanches_spotted'].iloc[0],
                                'wind_loading': self.sites.loc[site_mask, 'wind_loading'].iloc[0],
                                'notes': self.sites.loc[site_mask, 'notes'].iloc[0],
                                'split': split
                            }

                            # Append to the jsonl dataframe
                            jsonl_data.append(new_entry)

                            # Add image label to manifest
                            manifest_image_mask = self.manifest['image'] == image_filename
                            self.manifest.loc[manifest_image_mask, 'collector'] = self.sites.loc[site_mask, 'collector'].iloc[0]
                            self.manifest.loc[manifest_image_mask, 'site'] = label[0]
                            self.manifest.loc[manifest_image_mask, 'column'] = label[1]
                            self.manifest.loc[manifest_image_mask, 'core'] = label[2]
                            self.manifest.loc[manifest_image_mask, 'segment'] = label[3]

                        if image_number >= next_image_number:
                            break
        
            metadata_path = f"{self.dataset_dir}metadata/{processing_state}.jsonl"
            with open(metadata_path, 'w') as f:
                for entry in jsonl_data:
                    f.write(json.dumps(entry) + '\n')



    def backup_intakes(self):
        """
        Create a backup of the intake folder to ensure dataset can be replicated
        in case of catastrophy.
        """

    def upload_huggingface(self):
        """
        Upload new data to hugging face repository
        """

    def save_state(self):
        """
        Save the current state of the image manifest
        """
        self.manifest.to_csv(f"{self.dataset_dir}intake/image_manifest.csv", index = False)

        # Save site data
        self.sites.to_csv(f"{self.dataset_dir}intake/site_logs.csv", index = False)

        # Load temperature data
        self.temps.to_csv(f"{self.dataset_dir}intake/site_temps.csv", index = False)

        print(f"Intake records saved...")
