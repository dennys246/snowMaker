import os, cv2, atexit, json
import pandas as pd
from glob import glob
from segmenter import colorSegmenter
import schema
import pits, layers, cores, ect


# The side tables, in the order they are validated and written. Each is one grain:
# pits (site-day), layers (site, column, layer_index), cores (site, column, core),
# ect (site, column, test_index, fracture_index).
SIDE_TABLES = (pits.SPEC, layers.SPEC, cores.SPEC, ect.SPEC)

# The image-table files. Both carry the same columns; 'raw' holds the source
# photographs and 'preprocessed' the segmented ones.
IMAGE_TABLES = ('raw', 'preprocessed')

# Where the dataset is published
HF_DATASET_REPO = "rmdig/rocky_mountain_snowpack"


def side_table_csv(spec):
    """Intake CSV filename for a side table: site_pits.csv, site_layers.csv, ..."""
    return f"site_{spec.name}.csv"


def read_jsonl(path):
    """Read a JSONL metadata file into a list of dicts, preserving order."""
    return schema.read_jsonl(path)


def strip_legacy_ect(entry, context = ""):
    """
    Remove the seven ect_* columns the image table carried until 2026-08-25.

    They were null on every published row. A non-null value here would be a real
    test recorded against the old (site, column) grain, which cannot be moved to
    ect.jsonl automatically because the new grain needs test_index and
    fracture_index -- so that raises rather than being dropped.
    """
    for column in ect.LEGACY_IMAGE_COLUMNS:
        if column not in entry:
            continue
        if entry[column] is not None:
            raise schema.SchemaError(
                f"{context}{column}={entry[column]!r} on an image row. ECT values "
                f"live in metadata/ect.jsonl now; move this test there by hand "
                f"(see docs/SCHEMA_V2.md) before re-running."
            )
        del entry[column]
    return entry


def fetch_hub_jsonl(api, repo_id, filename):
    """
    Read one metadata file from the Hub, or None if it is not there yet.

    Arguments:
        - api (HfApi) - Authenticated or anonymous Hub client
        - repo_id (str) - Dataset repo to read from
        - filename (str) - Path within the repo, e.g. 'metadata/raw.jsonl'
    """
    from huggingface_hub.errors import EntryNotFoundError

    try:
        local_copy = api.hf_hub_download(repo_id, filename, repo_type = "dataset")
    except EntryNotFoundError:
        return None
    return read_jsonl(local_copy)


def diff_metadata(local_rows, hub_rows, key = None):
    """
    Compare two metadata files by row identity.

    Image tables are identified by file_path; a side table by its key columns
    (pass ``key`` as the tuple of column names). Returns ``(added, removed)`` --
    the identities this copy would add to the Hub and the ones it would delete
    from it. A non-empty ``removed`` means the local copy is missing rows the Hub
    already has, which is almost always a stale working copy rather than an
    intended withdrawal.
    """
    def identity(row):
        if key is None:
            return row['file_path']
        return tuple(row[k] for k in key)

    local_ids = {identity(row) for row in local_rows}
    if hub_rows is None:
        return local_ids, set()
    hub_ids = {identity(row) for row in hub_rows}
    return local_ids - hub_ids, hub_ids - local_ids


def validate_metadata_dir(dataset_dir):
    """
    Validate every metadata table under ``dataset_dir`` and the links between them.

    Returns ``{name: rows}`` for every file present. Raises SchemaError on any
    violation, so nothing invalid can be uploaded. Prints warnings.
    """
    if not dataset_dir.endswith('/'):
        dataset_dir += '/'

    tables = {}
    image_rows = []
    for name in IMAGE_TABLES:
        path = f"{dataset_dir}metadata/{name}.jsonl"
        if os.path.exists(path):
            rows = read_jsonl(path)
            for position, entry in enumerate(rows):
                leftover = [c for c in ect.LEGACY_IMAGE_COLUMNS if c in entry]
                if leftover:
                    raise schema.SchemaError(
                        f"{name}.jsonl row {position} still carries {leftover}; run "
                        f"scripts/migrate_side_tables.py to move the ECT columns off "
                        f"the image table"
                    )
            tables[name] = rows
            image_rows.extend(rows)

    warnings = []
    for spec in SIDE_TABLES:
        path = f"{dataset_dir}metadata/{spec.name}.jsonl"
        if not os.path.exists(path):
            raise schema.SchemaError(
                f"{path} is missing; run scripts/migrate_side_tables.py to create the "
                f"side tables"
            )
        rows = read_jsonl(path)
        warnings.extend(spec.validate_table(rows, context = f"{spec.name}.jsonl: "))
        tables[spec.name] = rows

    warnings.extend(schema.validate_dataset(
        {spec.name: tables[spec.name] for spec in SIDE_TABLES},
        image_rows = image_rows or None,
        context = "metadata/: ",
    ))
    for warning in warnings:
        print(f"WARNING: {warning}")
    return tables


def upload_metadata(dataset_dir, repo_id = HF_DATASET_REPO, dry_run = True,
                    token = None, allow_shrink = False, commit_message = None):
    """
    Upload the metadata files and dataset card to the Hugging Face dataset repo.

    Only metadata/*.jsonl and README.md are sent. The images are already on the
    Hub and are not touched, so this is a few megabytes rather than the ~96 GB the
    full repo weighs.

    Defaults to a dry run. Passing dry_run = False is the explicit confirmation to
    write to the Hub.

    Before uploading anything it validates every table, then checks the local copy
    against what is already there and refuses to proceed if the Hub holds rows this
    copy does not, because uploading would delete them. A stale working copy is the
    normal way that happens -- on 2026-08-19 the working copy held sites 0-2 (2345
    preprocessed rows) while the Hub held sites 0-6 (4040), and an unguarded upload
    would have dropped 1695 rows.

    Arguments:
        - dataset_dir (str) - Root of the dataset copy to upload from
        - repo_id (str) - Hugging Face dataset repo to upload to
        - dry_run (bool) - Report the plan without writing to the Hub
        - token (str) - Hugging Face token, or None to use the ambient login
        - allow_shrink (bool) - Upload even though it would remove rows from the
          Hub. Only correct when rows are being deliberately withdrawn.
        - commit_message (str) - Commit message on the Hub
    """
    from huggingface_hub import HfApi

    if not dataset_dir.endswith('/'):
        dataset_dir += '/'

    api = HfApi(token = token)

    # Never push a schema violation to the Hub
    tables = validate_metadata_dir(dataset_dir)
    keys = {spec.name: spec.key for spec in SIDE_TABLES}

    print(f"Checking {len(tables)} metadata file(s) against {repo_id}...")

    shrinking = []
    for name, local_rows in tables.items():
        filename = f"{name}.jsonl"
        hub_rows = fetch_hub_jsonl(api, repo_id, f"metadata/{filename}")
        added, removed = diff_metadata(local_rows, hub_rows, key = keys.get(name))

        state = "not on Hub yet" if hub_rows is None else f"{len(hub_rows)} on Hub"
        print(
            f"  {filename:<22} {len(local_rows):>6} row(s) local, {state:<16}"
            f"  +{len(added)} / -{len(removed)}"
        )
        if removed:
            shrinking.append((filename, sorted(removed, key = repr)))

    if shrinking and not allow_shrink:
        summary = "\n".join(
            f"  {name}: {len(ids)} row(s) would be deleted, e.g. {ids[:3]}"
            for name, ids in shrinking
        )
        raise RuntimeError(
            f"Refusing to upload: the Hub holds rows this copy does not, so "
            f"uploading would delete them.\n{summary}\n"
            f"Re-sync this working copy from {repo_id} first. Pass "
            f"allow_shrink = True only if the removal is intended."
        )

    if dry_run:
        print(
            f"\nDRY RUN -- nothing was uploaded. Re-run with dry_run = False to "
            f"write metadata/*.jsonl and README.md to {repo_id}."
        )
        return

    print(f"\nUploading metadata and dataset card to {repo_id}...")
    api.upload_folder(
        repo_id = repo_id,
        repo_type = "dataset",
        folder_path = dataset_dir,
        allow_patterns = ["metadata/*.jsonl", "README.md"],
        commit_message = commit_message or "Update metadata tables and dataset card",
    )
    print("Upload complete.")


class valve:
    """
    This script is used for intaking new samples for the Rocky Mountain
    snowpack dataset and orchestrating the segmentation and labeling process.

    Class Atributes:
        - dataset_dir (str) - Directory storing the Rocky Mountain snowpack dataset

    Class Functions:
        - intake() - Copy a site's photographs and field records into the dataset
        - segment_cards() - Segment crystal card images
        - update_metadata() - Write the image table and side tables from labels
        - upload_huggingface() - Push metadata and the card to the Hub
        - save_state() - Save the intake bookkeeping CSVs

    """
    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir
        if self.dataset_dir[-1] != '/':
            self.dataset_dir += '/'

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

        # Load the side tables collected under the field protocol: pits, layer
        # profiles, cores and extended column tests. Each is validated on load.
        self.side_tables = {spec.name: self.load_side_table(spec) for spec in SIDE_TABLES}

        atexit.register(self.save_state)

    def load_side_table(self, spec):
        """
        Load one master side-table CSV from intake/ and validate it.

        A site collected before the field protocol simply has no rows here. For
        the ECT table that reads as *not tested*, which is distinct from an "X"
        result (tested, no fracture in 30 taps).

        Arguments:
            - spec (schema.TableSpec) - Which table to load
        """
        path = f"{self.dataset_dir}intake/{side_table_csv(spec)}"
        if not os.path.exists(path):
            print(f"No {spec.name} records found at {path}, starting an empty {spec.name} table...")
            return []
        warnings = []
        records = schema.read_csv(spec, path, warnings = warnings)
        warnings.extend(spec.validate_table(records, context = f"{path}: "))
        for warning in warnings:
            print(f"WARNING: {warning}")
        print(f"Loaded {len(records)} {spec.name} record(s) from {path}")
        return records

    def intake_side_table(self, spec, site, site_folder):
        """
        Read one side-table CSV from a site's intake folder and add it to the master.

        The CSV's site column, if present, must match the folder; if absent it is
        filled from the folder. Duplicate keys and any schema violation raise.

        Arguments:
            - spec (schema.TableSpec) - Which table
            - site (int) - Site number from the folder name
            - site_folder (str) - Site intake folder, relative to dataset_dir
        """
        path = f"{self.dataset_dir}{site_folder}{side_table_csv(spec)}"
        if not os.path.exists(path):
            print(f"No {side_table_csv(spec)} for site {site}, leaving {spec.name} empty for it...")
            return

        if any(record['site'] == site for record in self.side_tables[spec.name]):
            print(f"Site {site} already in the {spec.name} table, skipping {side_table_csv(spec)}...")
            return

        # Fill or check the site column before coercion, using the raw cells
        import csv
        raw_rows = []
        with open(path, 'r', encoding = 'utf-8-sig', newline = '') as handle:
            for raw in csv.DictReader(handle):
                recorded = raw.get('site')
                if schema.is_null(recorded):
                    raw['site'] = str(site)
                elif str(recorded).strip() != str(site):
                    raise schema.SchemaError(
                        f"{path}: site column says {recorded!r} but the folder is site {site}"
                    )
                raw_rows.append(raw)

        warnings = []
        records = [
            spec.normalize(raw, f"{path} row {position + 1}: ", warnings)
            for position, raw in enumerate(raw_rows)
        ]
        merged = schema.merge_records(spec, self.side_tables[spec.name], records, context = f"{path}: ")
        warnings.extend(spec.validate_table(merged, context = f"{path}: "))
        for warning in warnings:
            print(f"WARNING: {warning}")
        self.side_tables[spec.name] = merged
        print(f"Recorded {len(records)} {spec.name} record(s) for site {site}...")

    def parse_coordinates(self, raw, site):
        """
        Parse a 'latitude, longitude' pair at the full precision it was recorded at.

        Nothing here rounds or truncates -- whatever precision the receiver gave is
        what gets logged. Site 0 was recorded as [39.66, -105.88]; two decimal
        places is a ~1.1 km box, and in Loveland Pass terrain that box spans
        3533-3748 m and straddles an elevation-band boundary, so a DEM elevation
        lookup against it cannot be trusted. Low precision warns rather than raises
        so historical sites still load.

        Elevation deliberately has no column of its own: it is recoverable from the
        GPS fix via a DEM (USGS 3DEP). Aspect is not recoverable that way, which is
        why it is carried explicitly (pits.aspect_deg_true, legacy slope_face).

        Arguments:
            - raw (str) - Coordinate string from the site log
            - site (int) - Site number, for the warning message
        """
        warnings = []
        try:
            coordinates = schema.coerce_value(
                pits.SPEC.by_name['coordinates'], raw, f"site {site} ", warnings
            )
        except schema.SchemaError as error:
            raise ValueError(
                f"Site {site} coordinates {raw!r} are not a 'latitude, longitude' pair: {error}"
            ) from None
        if coordinates is None:
            raise ValueError(f"Site {site} has no coordinates")
        for warning in warnings:
            print(f"WARNING: {warning}")
        return coordinates

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
        
        # Add site logs and temps unless already recorded in the master CSVs
        if site in self.sites['site'].values:
            print(f"Site {site} already in site logs, skipping site log and temperature intake...")
        else:
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

        # Grab the field-protocol tables: site_pits.csv, site_layers.csv,
        # site_cores.csv, site_ect.csv. A site dug before the protocol has none of
        # them; its side-table rows stay absent (not measured), never placeholders.
        for spec in SIDE_TABLES:
            self.intake_side_table(spec, site, site_folder)


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
        back to the intake folder for archiving.

        Writes the two image tables (raw.jsonl, preprocessed.jsonl) and then the
        four side tables (pits, layers, cores, ect), checking the links between
        them before anything is written.
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

        image_tables = {}

        # Iterate through preprocessing states
        for data_dirs, processing_state in zip([raw_dirs, preproc_dirs], ['raw', 'preprocessed']):
            old_data = []

            with open(f"{self.dataset_dir}metadata/{processing_state}.jsonl", 'r', encoding='utf-8') as f:
                for position, line in enumerate(f):
                    datum = json.loads(line)
                    datum.pop('split', None)  # row-level splits were dropped on 2026-08-19
                    # Rows written before 2026-08-25 carried the seven ect_* columns,
                    # null on every row. They live in ect.jsonl now.
                    strip_legacy_ect(datum, context = f"{processing_state}.jsonl row {position}: ")
                    old_data.append(datum)

            previously_handled = [datum.get('file_path', datum['image']) for datum in old_data]

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
                        label.append(-1)
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
                        core_temp = self.temps.loc[temp_mask, 'core_temperature'].iloc[0]
                    else:
                        core_temp = None
                    print(f"Temp mask for labels {label}: {temp_mask}")

                    # Iterate through all files
                    for image_filepath in datatype_images:
                        image_filename = os.path.basename(image_filepath)

                        # Check if previously handled
                        if f"{data_dir}{image_filename}" in previously_handled:
                            continue # Keep existing data

                        image_number = extract_number(image_filename) 
                        # If the file is between the current and next label
                        if image_number >= label_image_number and image_number < next_image_number:
                            # Assess core depth
                            core_depth = label[2] * 10.0

                            # Create metadata entry (cast pandas/numpy scalars to native
                            # Python types so the entries stay JSON serializable)
                            new_entry = {
                                'image': f"https://huggingface.co/datasets/RMDig/rocky_mountain_snowpack/resolve/main/{data_dir}{image_filename}",
                                'file_path': f"{data_dir}{image_filename}",
                                'datatype': data_dir.split('/')[-2][:-1],
                                'site': label[0],
                                'column': label[1],
                                'core': label[2],
                                'segment': label[3],
                                'core_temperature': None if core_temp is None else float(core_temp),
                                'air_temperature': float(self.sites.loc[site_mask, 'air_temperature'].iloc[0]),
                                'ascending_mountain': str(self.sites.loc[site_mask, 'ascending_mountain'].iloc[0]),
                                'city_state_country': str(self.sites.loc[site_mask, 'city_state_country'].iloc[0]),
                                'collector': str(self.sites.loc[site_mask, 'collector'].iloc[0]),
                                'coordinates': self.parse_coordinates(self.sites.loc[site_mask, 'coordinates'].iloc[0], label[0]),
                                'date': str(self.sites.loc[site_mask, 'date'].iloc[0]),
                                'time': str(self.sites.loc[site_mask, 'time'].iloc[0]),
                                'snowpack_depth': float(self.sites.loc[site_mask, 'snowpack_depth'].iloc[0]),
                                'core_depth': core_depth,
                                'slope_face': float(self.sites.loc[site_mask, 'slope_face'].iloc[0]),
                                'slope_angle': float(self.sites.loc[site_mask, 'slope_gradient'].iloc[0]),
                                'avalanches_spotted': int(self.sites.loc[site_mask, 'avalanches_spotted'].iloc[0]),
                                'wind_loading': str(self.sites.loc[site_mask, 'wind_loading'].iloc[0]),
                                'notes': str(self.sites.loc[site_mask, 'notes'].iloc[0]),
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
            
            # Append new data onto old
            image_tables[processing_state] = old_data + jsonl_data

        # Assemble the side tables and check every link before writing anything
        side_tables = self.build_side_tables(image_tables)

        for processing_state, rows in image_tables.items():
            schema.write_jsonl(f"{self.dataset_dir}metadata/{processing_state}.jsonl", rows)

        for spec in SIDE_TABLES:
            schema.write_jsonl(f"{self.dataset_dir}metadata/{spec.name}.jsonl", side_tables[spec.name])
            print(f"Wrote {len(side_tables[spec.name])} {spec.name} record(s)")

    def build_side_tables(self, image_tables):
        """
        Merge the published side tables with the intake masters and validate.

        Rows already in metadata/<table>.jsonl are kept (the same way image rows
        are); rows from intake/site_<table>.csv are added. The same key recorded
        differently in the two raises. A core that has image rows but no cores
        record gets one carrying only what the legacy pipeline recorded -- its
        ladder depth and, if there is a site_temps reading, its temperature -- with
        a warning, so photographed cores are never dangling. A site with image rows
        but no pits record raises: collector_id and method_deviation cannot be
        invented.

        Arguments:
            - image_tables (dict) - {'raw': rows, 'preprocessed': rows}
        """
        image_rows = [row for rows in image_tables.values() for row in rows]
        warnings = []
        side_tables = {}
        for spec in SIDE_TABLES:
            path = f"{self.dataset_dir}metadata/{spec.name}.jsonl"
            published = read_jsonl(path) if os.path.exists(path) else []
            for position, row in enumerate(published):
                spec.validate(row, context = f"{spec.name}.jsonl row {position}: ")
            side_tables[spec.name] = schema.merge_records(
                spec, published, self.side_tables[spec.name], context = f"{spec.name}: "
            )

        # Photographed cores without a cores record
        known = {cores.SPEC.key_of(row) for row in side_tables['cores']}
        missing = sorted({(row['site'], row['column'], row['core']) for row in image_rows} - known)
        for site, column, core in missing:
            temp_mask = (self.temps['site'] == site) & (self.temps['column'] == column) & (self.temps['core'] == core)
            reading = float(self.temps.loc[temp_mask, 'core_temperature'].iloc[0]) if temp_mask.any() else None
            if reading is not None and pd.isna(reading):
                reading = None
            side_tables['cores'].append(cores.legacy_core_row(site, column, core, reading))
        if missing:
            warnings.append(
                f"{len(missing)} photographed core(s) had no site_cores.csv record; "
                f"wrote rows carrying only the ladder depth and any site_temps reading "
                f"(breakability_field_count, recovery_quality, layer_ids are null): "
                f"{missing[:8]}{' ...' if len(missing) > 8 else ''}"
            )
        side_tables['cores'].sort(key = cores.SPEC.key_of)

        for spec in SIDE_TABLES:
            warnings.extend(spec.validate_table(side_tables[spec.name], context = f"{spec.name}: "))
        warnings.extend(schema.validate_dataset(side_tables, image_rows = image_rows, context = "metadata: "))
        for warning in warnings:
            print(f"WARNING: {warning}")
        return side_tables


    def backup_intakes(self):
        """
        Create a backup of the intake folder to ensure dataset can be replicated
        in case of catastrophy.
        """

    def upload_huggingface(self, repo_id = HF_DATASET_REPO, dry_run = True,
                           token = None, allow_shrink = False, commit_message = None):
        """
        Upload this dataset's metadata and card to the Hugging Face dataset repo.

        Thin wrapper around upload_metadata, which does the work and can also be
        called without a valve. See that function for the pre-flight behaviour.
        """
        return upload_metadata(
            self.dataset_dir,
            repo_id = repo_id,
            dry_run = dry_run,
            token = token,
            allow_shrink = allow_shrink,
            commit_message = commit_message,
        )

    def save_state(self):
        """
        Save the current state of the image manifest
        """
        self.manifest.to_csv(f"{self.dataset_dir}intake/image_manifest.csv", index = False)

        # Save site data
        self.sites.to_csv(f"{self.dataset_dir}intake/site_logs.csv", index = False)

        # Save temperature data
        self.temps.to_csv(f"{self.dataset_dir}intake/site_temps.csv", index = False)

        # Save the side tables collected under the field protocol
        for spec in SIDE_TABLES:
            schema.write_csv(spec, f"{self.dataset_dir}intake/{side_table_csv(spec)}", self.side_tables[spec.name])

        print(f"Intake records saved...")
