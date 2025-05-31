
from glob import glob


def grab_filenames(filepaths):
    return [filepath.split('/')[-1] for filepath in filepaths]

# Grab all images
raw_images = set(grab_filenames(glob("../snow-profiles/raw/*.JPG")))

# Grab magnified images
magnified_images = set(grab_filenames(glob("../snow-profiles/segmented/magnified-profiles/tempA/*.JPG")))
print(f"Magnified images: {magnified_images}\n\n")

# Grab segmented profiles
profile_images = set(grab_filenames(glob("../snow-profiles/segmented/profiles/*.JPG")))
print(f"Profile images: {profile_images}\n\n")

# Grab segmented cores
core_images = set(grab_filenames(glob("../snow-profiles/segmented/cores/*.JPG")))
print(f"Core images: {core_images}\n\n")
# Grab segmented labels
label_images = grab_filenames(glob("../snow-profiles/segmented/labels/*.JPG"))
label_images = set([f"{"_".join(image.split('_')[:2])}.JPG" for image in magnified_images])
print(f"Label images: {label_images}\n\n")

# Find images that failed to process altogether
failed_processing = raw_images
failed_processing -= magnified_images # Remove non-crystal card images
failed_processing -= profile_images 
failed_processing -= label_images
failed_processing -= core_images
failed_processing = list(failed_processing)
print(f"Images that failed to segment (count {len(failed_processing)}):\n -{"\n -".join(list(failed_processing))}")

failed_processing = raw_images
failed_processing -= magnified_images # Remove non-crystal card images
failed_processing -= profile_images
print(f"\n\nImages that failed to segment profile (red)\n - {'\n - '.join(list(failed_processing))}")

failed_processing = raw_images
failed_processing -= magnified_images # Remove non-crystal card images
failed_processing -= profile_images
failed_processing -= label_images
print(f"\n\nImages that failed to segment labels (green)\n - {'\n - '.join(list(failed_processing))}")

failed_processing = raw_images
failed_processing -= magnified_images # Remove non-crystal card images
failed_processing -= profile_images
failed_processing -= label_images
failed_processing -= core_images
print(f"\n\nImages that failed to segment cores (blue)\n - {'\n - '.join(list(failed_processing))}")