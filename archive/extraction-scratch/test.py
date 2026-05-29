import radiomics
from radiomics import featureextractor
import SimpleITK as sitk

# Define paths to your image, mask, and parameter file (e.g., params.yaml)
imageFile = 'path/to/image.nii.gz'
maskFile = 'path/to/mask.nii.gz'
paramFile = 'path/to/params.yaml'

# Initialize the extractor
extractor = featureextractor.RadiomicsFeatureExtractor(paramFile)

# Execute voxel-based extraction
result = extractor.execute(imageFile, maskFile, voxelBased=True)

# The result dictionary will contain sitk.Image objects for the computed feature maps
for feature_name, feature_value in result.items():
    if isinstance(feature_value, sitk.Image):
        print(f"Saving feature map for: {feature_name}")
        sitk.WriteImage(feature_value, f"map_{feature_name}.nrrd")
