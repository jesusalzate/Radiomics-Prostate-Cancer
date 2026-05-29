import os
import numpy as np
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk

##############################################################################
# PATHS
##############################################################################

current_file = os.path.abspath(__file__)

script_dir = os.path.dirname(current_file)

# YAML de configuración
PARAMS_FILE = os.path.join(
    script_dir,
    "Params_T2w.yaml"
)

# Directorio raíz del proyecto
project_root = os.path.abspath(
    os.path.join(
        current_file,
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
    )
)

# Carpeta base de datos
pre_path = "/projects/ceib/data_picai"

##############################################################################
# INPUTS
##############################################################################

imageFile = os.path.join(
    pre_path,
    "data/images/10005/10005_1000005_t2w.mha"
)

maskFile = os.path.join(
    pre_path,
    "data/labels/csPCa_lesion_delineations/human_expert/resampled/10005_1000005.nii.gz"
)

##############################################################################
# OUTPUTS
##############################################################################

output_dir = os.path.join(
    project_root,
    "artifacts",
    "voxel_feature_maps"
)

os.makedirs(output_dir, exist_ok=True)

##############################################################################
# BINARIZE MASK
##############################################################################

def binarize_mask(mask_image):
    """
    Convierte cualquier valor > 0 a 1.
    """

    mask_array = sitk.GetArrayFromImage(mask_image)

    print("Valores originales en máscara:")
    print(np.unique(mask_array))

    binary_array = (mask_array > 0).astype(np.uint8)

    print("Valores binarizados:")
    print(np.unique(binary_array))

    binary_mask = sitk.GetImageFromArray(binary_array)

    binary_mask.CopyInformation(mask_image)

    return binary_mask

##############################################################################
# LOAD IMAGES
##############################################################################

image = sitk.ReadImage(imageFile)

mask = sitk.ReadImage(maskFile)

##############################################################################
# BINARIZATION
##############################################################################

mask = binarize_mask(mask)

##############################################################################
# RADIOMICS
##############################################################################

# Inicializar extractor
extractor = featureextractor.RadiomicsFeatureExtractor(
    PARAMS_FILE
)

# Ejecutar extracción voxel-based
result = extractor.execute(
    image,
    mask,
    voxelBased=True
)

##############################################################################
# SAVE FEATURE MAPS
##############################################################################

for feature_name, feature_value in result.items():

    if isinstance(feature_value, sitk.Image):

        print(f"Saving feature map for: {feature_name}")

        output_path = os.path.join(
            output_dir,
            f"map_{feature_name}.nrrd"
        )

        sitk.WriteImage(
            feature_value,
            output_path
        )