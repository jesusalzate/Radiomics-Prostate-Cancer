import os
import json
import re
import zipfile
import tempfile
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

##############################################################################
# PATHS
##############################################################################

current_file = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file)

project_root = os.path.abspath(
    os.path.join(
        current_file,
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
    )
)

pre_path = "/projects/ceib/data_picai"

output_dir = "/projects/ceib/data_picai/test_voxel"

os.makedirs(output_dir, exist_ok=True)

os.makedirs(output_dir, exist_ok=True)

##############################################################################
# CONFIGURACIÓN DEL PACIENTE / ESTUDIO
##############################################################################

patient_id = "10005"
study_id = "1000005"

maskFile = os.path.join(
    pre_path,
    f"data/labels/csPCa_lesion_delineations/human_expert/resampled/{patient_id}_{study_id}.nii.gz"
)

# OJO:
# En PI-CAI muchas veces la modalidad DWI de alto b-value aparece como "hbv".
# Si tus archivos realmente se llaman "_dwi.mha", cambia "hbv" por "dwi".
modalities = {
    "t2w": {
        "image": os.path.join(
            pre_path,
            f"data/images/{patient_id}/{patient_id}_{study_id}_t2w.mha"
        ),
        "params": os.path.join(script_dir, "Params_T2w.yaml"),
    },
    "adc": {
        "image": os.path.join(
            pre_path,
            f"data/images/{patient_id}/{patient_id}_{study_id}_adc.mha"
        ),
        "params": os.path.join(script_dir, "Params_ADC.yaml"),
    },
    "hbv": {
        "image": os.path.join(
            pre_path,
            f"data/images/{patient_id}/{patient_id}_{study_id}_hbv.mha"
        ),
        "params": os.path.join(script_dir, "Params_DWI.yaml"),
    },
}

##############################################################################
# HELPERS
##############################################################################

def sanitize_name(name):
    """
    Limpia nombres para evitar caracteres problemáticos dentro del zip.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


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


def resample_mask_to_image(mask_image, reference_image):
    """
    Ajusta la máscara a la geometría de la imagen de referencia.

    Esto es importante si T2W, ADC y HBV/DWI no tienen exactamente
    el mismo size, spacing, origin o direction.
    """
    same_geometry = (
        mask_image.GetSize() == reference_image.GetSize()
        and mask_image.GetSpacing() == reference_image.GetSpacing()
        and mask_image.GetOrigin() == reference_image.GetOrigin()
        and mask_image.GetDirection() == reference_image.GetDirection()
    )

    if same_geometry:
        return mask_image

    print("Re-muestreando máscara a la geometría de la imagen...")

    resampled = sitk.Resample(
        mask_image,
        reference_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    return resampled


def write_sitk_image_to_zip(zip_file, sitk_image, internal_path, tmp_dir):
    """
    Guarda una imagen SimpleITK como archivo temporal comprimido .nrrd,
    la mete al ZIP y luego borra el temporal.

    Así solo existe 1 mapa temporal en disco a la vez.
    """
    safe_name = sanitize_name(os.path.basename(internal_path))
    tmp_path = os.path.join(tmp_dir, safe_name)

    sitk.WriteImage(
        sitk_image,
        tmp_path,
        True  # useCompression=True
    )

    zip_file.write(
        tmp_path,
        arcname=internal_path
    )

    os.remove(tmp_path)


##############################################################################
# MAIN
##############################################################################

zip_output_path = os.path.join(
    output_dir,
    f"{patient_id}_{study_id}_voxel_feature_maps.zip"
)

manifest = {
    "patient_id": patient_id,
    "study_id": study_id,
    "modalities": {},
}

base_mask = sitk.ReadImage(maskFile)
base_mask = binarize_mask(base_mask)

with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:

    # ZIP_STORED porque el .nrrd ya se guarda comprimido con SimpleITK.
    # Si quieres comprimir también el ZIP, cambia ZIP_STORED por ZIP_DEFLATED.
    with zipfile.ZipFile(
        zip_output_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as zip_file:

        for modality_name, modality_config in modalities.items():

            image_path = modality_config["image"]
            params_path = modality_config["params"]

            if not os.path.exists(image_path):
                print(f"[SKIP] No existe imagen para modalidad {modality_name}: {image_path}")
                continue

            if not os.path.exists(params_path):
                print(f"[SKIP] No existe YAML para modalidad {modality_name}: {params_path}")
                continue

            print("=" * 80)
            print(f"Procesando modalidad: {modality_name}")
            print(f"Imagen: {image_path}")
            print(f"Params: {params_path}")

            image = sitk.ReadImage(image_path)

            mask = resample_mask_to_image(
                base_mask,
                image
            )

            extractor = featureextractor.RadiomicsFeatureExtractor(
                params_path
            )

            result = extractor.execute(
                image,
                mask,
                voxelBased=True
            )

            manifest["modalities"][modality_name] = {
                "image_path": image_path,
                "params_path": params_path,
                "features": [],
            }

            for feature_name, feature_value in result.items():

                if not isinstance(feature_value, sitk.Image):
                    continue

                safe_feature_name = sanitize_name(feature_name)

                internal_path = (
                    f"{patient_id}_{study_id}/"
                    f"{modality_name}/"
                    f"map_{safe_feature_name}.nrrd"
                )

                print(f"Guardando en ZIP: {internal_path}")

                write_sitk_image_to_zip(
                    zip_file=zip_file,
                    sitk_image=feature_value,
                    internal_path=internal_path,
                    tmp_dir=tmp_dir,
                )

                manifest["modalities"][modality_name]["features"].append(
                    {
                        "feature_name": feature_name,
                        "zip_path": internal_path,
                    }
                )

        zip_file.writestr(
            f"{patient_id}_{study_id}/manifest.json",
            json.dumps(manifest, indent=2)
        )

print("=" * 80)
print(f"ZIP generado:")
print(zip_output_path)