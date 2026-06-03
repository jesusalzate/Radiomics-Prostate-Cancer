#!/usr/bin/env python
"""
Extracción paralela de características radiómicas usando únicamente
la máscara de lesión csPCa.

Cambios realizados:
- Se eliminó el uso de whole_gland_path.
- Ahora SOLO se usa:
    csPCa_lesion_delineation_path
- Se verifica case_csPCa:
    - 1 -> tiene lesión
    - 0 -> no tiene lesión (se omite extracción)
- Se agrega binarización de máscaras:
    - cualquier valor > 0 se convierte en 1
    - fondo permanece en 0
- Se eliminó la extracción "full".
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

##############################################################################
# CONFIGURACIÓN GENERAL
##############################################################################

pre_path = "/projects/ceib/data_picai"
input_csv = "./artifacts/data_2.csv"

current_file = os.path.abspath(__file__)

script_dir = os.path.dirname(current_file)

PARAMS_T2 = os.path.join(script_dir, "Params_T2w.yaml")

PARAMS_ADC = os.path.join(script_dir, "Params_ADC.yaml")

PARAMS_DWI = os.path.join(script_dir, "Params_DWI.yaml")

project_root = os.path.abspath(
    os.path.join(
        current_file,
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
    )
)

base_dir = os.path.join(project_root, "artifacts", "radiomics_lesion")
os.makedirs(base_dir, exist_ok=True)

##############################################################################
# OUTPUTS
##############################################################################

t2_features_csv  = os.path.join(base_dir, "features_t2_lesion.csv")
adc_features_csv = os.path.join(base_dir, "features_adc_lesion.csv")
dwi_features_csv = os.path.join(base_dir, "features_dwi_lesion.csv")

##############################################################################
# LOGGING
##############################################################################

logger = logging.getLogger("RadiomicsLesion")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

##############################################################################
# UTILIDADES
##############################################################################

def resample_to_reference(moving_image, reference_image, is_mask=False):
    """
    Remuestrea una imagen al espacio de referencia.
    """

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)

    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(moving_image)


def binarize_mask(mask_image):
    """
    Convierte cualquier valor > 0 a 1.

    Ejemplo:
        0 -> 0
        3 -> 1
        255 -> 1
    """

    mask_array = sitk.GetArrayFromImage(mask_image)

    binary_array = (mask_array > 0).astype(np.uint8)

    binary_mask = sitk.GetImageFromArray(binary_array)
    binary_mask.CopyInformation(mask_image)

    return binary_mask


def bias_field_correction(
    image_float32,
    shrink_factor=4,
    control_points=[4, 4, 4]
):
    """
    Corrección N4 Bias Field.
    """

    shrinked_image = sitk.Shrink(
        image_float32,
        [shrink_factor] * image_float32.GetDimension()
    )

    bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()

    bias_field_filter.SetNumberOfControlPoints(control_points)

    bias_field_filter.UseMaskLabelOff()

    bias_field_filter.Execute(shrinked_image)

    log_bias_field = bias_field_filter.GetLogBiasFieldAsImage(
        image_float32
    )

    corrected_image = image_float32 / sitk.Exp(log_bias_field)

    return corrected_image


def preprocess_image(image):
    """
    Preprocesamiento de imagen.
    """

    image_float32 = sitk.Cast(image, sitk.sitkFloat32)

    bias_corrected = bias_field_correction(image_float32)

    denoised = sitk.CurvatureAnisotropicDiffusion(
        bias_corrected,
        timeStep=0.01875
    )

    return denoised


##############################################################################
# EXTRACCIÓN RADIÓMICA
##############################################################################

def extract_radiomic_features(
    extractor_local,
    image_sitk,
    mask_sitk,
    patient_id,
    study_id,
    label_value
):
    """
    Extrae características radiómicas.
    """

    features = extractor_local.execute(image_sitk, mask_sitk)

    out_dict = {
        "patient_id": patient_id,
        "study_id": study_id,
        "label": label_value
    }

    for k, v in features.items():
        out_dict[k] = v

    return out_dict


##############################################################################
# PROCESAMIENTO DE MODALIDADES
##############################################################################

def process_modality(
    modality_key,
    image_rel_path,
    patient_id,
    study_id,
    label_val,
    lesion_mask
):
    """
    Procesa una modalidad.
    """

    if modality_key == "T2":
        extractor_local = featureextractor.RadiomicsFeatureExtractor(
            PARAMS_T2
        )

    elif modality_key == "ADC":
        extractor_local = featureextractor.RadiomicsFeatureExtractor(
            PARAMS_ADC
        )

    elif modality_key == "DWI":
        extractor_local = featureextractor.RadiomicsFeatureExtractor(
            PARAMS_DWI
        )

    else:
        raise ValueError("Modalidad desconocida")

    modality_img_path = os.path.join(pre_path, image_rel_path)

    if not os.path.isfile(modality_img_path):
        logger.warning(
            f"Imagen {modality_key} no encontrada para {patient_id}"
        )
        return None

    try:

        image = sitk.ReadImage(modality_img_path)

        preprocessed = preprocess_image(image)

        lesion_mask = resample_to_reference(
            lesion_mask,
            preprocessed,
            is_mask=True
        )

        features = extract_radiomic_features(
            extractor_local=extractor_local,
            image_sitk=preprocessed,
            mask_sitk=lesion_mask,
            patient_id=patient_id,
            study_id=study_id,
            label_value=label_val
        )

        return features

    except Exception as e:

        logger.error(
            f"Error procesando {modality_key} para "
            f"{patient_id}: {e}",
            exc_info=True
        )

        return None


##############################################################################
# PROCESAMIENTO POR PACIENTE
##############################################################################

def process_row(row):
    """
    Procesa un paciente.
    """

    results = {
        "t2": None,
        "adc": None,
        "dwi": None
    }

    patient_id = row["patient_id"]
    study_id = row["study_id"]

    label_val = int(row["case_csPCa"])

    ##########################################################################
    # SI NO HAY LESIÓN -> NO EXTRAER
    ##########################################################################

    if label_val == 0:

        logger.info(
            f"Paciente {patient_id} sin lesión (case_csPCa=0)"
        )

        return results

    ##########################################################################
    # CARGAR MÁSCARA DE LESIÓN
    ##########################################################################

    lesion_mask_path = os.path.join(
        pre_path,
        row["csPCa_lesion_delineation_path"]
    )

    if not os.path.isfile(lesion_mask_path):

        logger.warning(
            f"Máscara no encontrada para {patient_id}"
        )

        return results

    try:

        lesion_mask = sitk.ReadImage(lesion_mask_path)

        ######################################################################
        # BINARIZACIÓN
        ######################################################################

        lesion_mask = binarize_mask(lesion_mask)

        ######################################################################
        # VALIDAR QUE LA MÁSCARA NO ESTÉ VACÍA
        ######################################################################

        lesion_array = sitk.GetArrayFromImage(lesion_mask)

        if np.sum(lesion_array) == 0:

            logger.warning(
                f"Máscara vacía para paciente {patient_id}"
            )

            return results

    except Exception as e:

        logger.error(
            f"Error leyendo máscara para {patient_id}: {e}",
            exc_info=True
        )

        return results

    ##########################################################################
    # PROCESAR MODALIDADES
    ##########################################################################

    results["t2"] = process_modality(
        "T2",
        row["t2w_path"],
        patient_id,
        study_id,
        label_val,
        lesion_mask
    )

    results["adc"] = process_modality(
        "ADC",
        row["adc_path"],
        patient_id,
        study_id,
        label_val,
        lesion_mask
    )

    results["dwi"] = process_modality(
        "DWI",
        row["hbv_path"],
        patient_id,
        study_id,
        label_val,
        lesion_mask
    )

    return results


##############################################################################
# MAIN
##############################################################################

def main():

    logger.info(
        "Iniciando extracción radiomica de lesiones csPCa"
    )

    df = pd.read_csv(input_csv)

    t2_features = []
    adc_features = []
    dwi_features = []

    max_workers = min(4, multiprocessing.cpu_count())

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(process_row, row): idx
            for idx, row in df.iterrows()
        }

        with tqdm(
            total=len(futures),
            desc="Procesando pacientes"
        ) as pbar:

            for future in as_completed(futures):

                row_result = future.result()

                if row_result["t2"] is not None:
                    t2_features.append(row_result["t2"])

                if row_result["adc"] is not None:
                    adc_features.append(row_result["adc"])

                if row_result["dwi"] is not None:
                    dwi_features.append(row_result["dwi"])

                pbar.update(1)

    ##########################################################################
    # GUARDAR CSVs
    ##########################################################################

    pd.DataFrame(t2_features).to_csv(
        t2_features_csv,
        index=False
    )

    pd.DataFrame(adc_features).to_csv(
        adc_features_csv,
        index=False
    )

    pd.DataFrame(dwi_features).to_csv(
        dwi_features_csv,
        index=False
    )

    logger.info("Extracción completada")


if __name__ == "__main__":
    main()