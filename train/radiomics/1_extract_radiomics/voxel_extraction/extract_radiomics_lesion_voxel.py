#!/usr/bin/env python
"""
Extracción voxel-based de mapas radiómicos para múltiples pacientes
y múltiples modalidades usando un CSV de entrada.

MODIFICACIÓN:
- Ya NO guarda .nrrd individuales.
- Cada mapa radiomico se:
    1. convierte a numpy
    2. flatten()
    3. se guarda como columna en CSV

Cada CSV:
- contiene TODAS las imágenes de un paciente
- agrupadas por modalidad
- sin índices

Estructura final:

/projects/ceib/data_picai/data/lesion_voxel/
    └── patient_id/
        ├── T2.csv
        ├── ADC.csv
        └── DWI.csv

Cada columna:
- nombre = nombre original del mapa
- valores = voxeles flattenizados

"""

import os
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import SimpleITK as sitk

from tqdm import tqdm
from radiomics import featureextractor

##############################################################################
# CONFIGURACIÓN GENERAL
##############################################################################

pre_path = "/projects/ceib/data_picai"

input_csv = "./artifacts/data_2.csv"

##############################################################################
# PATHS DEL SCRIPT
##############################################################################

current_file = os.path.abspath(__file__)

script_dir = os.path.dirname(current_file)

##############################################################################
# YAMLs
##############################################################################

PARAMS_T2 = os.path.join(
    script_dir,
    "..",
    "Params_T2w.yaml"
)

PARAMS_ADC = os.path.join(
    script_dir,
    "..",
    "Params_ADC.yaml"
)

PARAMS_DWI = os.path.join(
    script_dir,
    "..",
    "Params_DWI.yaml"
)

##############################################################################
# OUTPUT ROOT
##############################################################################

OUTPUT_ROOT = (
    "/projects/ceib/data_picai/data/lesion_voxel"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

##############################################################################
# LOGGING
##############################################################################

logger = logging.getLogger("VoxelRadiomics")

logger.setLevel(logging.INFO)

logger.propagate = False

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)

##############################################################################
# UTILIDADES
##############################################################################

def binarize_mask(mask_image):
    """
    Convierte cualquier valor > 0 a 1.
    """

    mask_array = sitk.GetArrayFromImage(mask_image)

    binary_array = (mask_array > 0).astype(np.uint8)

    binary_mask = sitk.GetImageFromArray(binary_array)

    binary_mask.CopyInformation(mask_image)

    return binary_mask


def resample_to_reference(
    moving_image,
    reference_image,
    is_mask=False
):
    """
    Remuestrea una imagen al espacio de referencia.
    """

    resample = sitk.ResampleImageFilter()

    resample.SetReferenceImage(reference_image)

    if is_mask:
        resample.SetInterpolator(
            sitk.sitkNearestNeighbor
        )
    else:
        resample.SetInterpolator(
            sitk.sitkLinear
        )

    return resample.Execute(moving_image)


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

    bias_field_filter = (
        sitk.N4BiasFieldCorrectionImageFilter()
    )

    bias_field_filter.SetNumberOfControlPoints(
        control_points
    )

    bias_field_filter.UseMaskLabelOff()

    bias_field_filter.Execute(shrinked_image)

    log_bias_field = (
        bias_field_filter.GetLogBiasFieldAsImage(
            image_float32
        )
    )

    corrected_image = (
        image_float32 / sitk.Exp(log_bias_field)
    )

    return corrected_image


def preprocess_image(image):
    """
    Preprocesamiento de imagen.
    """

    image_float32 = sitk.Cast(
        image,
        sitk.sitkFloat32
    )

    bias_corrected = bias_field_correction(
        image_float32
    )

    denoised = sitk.CurvatureAnisotropicDiffusion(
        bias_corrected,
        timeStep=0.01875
    )

    return denoised


##############################################################################
# EXTRACTOR SEGÚN MODALIDAD
##############################################################################

def get_extractor(modality):

    if modality == "T2":

        return featureextractor.RadiomicsFeatureExtractor(
            PARAMS_T2
        )

    elif modality == "ADC":

        return featureextractor.RadiomicsFeatureExtractor(
            PARAMS_ADC
        )

    elif modality == "DWI":

        return featureextractor.RadiomicsFeatureExtractor(
            PARAMS_DWI
        )

    else:

        raise ValueError(
            f"Modalidad desconocida: {modality}"
        )


##############################################################################
# PROCESAMIENTO DE UNA MODALIDAD
##############################################################################

def process_modality(
    modality,
    image_rel_path,
    lesion_mask,
    patient_id
):
    """
    Extrae mapas voxel-based para una modalidad y
    devuelve un DataFrame flattenizado.
    """

    try:

        ######################################################################
        # PATH IMAGEN
        ######################################################################

        image_path = os.path.join(
            pre_path,
            image_rel_path
        )

        if not os.path.isfile(image_path):

            logger.warning(
                f"[{patient_id}] "
                f"Imagen no encontrada: {image_path}"
            )

            return None

        ######################################################################
        # LOAD IMAGE
        ######################################################################

        image = sitk.ReadImage(image_path)

        ######################################################################
        # PREPROCESS
        ######################################################################

        image = preprocess_image(image)

        ######################################################################
        # RESAMPLE MASK
        ######################################################################

        lesion_mask_resampled = resample_to_reference(
            lesion_mask,
            image,
            is_mask=True
        )

        ######################################################################
        # VALIDAR MÁSCARA
        ######################################################################

        lesion_array = sitk.GetArrayFromImage(
            lesion_mask_resampled
        )

        if np.sum(lesion_array) == 0:

            logger.warning(
                f"[{patient_id}] "
                f"Máscara vacía tras resampling "
                f"en modalidad {modality}"
            )

            return None

        ######################################################################
        # EXTRACTOR
        ######################################################################

        extractor = get_extractor(modality)

        ######################################################################
        # VOXEL-BASED EXTRACTION
        ######################################################################

        result = extractor.execute(
            image,
            lesion_mask_resampled,
            voxelBased=True
        )

        ######################################################################
        # CREAR DICCIONARIO DE COLUMNAS
        ######################################################################

        feature_dict = {}

        saved_count = 0

        for feature_name, feature_value in result.items():

            if isinstance(feature_value, sitk.Image):

                ##################################################################
                # Convertir imagen a numpy
                ##################################################################

                feature_array = sitk.GetArrayFromImage(
                    feature_value
                )

                ##################################################################
                # Flatten
                ##################################################################

                feature_flat = feature_array.flatten()

                ##################################################################
                # Nombre columna
                ##################################################################

                column_name = (
                    f"map_{feature_name}"
                )

                feature_dict[column_name] = pd.Series(feature_flat)

                saved_count += 1

        ######################################################################
        # DATAFRAME
        ######################################################################

        if len(feature_dict) == 0:

            logger.warning(
                f"[{patient_id}] "
                f"No se generaron features para {modality}"
            )

            return None

        df_features = pd.DataFrame(feature_dict)

        logger.info(
            f"[{patient_id}] "
            f"{modality}: "
            f"{saved_count} mapas flattenizados"
        )

        return df_features

    except Exception as e:

        logger.error(
            f"[{patient_id}] "
            f"Error en modalidad {modality}: {e}",
            exc_info=True
        )

        return None


##############################################################################
# PROCESAMIENTO DE PACIENTE
##############################################################################

def process_row(row):
    """
    Procesa un paciente completo.
    """

    patient_id = row["patient_id"]

    label_val = int(row["case_csPCa"])

    logger.info(
        f"Procesando paciente {patient_id}"
    )

    ##########################################################################
    # SI NO HAY LESIÓN -> OMITIR
    ##########################################################################

    if label_val == 0:

        logger.info(
            f"[{patient_id}] "
            f"Sin lesión (case_csPCa=0)"
        )

        return

    ##########################################################################
    # LOAD LESION MASK
    ##########################################################################

    lesion_mask_path = os.path.join(
        pre_path,
        row["csPCa_lesion_delineation_path"]
    )

    if not os.path.isfile(lesion_mask_path):

        logger.warning(
            f"[{patient_id}] "
            f"Máscara no encontrada"
        )

        return

    try:

        lesion_mask = sitk.ReadImage(
            lesion_mask_path
        )

        ######################################################################
        # BINARIZAR
        ######################################################################

        lesion_mask = binarize_mask(
            lesion_mask
        )

        ######################################################################
        # VALIDAR MÁSCARA
        ######################################################################

        lesion_array = sitk.GetArrayFromImage(
            lesion_mask
        )

        if np.sum(lesion_array) == 0:

            logger.warning(
                f"[{patient_id}] "
                f"Máscara vacía"
            )

            return

    except Exception as e:

        logger.error(
            f"[{patient_id}] "
            f"Error leyendo máscara: {e}",
            exc_info=True
        )

        return

    ##########################################################################
    # MODALIDADES
    ##########################################################################

    modalities = [
        (
            "T2",
            row["t2w_path"]
        ),
        (
            "ADC",
            row["adc_path"]
        ),
        (
            "DWI",
            row["hbv_path"]
        )
    ]

    ##########################################################################
    # OUTPUT PATIENT DIR
    ##########################################################################

    patient_output_dir = os.path.join(
        OUTPUT_ROOT,
        str(patient_id)
    )

    os.makedirs(patient_output_dir, exist_ok=True)

    ##########################################################################
    # PROCESAR MODALIDADES
    ##########################################################################

    for modality, rel_path in modalities:

        if pd.isna(rel_path):

            logger.warning(
                f"[{patient_id}] "
                f"Path vacío para {modality}"
            )

            continue

        ######################################################################
        # EXTRAER FEATURES
        ######################################################################

        df_features = process_modality(
            modality=modality,
            image_rel_path=rel_path,
            lesion_mask=lesion_mask,
            patient_id=patient_id
        )

        if df_features is None:

            continue

        ######################################################################
        # CSV PATH
        ######################################################################

        csv_output_path = os.path.join(
            patient_output_dir,
            f"{modality}.csv"
        )

        ######################################################################
        # GUARDAR CSV
        ######################################################################

        df_features.to_csv(
            csv_output_path,
            index=False
        )

        logger.info(
            f"[{patient_id}] "
            f"CSV guardado: {csv_output_path}"
        )


##############################################################################
# MAIN
##############################################################################

def main():

    logger.info(
        "Iniciando extracción voxel-based"
    )

    ##########################################################################
    # LOAD CSV
    ##########################################################################

    df = pd.read_csv(input_csv)

    logger.info(
        f"Pacientes encontrados: {len(df)}"
    )

    ##########################################################################
    # SOLO PACIENTES CON LESIÓN
    ##########################################################################

    df = df[df["case_csPCa"] == 1]

    logger.info(
        f"Numero de pacientes con lesiones significativas {len(df)}"
    )

    ##########################################################################
    # MULTIPROCESSING
    ##########################################################################

    max_workers = multiprocessing.cpu_count()

    logger.info(
        f"Usando {max_workers} workers"
    )

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

                try:

                    future.result()

                except Exception as e:

                    logger.error(
                        f"Error en future: {e}",
                        exc_info=True
                    )

                pbar.update(1)

    logger.info(
        "Extracción voxel-based finalizada"
    )


##############################################################################
# ENTRYPOINT
##############################################################################

if __name__ == "__main__":

    main()