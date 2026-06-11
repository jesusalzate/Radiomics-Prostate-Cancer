#!/usr/bin/env python
"""
Extracción voxel-based de mapas radiómicos para múltiples pacientes
y múltiples modalidades usando un CSV de entrada.

Correcciones importantes:
- Desactiva GLCM_MCC porque falla con NaN/inf en voxel-based.
- Valida NaN/inf en la ROI antes de llamar a PyRadiomics.
- Guarda solamente voxeles dentro de la lesión, no toda la imagen.
- Respeta SLURM_CPUS_PER_TASK en vez de usar todos los CPUs del nodo.
- Limita threads internos de ITK, OpenBLAS, MKL y OMP.
"""

import os

##############################################################################
# LIMITAR THREADS INTERNOS ANTES DE IMPORTAR NUMPY / SITK
##############################################################################

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import SimpleITK as sitk

from tqdm import tqdm
from radiomics import featureextractor, glcm

##############################################################################
# CONFIGURACIÓN GENERAL
##############################################################################

pre_path = "/projects/ceib/data_picai"

input_csv = "./artifacts/data_2.csv"

OUTPUT_ROOT = "/projects/ceib/data_picai/data/lesion_voxel"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

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
# CONFIG SIMPLEITK
##############################################################################

try:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
except Exception:
    pass

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
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(0)

    return resample.Execute(moving_image)


def bias_field_correction(
    image_float32,
    shrink_factor=4,
    control_points=(4, 4, 4)
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


def validate_and_clean_image_in_roi(
    image,
    lesion_array,
    patient_id,
    modality,
    min_roi_voxels=10
):
    """
    Valida que la ROI tenga voxeles suficientes y valores finitos.
    Además limpia NaN/inf en la imagen antes de pasarla a PyRadiomics.
    """

    image_array = sitk.GetArrayFromImage(image).astype(np.float32)

    if image_array.shape != lesion_array.shape:
        logger.warning(
            f"[{patient_id}] Shape mismatch en {modality}: "
            f"imagen={image_array.shape}, mascara={lesion_array.shape}"
        )

        return None, None

    lesion_bool = lesion_array > 0

    roi_values = image_array[lesion_bool]

    if roi_values.size == 0:
        logger.warning(
            f"[{patient_id}] ROI vacía en modalidad {modality}"
        )

        return None, None

    finite_roi_values = roi_values[np.isfinite(roi_values)]

    if finite_roi_values.size == 0:
        logger.warning(
            f"[{patient_id}] ROI sin valores finitos en modalidad {modality}"
        )

        return None, None

    if finite_roi_values.size < min_roi_voxels:
        logger.warning(
            f"[{patient_id}] ROI muy pequeña para textura en {modality}: "
            f"{finite_roi_values.size} voxeles"
        )

        return None, None

    fill_value = float(np.median(finite_roi_values))

    max_value = float(np.max(finite_roi_values))

    min_value = float(np.min(finite_roi_values))

    image_array = np.nan_to_num(
        image_array,
        nan=fill_value,
        posinf=max_value,
        neginf=min_value
    ).astype(np.float32)

    clean_image = sitk.GetImageFromArray(image_array)

    clean_image.CopyInformation(image)

    return clean_image, lesion_bool


def clean_feature_vector(
    feature_flat,
    patient_id,
    modality,
    feature_name
):
    """
    Limpia NaN/inf en un vector de feature.
    Si toda la feature es inválida, devuelve None.
    """

    feature_flat = feature_flat.astype(np.float32)

    finite_values = feature_flat[np.isfinite(feature_flat)]

    if finite_values.size == 0:
        logger.warning(
            f"[{patient_id}] {modality} {feature_name}: "
            f"feature completamente NaN/inf, se omite"
        )

        return None

    if finite_values.size < feature_flat.size:
        fill_value = float(np.median(finite_values))

        feature_flat = np.nan_to_num(
            feature_flat,
            nan=fill_value,
            posinf=float(np.max(finite_values)),
            neginf=float(np.min(finite_values))
        ).astype(np.float32)

        logger.warning(
            f"[{patient_id}] {modality} {feature_name}: "
            f"tenía NaN/inf y fue limpiada"
        )

    return feature_flat


##############################################################################
# EXTRACTOR SEGÚN MODALIDAD
##############################################################################

def remove_unstable_features(extractor):
    """
    Elimina features inestables en voxel-based.

    GLCM_MCC puede fallar porque internamente calcula autovalores.
    Si la matriz contiene NaN/inf, numpy.linalg.eigvals revienta.
    """

    unstable_features = {
        "glcm": {"MCC"}
    }

    class_feature_getters = {
        "glcm": glcm.RadiomicsGLCM.getFeatureNames
    }

    for class_name, features_to_remove in unstable_features.items():

        if class_name not in extractor.enabledFeatures:
            continue

        enabled_features = extractor.enabledFeatures[class_name]

        if enabled_features is None or len(enabled_features) == 0:
            all_features = class_feature_getters[class_name]()

            enabled_features = [
                feature_name
                for feature_name, deprecated in all_features.items()
                if not deprecated
            ]

        cleaned_features = [
            feature_name
            for feature_name in enabled_features
            if feature_name not in features_to_remove
        ]

        if len(cleaned_features) == 0:
            extractor.enabledFeatures.pop(class_name, None)
        else:
            extractor.enabledFeatures[class_name] = cleaned_features

    return extractor


def get_extractor(modality):

    if modality == "T2":

        extractor = featureextractor.RadiomicsFeatureExtractor(
            PARAMS_T2
        )

    elif modality == "ADC":

        extractor = featureextractor.RadiomicsFeatureExtractor(
            PARAMS_ADC
        )

    elif modality == "DWI":

        extractor = featureextractor.RadiomicsFeatureExtractor(
            PARAMS_DWI
        )

    else:

        raise ValueError(
            f"Modalidad desconocida: {modality}"
        )

    extractor = remove_unstable_features(extractor)

    return extractor


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
    Extrae mapas voxel-based para una modalidad y devuelve un DataFrame
    donde cada fila corresponde a un voxel de la lesión.
    """

    try:

        ######################################################################
        # PATH IMAGEN
        ######################################################################

        image_path = os.path.join(
            pre_path,
            str(image_rel_path)
        )

        if not os.path.isfile(image_path):

            logger.warning(
                f"[{patient_id}] Imagen no encontrada: {image_path}"
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

        lesion_mask_resampled = binarize_mask(
            lesion_mask_resampled
        )

        ######################################################################
        # VALIDAR MÁSCARA
        ######################################################################

        lesion_array = sitk.GetArrayFromImage(
            lesion_mask_resampled
        )

        if np.sum(lesion_array) == 0:

            logger.warning(
                f"[{patient_id}] Máscara vacía tras resampling "
                f"en modalidad {modality}"
            )

            return None

        ######################################################################
        # VALIDAR Y LIMPIAR IMAGEN
        ######################################################################

        image, lesion_bool = validate_and_clean_image_in_roi(
            image=image,
            lesion_array=lesion_array,
            patient_id=patient_id,
            modality=modality
        )

        if image is None or lesion_bool is None:
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

        lesion_voxel_count = int(np.sum(lesion_bool))

        feature_dict["voxel_index"] = np.arange(
            lesion_voxel_count,
            dtype=np.int32
        )

        for feature_name, feature_value in result.items():

            if not isinstance(feature_value, sitk.Image):
                continue

            feature_array = sitk.GetArrayFromImage(
                feature_value
            )

            if feature_array.shape != lesion_bool.shape:
                logger.warning(
                    f"[{patient_id}] {modality} {feature_name}: "
                    f"shape mismatch feature={feature_array.shape}, "
                    f"mask={lesion_bool.shape}. Se omite."
                )

                continue

            feature_flat = feature_array[lesion_bool]

            feature_flat = clean_feature_vector(
                feature_flat=feature_flat,
                patient_id=patient_id,
                modality=modality,
                feature_name=feature_name
            )

            if feature_flat is None:
                continue

            column_name = f"map_{feature_name}"

            feature_dict[column_name] = feature_flat

            saved_count += 1

        ######################################################################
        # DATAFRAME
        ######################################################################

        if saved_count == 0:

            logger.warning(
                f"[{patient_id}] No se generaron features válidas para {modality}"
            )

            return None

        df_features = pd.DataFrame(feature_dict)

        logger.info(
            f"[{patient_id}] {modality}: "
            f"{saved_count} mapas guardados sobre "
            f"{lesion_voxel_count} voxeles de lesión"
        )

        return df_features

    except Exception as e:

        logger.error(
            f"[{patient_id}] Error en modalidad {modality}: {e}",
            exc_info=True
        )

        return None

    finally:

        gc.collect()


##############################################################################
# PROCESAMIENTO DE PACIENTE
##############################################################################

def process_row(row):
    """
    Procesa un paciente completo.
    """

    try:

        patient_id = row["patient_id"]

        label_val = int(row["case_csPCa"])

        logger.info(
            f"Procesando paciente {patient_id}"
        )

        ######################################################################
        # SI NO HAY LESIÓN -> OMITIR
        ######################################################################

        if label_val == 0:

            logger.info(
                f"[{patient_id}] Sin lesión (case_csPCa=0)"
            )

            return

        ######################################################################
        # LOAD LESION MASK
        ######################################################################

        lesion_mask_path = os.path.join(
            pre_path,
            str(row["csPCa_lesion_delineation_path"])
        )

        if not os.path.isfile(lesion_mask_path):

            logger.warning(
                f"[{patient_id}] Máscara no encontrada: {lesion_mask_path}"
            )

            return

        try:

            lesion_mask = sitk.ReadImage(
                lesion_mask_path
            )

            lesion_mask = binarize_mask(
                lesion_mask
            )

            lesion_array = sitk.GetArrayFromImage(
                lesion_mask
            )

            if np.sum(lesion_array) == 0:

                logger.warning(
                    f"[{patient_id}] Máscara vacía"
                )

                return

        except Exception as e:

            logger.error(
                f"[{patient_id}] Error leyendo máscara: {e}",
                exc_info=True
            )

            return

        ######################################################################
        # MODALIDADES
        ######################################################################

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

        ######################################################################
        # OUTPUT PATIENT DIR
        ######################################################################

        patient_output_dir = os.path.join(
            OUTPUT_ROOT,
            str(patient_id)
        )

        os.makedirs(patient_output_dir, exist_ok=True)

        ######################################################################
        # PROCESAR MODALIDADES
        ######################################################################

        for modality, rel_path in modalities:

            if pd.isna(rel_path):

                logger.warning(
                    f"[{patient_id}] Path vacío para {modality}"
                )

                continue

            csv_output_path = os.path.join(
                patient_output_dir,
                f"{modality}.csv"
            )

            ##################################################################
            # EXTRAER FEATURES
            ##################################################################

            df_features = process_modality(
                modality=modality,
                image_rel_path=rel_path,
                lesion_mask=lesion_mask,
                patient_id=patient_id
            )

            if df_features is None:

                continue

            ##################################################################
            # GUARDAR CSV
            ##################################################################

            df_features.to_csv(
                csv_output_path,
                index=False
            )

            logger.info(
                f"[{patient_id}] CSV guardado: {csv_output_path}"
            )

    except Exception as e:

        logger.error(
            f"Error procesando paciente: {e}",
            exc_info=True
        )

    finally:

        gc.collect()


##############################################################################
# MAIN
##############################################################################

def get_max_workers(total_rows):
    """
    Define número de workers respetando Slurm.
    """

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus is not None:

        try:
            max_workers = int(slurm_cpus)
        except ValueError:
            max_workers = 4

    else:

        max_workers = min(
            multiprocessing.cpu_count(),
            4
        )

    max_workers = max(1, max_workers)

    max_workers = min(max_workers, total_rows)

    return max_workers


def main():

    logger.info(
        "Iniciando extracción voxel-based"
    )

    ##########################################################################
    # LOAD CSV
    ##########################################################################

    if not os.path.isfile(input_csv):

        raise FileNotFoundError(
            f"No se encontró input_csv: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    logger.info(
        f"Pacientes encontrados: {len(df)}"
    )

    ##########################################################################
    # VALIDAR COLUMNAS
    ##########################################################################

    required_columns = [
        "patient_id",
        "case_csPCa",
        "csPCa_lesion_delineation_path",
        "t2w_path",
        "adc_path",
        "hbv_path"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if len(missing_columns) > 0:

        raise ValueError(
            f"Faltan columnas en el CSV: {missing_columns}"
        )

    ##########################################################################
    # SOLO PACIENTES CON LESIÓN
    ##########################################################################

    df = df[df["case_csPCa"] == 1].copy()

    logger.info(
        f"Número de pacientes con lesiones significativas: {len(df)}"
    )

    if len(df) == 0:

        logger.warning(
            "No hay pacientes con case_csPCa=1. Finalizando."
        )

        return

    ##########################################################################
    # MULTIPROCESSING
    ##########################################################################

    max_workers = get_max_workers(
        total_rows=len(df)
    )

    logger.info(
        f"Usando {max_workers} workers"
    )

    rows = [
        row
        for _, row in df.iterrows()
    ]

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(process_row, row): idx
            for idx, row in enumerate(rows)
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