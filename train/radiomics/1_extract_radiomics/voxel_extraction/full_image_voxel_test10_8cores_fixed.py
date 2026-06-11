#!/usr/bin/env python3
"""
Extracción voxel-based de mapas radiómicos sobre IMAGEN COMPLETA.

Salida:
    OUTPUT_ROOT/<patient_id>/<MODALITY>.csv

Cada CSV queda con una fila por voxel de la imagen completa de esa modalidad:
    voxel_index, voxel_z, voxel_y, voxel_x, image_value, map_<feature_1>, ...

Diferencia clave frente a versiones lesion-only:
    - NO usa la máscara de lesión para seleccionar voxeles.
    - Crea una máscara casi completa para que PyRadiomics calcule mapas sobre toda la imagen.
    - Flatten = array completo.reshape(-1), no feature[lesion_bool].
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import logging
import multiprocessing
import sys
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from radiomics import featureextractor, glcm

PRE_PATH = os.environ.get("PRE_PATH", "/projects/ceib/data_picai")
INPUT_CSV = os.environ.get("INPUT_CSV", "./artifacts/data_2.csv")
OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    "/projects/ceib/data_picai/data/full_image_voxel"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_T2 = os.path.join(SCRIPT_DIR, "..", "Params_T2w.yaml")
PARAMS_ADC = os.path.join(SCRIPT_DIR, "..", "Params_ADC.yaml")
PARAMS_DWI = os.path.join(SCRIPT_DIR, "..", "Params_DWI.yaml")

FORCE_REPROCESS = os.environ.get("FORCE_REPROCESS", "0") == "1"
PROCESS_ONLY_LESIONS = os.environ.get("PROCESS_ONLY_LESIONS", "1") == "1"
PYRAD_VERBOSE = os.environ.get("PYRAD_VERBOSE", "0") == "1"
SAVE_COORDS = os.environ.get("SAVE_COORDS", "1") == "1"
SAVE_IMAGE_VALUE = os.environ.get("SAVE_IMAGE_VALUE", "1") == "1"
COMPRESS_CSV = os.environ.get("COMPRESS_CSV", "0") == "1"
DISABLE_GLCM_MCC = os.environ.get("DISABLE_GLCM_MCC", "1") == "1"

try:
    MAX_PATIENTS = int(os.environ.get("MAX_PATIENTS", "0"))
except ValueError:
    MAX_PATIENTS = 0

try:
    DIFFUSION_TIMESTEP = float(os.environ.get("DIFFUSION_TIMESTEP", "0.01875"))
except ValueError:
    DIFFUSION_TIMESTEP = 0.01875


class OnlyInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO


logger = logging.getLogger("FullImageVoxelRadiomics")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.addFilter(OnlyInfoFilter())
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(
    os.path.join(OUTPUT_ROOT, "full_image_voxel_warnings_errors.log"),
    mode="a"
)
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if not PYRAD_VERBOSE:
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)
    logging.getLogger("radiomics.featureextractor").setLevel(logging.ERROR)

try:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
except Exception:
    pass


@contextmanager
def suppress_c_stdout_stderr(enabled=True):
    if not enabled:
        yield
        return

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def make_full_mask_like(image):
    """
    Crea una máscara casi completa con etiqueta 1 y un único voxel de fondo 0.

    PyRadiomics NO acepta una máscara compuesta solo por 1s.
    Necesita encontrar fondo 0 y etiqueta 1.
    """
    arr = sitk.GetArrayFromImage(image)
    mask_arr = np.ones(arr.shape, dtype=np.uint8)

    if mask_arr.size < 2:
        raise ValueError(
            "La imagen es demasiado pequeña para crear una máscara válida para PyRadiomics"
        )

    mask_arr[0, 0, 0] = 0

    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(image)

    return mask


def clean_image_global(image):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        raise ValueError("La imagen no tiene valores finitos")

    if finite.size < arr.size:
        fill_value = float(np.median(finite))
        arr = np.nan_to_num(
            arr,
            nan=fill_value,
            posinf=float(np.max(finite)),
            neginf=float(np.min(finite))
        ).astype(np.float32)

        clean = sitk.GetImageFromArray(arr)
        clean.CopyInformation(image)
        return clean

    return image


def bias_field_correction(image_float32, shrink_factor=4, control_points=(4, 4, 4)):
    try:
        shrinked_image = sitk.Shrink(
            image_float32,
            [shrink_factor] * image_float32.GetDimension()
        )

        bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
        bias_field_filter.SetNumberOfControlPoints(control_points)
        bias_field_filter.UseMaskLabelOff()

        with suppress_c_stdout_stderr(enabled=not PYRAD_VERBOSE):
            bias_field_filter.Execute(shrinked_image)

        log_bias_field = bias_field_filter.GetLogBiasFieldAsImage(image_float32)
        corrected_image = image_float32 / sitk.Exp(log_bias_field)

        return corrected_image

    except Exception as e:
        logger.warning(
            f"N4 falló. Se usa imagen sin corrección N4. "
            f"Error: {type(e).__name__}: {e}"
        )
        return image_float32


def preprocess_image(image):
    image_float32 = sitk.Cast(image, sitk.sitkFloat32)
    image_float32 = clean_image_global(image_float32)

    bias_corrected = bias_field_correction(image_float32)

    try:
        with suppress_c_stdout_stderr(enabled=not PYRAD_VERBOSE):
            denoised = sitk.CurvatureAnisotropicDiffusion(
                bias_corrected,
                timeStep=DIFFUSION_TIMESTEP
            )

        return clean_image_global(denoised)

    except Exception as e:
        logger.warning(
            f"CurvatureAnisotropicDiffusion falló. Se usa imagen sin denoising. "
            f"Error: {type(e).__name__}: {e}"
        )
        return clean_image_global(bias_corrected)


def resample_to_reference(moving_image, reference_image, default_value=np.nan):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputPixelType(sitk.sitkFloat32)

    try:
        resample.SetDefaultPixelValue(float(default_value))
    except Exception:
        resample.SetDefaultPixelValue(0.0)

    return resample.Execute(moving_image)


def feature_image_to_full_array(feature_image, reference_image, full_shape):
    feature_array = sitk.GetArrayFromImage(feature_image).astype(np.float32)

    if tuple(feature_array.shape) == tuple(full_shape):
        return feature_array

    try:
        resampled = resample_to_reference(
            moving_image=sitk.Cast(feature_image, sitk.sitkFloat32),
            reference_image=reference_image,
            default_value=np.nan
        )

        resampled_array = sitk.GetArrayFromImage(resampled).astype(np.float32)

        if tuple(resampled_array.shape) == tuple(full_shape):
            return resampled_array

    except Exception:
        return None

    return None


def remove_unstable_features(extractor):
    if not DISABLE_GLCM_MCC:
        return extractor

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

        if cleaned_features:
            extractor.enabledFeatures[class_name] = cleaned_features
        else:
            extractor.enabledFeatures.pop(class_name, None)

    return extractor


def get_extractor(modality):
    if modality == "T2":
        params_path = PARAMS_T2
    elif modality == "ADC":
        params_path = PARAMS_ADC
    elif modality == "DWI":
        params_path = PARAMS_DWI
    else:
        raise ValueError(f"Modalidad desconocida: {modality}")

    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"No existe YAML de parámetros: {params_path}")

    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    extractor = remove_unstable_features(extractor)

    return extractor


def output_csv_path(patient_output_dir, modality):
    suffix = ".csv.gz" if COMPRESS_CSV else ".csv"
    return os.path.join(patient_output_dir, f"{modality}{suffix}")


def process_modality(modality, image_rel_path, patient_id):
    image_path = os.path.join(PRE_PATH, str(image_rel_path))

    if not os.path.isfile(image_path):
        logger.warning(f"[{patient_id}] {modality}: imagen no encontrada: {image_path}")
        return None

    try:
        image = sitk.ReadImage(image_path)
        image = preprocess_image(image)

        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        full_shape = image_array.shape
        total_voxels = int(image_array.size)

        full_mask = make_full_mask_like(image)
        extractor = get_extractor(modality)

        with suppress_c_stdout_stderr(enabled=not PYRAD_VERBOSE):
            result = extractor.execute(
                image,
                full_mask,
                voxelBased=True
            )

        feature_dict = {
            "voxel_index": np.arange(total_voxels, dtype=np.int64)
        }

        if SAVE_COORDS:
            z_idx, y_idx, x_idx = np.indices(full_shape, dtype=np.int32)
            feature_dict["voxel_z"] = z_idx.reshape(-1)
            feature_dict["voxel_y"] = y_idx.reshape(-1)
            feature_dict["voxel_x"] = x_idx.reshape(-1)
            del z_idx, y_idx, x_idx

        if SAVE_IMAGE_VALUE:
            feature_dict["image_value"] = image_array.reshape(-1)

        saved_count = 0
        skipped_non_image = 0
        skipped_shape = 0
        skipped_invalid = 0

        for feature_name, feature_value in result.items():
            if not isinstance(feature_value, sitk.Image):
                skipped_non_image += 1
                continue

            full_feature_array = feature_image_to_full_array(
                feature_image=feature_value,
                reference_image=image,
                full_shape=full_shape
            )

            if full_feature_array is None or tuple(full_feature_array.shape) != tuple(full_shape):
                skipped_shape += 1
                continue

            flat = full_feature_array.reshape(-1).astype(np.float32)

            if np.isinf(flat).any():
                flat = flat.copy()
                flat[np.isinf(flat)] = np.nan

            if np.all(np.isnan(flat)):
                skipped_invalid += 1
                continue

            feature_dict[f"map_{feature_name}"] = flat
            saved_count += 1

            del full_feature_array, flat

        if saved_count == 0:
            logger.warning(
                f"[{patient_id}] {modality}: no se generaron mapas válidos. "
                f"shape={full_shape}, voxeles={total_voxels}, "
                f"omitidas_shape={skipped_shape}, "
                f"invalidas={skipped_invalid}, "
                f"no_imagen={skipped_non_image}"
            )
            return None

        df_features = pd.DataFrame(feature_dict)

        logger.info(
            f"[{patient_id}] {modality}: "
            f"shape={full_shape}, voxeles={total_voxels}, mapas={saved_count}"
        )

        return df_features

    except Exception as e:
        logger.warning(
            f"[{patient_id}] {modality}: error {type(e).__name__}: {e}",
            exc_info=True
        )
        return None

    finally:
        gc.collect()


def process_row(row):
    patient_id = row.get("patient_id", "UNKNOWN")

    try:
        modalities = [
            ("T2", row.get("t2w_path")),
            ("ADC", row.get("adc_path")),
            ("DWI", row.get("hbv_path")),
        ]

        patient_output_dir = os.path.join(OUTPUT_ROOT, str(patient_id))
        os.makedirs(patient_output_dir, exist_ok=True)

        saved_modalities = []
        skipped_modalities = []

        for modality, rel_path in modalities:
            if rel_path is None or pd.isna(rel_path):
                skipped_modalities.append(modality)
                logger.warning(f"[{patient_id}] {modality}: path vacío")
                continue

            csv_path = output_csv_path(patient_output_dir, modality)

            if (not FORCE_REPROCESS) and os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
                logger.info(
                    f"[{patient_id}] {modality}: ya existe, se omite por FORCE_REPROCESS=0"
                )
                saved_modalities.append(modality)
                continue

            df_features = process_modality(modality, rel_path, patient_id)

            if df_features is None:
                skipped_modalities.append(modality)
                continue

            if COMPRESS_CSV:
                df_features.to_csv(csv_path, index=False, compression="gzip")
            else:
                df_features.to_csv(csv_path, index=False)

            logger.info(f"[{patient_id}] {modality}: CSV guardado en {csv_path}")
            saved_modalities.append(modality)

            del df_features
            gc.collect()

        status = "ok" if saved_modalities else "sin_modalidades_guardadas"

        return {
            "patient_id": patient_id,
            "status": status,
            "saved_modalities": ",".join(saved_modalities),
            "skipped_modalities": ",".join(skipped_modalities),
        }

    except Exception as e:
        logger.warning(
            f"[{patient_id}] error procesando paciente: {type(e).__name__}: {e}",
            exc_info=True
        )
        return {
            "patient_id": patient_id,
            "status": "patient_error",
            "saved_modalities": "",
            "skipped_modalities": "T2,ADC,DWI",
        }

    finally:
        gc.collect()


def get_max_workers(total_rows):
    env_workers = os.environ.get("MAX_WORKERS")

    if env_workers is not None:
        try:
            max_workers = int(env_workers)
        except ValueError:
            max_workers = 1
    else:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

        if slurm_cpus is not None:
            try:
                max_workers = min(int(slurm_cpus), 8)
            except ValueError:
                max_workers = 1
        else:
            max_workers = min(multiprocessing.cpu_count(), 8)

    max_workers = max(1, max_workers)
    max_workers = min(max_workers, total_rows)

    return max_workers


def run_sequential(rows):
    results = []

    for row in tqdm(rows, total=len(rows), desc="Procesando pacientes"):
        results.append(process_row(row))

    return results


def run_parallel(rows, max_workers):
    results = []

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_row, row): idx
                for idx, row in enumerate(rows)
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Procesando pacientes"
            ):
                idx = futures[future]
                patient_id = rows[idx].get("patient_id", "UNKNOWN")

                try:
                    results.append(future.result())

                except BrokenProcessPool:
                    raise

                except Exception as e:
                    logger.warning(
                        f"[{patient_id}] future falló: {type(e).__name__}: {e}",
                        exc_info=True
                    )
                    results.append({
                        "patient_id": patient_id,
                        "status": "future_error",
                        "saved_modalities": "",
                        "skipped_modalities": "T2,ADC,DWI",
                    })

    except BrokenProcessPool:
        logger.warning(
            "El ProcessPool se rompió. Casi siempre es memoria insuficiente o crash nativo "
            "de SimpleITK/PyRadiomics. Reejecuta con MAX_WORKERS=1, 2 o 4."
        )
        raise

    return results


def main():
    logger.info("Iniciando extracción FULL IMAGE voxel-based")
    logger.info(f"PRE_PATH={PRE_PATH}")
    logger.info(f"INPUT_CSV={INPUT_CSV}")
    logger.info(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    logger.info(f"DIFFUSION_TIMESTEP={DIFFUSION_TIMESTEP}")
    logger.info(f"PROCESS_ONLY_LESIONS={int(PROCESS_ONLY_LESIONS)}")

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"No se encontró INPUT_CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Filas en CSV: {len(df)}")

    required_columns = [
        "patient_id",
        "t2w_path",
        "adc_path",
        "hbv_path"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing_columns}")

    if PROCESS_ONLY_LESIONS:
        if "case_csPCa" not in df.columns:
            raise ValueError("PROCESS_ONLY_LESIONS=1 pero no existe la columna case_csPCa")

        df = df[df["case_csPCa"] == 1].copy()
        logger.info(f"Pacientes filtrados con case_csPCa=1: {len(df)}")

    if MAX_PATIENTS > 0:
        df = df.head(MAX_PATIENTS).copy()
        logger.info(f"MAX_PATIENTS={MAX_PATIENTS}: se procesarán {len(df)} filas")
    else:
        logger.info(f"MAX_PATIENTS=0: se procesarán todas las filas filtradas: {len(df)}")

    if len(df) == 0:
        logger.info("No hay filas para procesar. Finalizando.")
        return

    rows = df.to_dict("records")
    max_workers = get_max_workers(len(rows))

    logger.info(f"Workers usados: {max_workers}")

    if max_workers == 1:
        results = run_sequential(rows)
    else:
        results = run_parallel(rows, max_workers)

    summary_path = os.path.join(OUTPUT_ROOT, "full_image_voxel_run_summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)

    logger.info(f"Resumen guardado en: {summary_path}")
    logger.info("Extracción FULL IMAGE voxel-based finalizada")


if __name__ == "__main__":
    main()