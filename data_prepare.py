import json
import os
import torchaudio
import random
import shutil

from pathlib import Path
import math
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

PAIRING_MODE_PAIRED = "paired"
PAIRING_MODE_SCRAMBLED = "scrambled"
PAIRING_MODE_SPEAKER_SPLIT = "speaker_split"
PAIRING_MODE_LJSPEECH = "ljspeech"


def prepare_data(
    data_folder,
    model_id,
    save_json_train,
    save_json_valid,
    save_json_test,
    pairing_mode=PAIRING_MODE_PAIRED,
):
    """
    Prepares the JSON manifests for the VoiceBank+DEMAND dataset embeddings.

    Supports multiple pairing modes for the training set:
    1. "paired": Standard noisy/clean VBD pairs.
    2. "scrambled": Noisy VBD samples paired with randomly chosen clean VBD samples.
    3. "speaker_split": VBD Speakers split; noisy from group A paired with clean
                       from group B, and vice-versa.
    4. "ljspeech": Noisy VBD samples paired with randomly chosen clean LJSpeech samples.

    Arguments
    ---------
    data_folder : str
        Path to the base folder containing dataset directories like "VB+DMD"
        and potentially "LJSpeech-1.1".
    model_id : str
        Identifier for the embedding model used (e.g., "Wav2Vec2-base").
    save_json_train : str
        Path to save the training data manifest (.json file).
    save_json_valid : str
        Path to save the validation data manifest (.json file).
    save_json_test : str
        Path to save the test data manifest (.json file).
    pairing_mode : str, optional
        Determines how noisy/clean samples are paired in the training set.
        Options: "paired", "scrambled", "speaker_split", "ljspeech".
        Defaults to "paired".
    """
    # Check if this phase is already done (if so, skip it)
    # Basic check: if all target files exist. Might need deletion if mode changes.
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation JSON files found, skipping basic manifest creation.")
        # Decide if we should still proceed to apply pairing modes even if files exist.
        # For simplicity, let's assume if files exist, the correct mode was already applied.
        # User should delete train.json if they want to re-run with a different mode.
        # return # Uncomment this if you want strict skipping

    # --- VBD Embedding Path ---
    vbd_emb_folder = os.path.join(data_folder, "VB+DMD", f"{model_id}")

    # --- Compute VBD Embeddings if necessary ---
    if not os.path.exists(vbd_emb_folder):
        logger.info(f"{vbd_emb_folder} doesn't exist, computing VBD audio embeddings")
        vbd_wav_folder = os.path.join(data_folder, "VB+DMD")
        try:
            from .get_wspr_emb import get_wspr_emb
        except ImportError:
            logger.error("Could not import get_wspr_emb. Please ensure it's available.")
            raise
        get_wspr_emb(
            input_dir=vbd_wav_folder,
            output_dir=vbd_emb_folder,
            model_name=model_id,
            device_arg="cuda",
            force_overwrite=True,
        )
    else:
        logger.info(f"VBD {model_id} embeddings found in {vbd_emb_folder}")

    # --- Compute LJSpeech Embeddings if necessary (only if needed for pairing) ---
    ljspeech_emb_folder = None
    if pairing_mode == PAIRING_MODE_LJSPEECH:
        ljspeech_emb_folder = os.path.join(
            data_folder, "LJSpeech-1.1", f"{model_id}", "audio_emb"
        )
        if not os.path.exists(ljspeech_emb_folder):
            logger.info(
                f"{ljspeech_emb_folder} doesn't exist, computing LJSpeech audio embeddings"
            )
            ljs_wav_folder = os.path.join(
                data_folder, "LJSpeech-1.1", "wavs"
            )  # Standard LJSpeech structure
            ljs_output_base = os.path.join(data_folder, "LJSpeech-1.1", f"{model_id}")

            if not os.path.exists(ljs_wav_folder):
                logger.error(
                    f"LJSpeech wav folder not found at {ljs_wav_folder}. Cannot compute embeddings."
                )
                logger.error(
                    f"Cannot proceed with pairing mode '{PAIRING_MODE_LJSPEECH}'."
                )
                # Option 1: Fallback to default paired mode
                # logger.warning("Falling back to 'paired' mode.")
                # pairing_mode = PAIRING_MODE_PAIRED
                # Option 2: Stop execution
                raise FileNotFoundError(
                    f"Required LJSpeech WAVs not found at {ljs_wav_folder}"
                )

            try:
                from .get_wspr_emb import get_wspr_emb
            except ImportError:
                logger.error(
                    "Could not import get_wspr_emb. Please ensure it's available."
                )
                raise
            # Note: get_wspr_emb might need adaptation if LJSpeech has a flat structure
            # Assuming get_wspr_emb can handle input_dir being flat and output_dir structure
            get_wspr_emb(
                input_dir=ljs_wav_folder,
                output_dir=ljs_output_base,  # Output goes to LJSpeech/<model_id>
                model_name=model_id,
                device_arg="cuda",
                force_overwrite=True,
            )
            # Check if the specific audio_emb subfolder was created
            if not os.path.exists(ljspeech_emb_folder):
                logger.error(
                    f"LJSpeech embedding computation finished, but expected folder {ljspeech_emb_folder} was not found."
                )
                raise FileNotFoundError(
                    f"Expected LJSpeech embeddings not found at {ljspeech_emb_folder} after computation."
                )
        else:
            logger.info(
                f"LJSpeech {model_id} embeddings found in {ljspeech_emb_folder}"
            )

    # --- Define VBD data folders ---
    train_folder = os.path.join(vbd_emb_folder, "audio_emb", "train", "clean")
    valid_folder = os.path.join(vbd_emb_folder, "audio_emb", "valid", "clean")
    test_folder = os.path.join(vbd_emb_folder, "audio_emb", "test", "clean")

    # --- Create initial paired VBD manifests ---
    logger.info(
        f"Creating initial paired VBD manifests: {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".pt"]
    try:
        wav_list_train = get_all_files(train_folder, match_and=extension)
        wav_list_valid = get_all_files(valid_folder, match_and=extension)
        wav_list_test = get_all_files(test_folder, match_and=extension)
    except FileNotFoundError as e:
        logger.error(
            f"Error finding VBD embedding files (e.g., in {train_folder}). Did embedding generation complete? Details: {e}"
        )
        return

    # Create temporary paired train manifest if we are doing a special pairing mode
    # This avoids issues if the special pairing fails mid-way
    temp_train_json = save_json_train
    if pairing_mode != PAIRING_MODE_PAIRED:
        temp_train_json = save_json_train + ".tmp"

    create_json(wav_list_train, temp_train_json)  # Create paired VBD train data
    create_json(wav_list_valid, save_json_valid)  # Valid/Test are always paired VBD
    create_json(wav_list_test, save_json_test)

    # --- Apply specific pairing mode to the training set ---
    try:
        if pairing_mode == PAIRING_MODE_SCRAMBLED:
            logger.info(
                f"Applying '{PAIRING_MODE_SCRAMBLED}' pairing to {save_json_train}..."
            )
            scramble_json_pairs(temp_train_json, save_json_train)
            logger.info(f"Scrambled training data saved to {save_json_train}")
        elif pairing_mode == PAIRING_MODE_SPEAKER_SPLIT:
            logger.info(
                f"Applying '{PAIRING_MODE_SPEAKER_SPLIT}' pairing to {save_json_train}..."
            )
            speaker_split_pairing(temp_train_json, save_json_train)
            logger.info(
                f"Speaker-split paired training data saved to {save_json_train}"
            )
        elif pairing_mode == PAIRING_MODE_LJSPEECH:
            logger.info(
                f"Applying '{PAIRING_MODE_LJSPEECH}' pairing to {save_json_train}..."
            )
            if ljspeech_emb_folder and os.path.exists(ljspeech_emb_folder):
                ljspeech_pairing(temp_train_json, save_json_train, ljspeech_emb_folder)
                logger.info(f"LJSpeech paired training data saved to {save_json_train}")
            else:
                logger.error(
                    f"LJSpeech embedding folder not found or specified ({ljspeech_emb_folder}). Cannot apply LJSpeech pairing."
                )
                # Fallback: Keep the original paired VBD data
                if temp_train_json != save_json_train:
                    shutil.move(temp_train_json, save_json_train)
                logger.warning(
                    f"Using standard paired VBD data for {save_json_train} instead."
                )

        elif pairing_mode == PAIRING_MODE_PAIRED:
            logger.info(f"Using standard '{PAIRING_MODE_PAIRED}' VBD training data.")
            # If we used a temp file, rename it to the final name
            if temp_train_json != save_json_train:
                shutil.move(temp_train_json, save_json_train)
        else:
            logger.warning(
                f"Unknown pairing_mode '{pairing_mode}'. "
                f"Using standard '{PAIRING_MODE_PAIRED}' VBD data for training set."
            )
            if temp_train_json != save_json_train:
                shutil.move(temp_train_json, save_json_train)

    finally:
        # Clean up temporary file if it exists
        if pairing_mode != PAIRING_MODE_PAIRED and os.path.exists(temp_train_json):
            try:
                os.remove(temp_train_json)
            except OSError as e:
                logger.warning(
                    f"Could not remove temporary file {temp_train_json}: {e}"
                )


def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """

    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        emb_clean = os.path.join("/", *path_parts)
        path_parts[-2] = "noisy"
        emb_noisy = os.path.join("/", *path_parts)

        txt_label = os.path.join(
            "/", *path_parts[:-5], "txt", path_parts[-3], f"{uttid}.txt"
        )

        with open(txt_label, "r") as f:
            txt_label = f.read().strip()

        wav = os.path.join(
            "/", *path_parts[:-5], path_parts[-3], path_parts[-2], f"{uttid}.wav"
        )

        wav_len = torchaudio.load(wav)[0].shape[1]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wrd": txt_label,
            "path_emb_noisy": emb_noisy,
            "path_emb_clean": emb_clean,
            "wav_len": wav_len,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def scramble_json_pairs(input_json_path: str, output_json_path: str):
    """
    Reads a JSON manifest file with paired noisy/clean paths, scrambles the
    clean paths relative to the noisy paths, and writes a new scrambled JSON file.

    Assumes the input JSON is a dictionary where keys are utterance IDs and
    values are dictionaries containing at least 'path_emb_noisy' and
    'path_emb_clean' keys, along with other metadata (e.g., 'wrd', 'wav_len').

    Arguments:
    ---------
    input_json_path : str
        Path to the original JSON file (e.g., "train.json").
    output_json_path : str
        Path where the scrambled JSON file will be saved (e.g., "train_scrambled.json").
    """
    logger.info(f"Attempting to read input JSON: {input_json_path}")

    # --- Step 1: Read the input JSON file ---
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        logger.info(
            f"Successfully read {len(original_data)} entries from {input_json_path}"
        )
    except FileNotFoundError:
        logger.error(f"Input JSON file not found: {input_json_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {input_json_path}: {e}")
        return
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while reading {input_json_path}: {e}"
        )
        return

    if not isinstance(original_data, dict):
        logger.error(
            "Input JSON does not seem to be a dictionary (uttid -> details). Aborting."
        )
        return

    # --- Step 2: Extract clean paths and other data associated with noisy paths ---
    clean_paths = []
    noisy_data_associated = []  # List to store {'uttid': uttid, **other_metadata_excluding_clean_path}

    logger.info("Extracting clean paths and associated noisy data...")
    missing_keys_count = 0
    for uttid, entry_data in original_data.items():
        if not isinstance(entry_data, dict):
            logger.warning(f"Entry for uttid '{uttid}' is not a dictionary. Skipping.")
            missing_keys_count += 1
            continue

        if "path_emb_clean" not in entry_data or "path_emb_noisy" not in entry_data:
            logger.warning(
                f"Entry for uttid '{uttid}' is missing 'path_emb_clean' or 'path_emb_noisy'. Skipping."
            )
            missing_keys_count += 1
            continue

        # Store the clean path
        clean_paths.append(entry_data["path_emb_clean"])

        # Store all other data associated with this uttid/noisy path
        noisy_info = {"uttid": uttid}  # Store the uttid itself for easy reconstruction
        for key, value in entry_data.items():
            if key != "path_emb_clean":  # Exclude the clean path from this structure
                noisy_info[key] = value
        noisy_data_associated.append(noisy_info)

    num_entries = len(noisy_data_associated)
    logger.info(
        f"Extracted {num_entries} valid entries. Skipped {missing_keys_count} entries due to missing keys/format issues."
    )

    if num_entries == 0:
        logger.error("No valid entries found to process. Aborting.")
        return
    if len(clean_paths) != num_entries:
        logger.error(
            f"CRITICAL: Mismatch in extracted counts ({len(clean_paths)} clean vs {num_entries} noisy). Aborting."
        )
        return

    # --- Step 3: Shuffle the clean paths list ---
    logger.info(f"Shuffling {len(clean_paths)} clean paths...")
    random.shuffle(clean_paths)
    logger.info("Shuffling complete.")

    # --- Step 4: Reconstruct the new dictionary with scrambled pairings ---
    scrambled_json_dict = {}
    logger.info("Reconstructing JSON with scrambled pairings...")
    for i in range(num_entries):
        noisy_info = noisy_data_associated[i]
        shuffled_clean_path = clean_paths[i]  # Get the i-th shuffled clean path
        uttid = noisy_info["uttid"]  # Retrieve the original utterance ID

        # Create the new entry, starting with the noisy info (which includes noisy path, wrd, wav_len etc.)
        new_entry = noisy_info.copy()
        # Assign the shuffled clean path
        new_entry["path_emb_clean"] = shuffled_clean_path
        # Remove the redundant 'uttid' key from the value dict if you prefer
        # del new_entry['uttid']

        scrambled_json_dict[uttid] = new_entry

    # --- Step 5: Write the scrambled dictionary to the output JSON file ---
    logger.info(
        f"Attempting to write {len(scrambled_json_dict)} scrambled entries to {output_json_path}"
    )
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(scrambled_json_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully wrote scrambled JSON to: {output_json_path}")
    except IOError as e:
        logger.error(f"Failed to write JSON file {output_json_path}: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while writing {output_json_path}: {e}"
        )


def get_speaker_id_from_path(file_path):
    """
    Extracts the speaker ID (e.g., 'p226') from the filename.
    Assumes filename format: pXXX_YYY.ext

    Arguments
    ---------
    file_path : str or pathlib.Path
        The full path to the file (e.g., .pt, .wav, .txt).

    Returns
    -------
    str or None
        The extracted speaker ID (e.g., 'p226') or None if parsing fails.
    """
    try:
        # Get the filename without the extension (stem)
        filename_stem = Path(file_path).stem
        # Split by the first underscore
        parts = filename_stem.split("_", 1)
        if len(parts) > 0 and parts[0].startswith("p") and parts[0][1:].isdigit():
            # Basic check: starts with 'p' and followed by digits
            speaker_id = parts[0]
            return speaker_id
        else:
            logger.warning(
                f"Could not extract speaker ID from filename stem: '{filename_stem}'. "
                f"Expected format 'pXXX_YYY'. Path: {file_path}"
            )
            return None
    except Exception as e:
        logger.warning(
            f"Error extracting speaker ID from path: {file_path}. Error: {e}"
        )
        return None


def speaker_split_pairing(input_json_path: str, output_json_path: str):
    """
    Reads a JSON manifest, splits speakers into two groups, and creates new
    pairings where noisy samples from Group A are paired with clean samples
    from Group B, and vice-versa. Writes the result to a new JSON file.

    Arguments:
    ---------
    input_json_path : str
        Path to the original paired JSON file (e.g., "train.json").
    output_json_path : str
        Path where the speaker-split paired JSON file will be saved.
    """
    logger.info(f"Attempting speaker-split pairing for: {input_json_path}")

    # --- Step 1: Read the input JSON file ---
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        logger.info(
            f"Successfully read {len(original_data)} entries from {input_json_path}"
        )
    except FileNotFoundError:
        logger.error(f"Input JSON file not found: {input_json_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {input_json_path}: {e}")
        return
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while reading {input_json_path}: {e}"
        )
        return

    if not isinstance(original_data, dict):
        logger.error(
            "Input JSON does not seem to be a dictionary (uttid -> details). Aborting speaker split."
        )
        return

    # --- Step 2: Group entries by speaker ID ---
    speakers_data = {}
    logger.info("Grouping entries by speaker...")
    missing_keys_count = 0
    speaker_extraction_errors = 0
    processed_count = 0

    for uttid, entry_data in original_data.items():
        if (
            not isinstance(entry_data, dict)
            or "path_emb_clean" not in entry_data
            or "path_emb_noisy" not in entry_data
        ):
            logger.warning(
                f"Skipping entry {uttid} due to missing keys or incorrect format."
            )
            missing_keys_count += 1
            continue

        # Extract speaker ID from one of the paths (e.g., clean path)
        speaker_id = get_speaker_id_from_path(entry_data["path_emb_clean"])
        if speaker_id is None:
            logger.warning(
                f"Could not determine speaker ID for entry {uttid}. Skipping."
            )
            speaker_extraction_errors += 1
            continue

        if speaker_id not in speakers_data:
            speakers_data[speaker_id] = []

        # Store the full entry data along with the uttid for later reconstruction
        speakers_data[speaker_id].append({"uttid": uttid, **entry_data})
        processed_count += 1

    logger.info(f"Processed {processed_count} valid entries.")
    if missing_keys_count > 0:
        logger.warning(
            f"Skipped {missing_keys_count} entries due to missing keys/format."
        )
    if speaker_extraction_errors > 0:
        logger.warning(
            f"Skipped {speaker_extraction_errors} entries due to speaker ID extraction errors."
        )

    if len(speakers_data) < 2:
        logger.error(
            f"Found {len(speakers_data)} speakers. Need at least 2 speakers for speaker-split pairing. Aborting."
        )
        return

    # --- Step 3: Split speakers into two groups ---
    speaker_ids = list(speakers_data.keys())
    random.shuffle(speaker_ids)
    split_point = math.ceil(len(speaker_ids) / 2)  # Split roughly in half
    group_a_speakers = speaker_ids[:split_point]
    group_b_speakers = speaker_ids[split_point:]
    logger.info(
        f"Split {len(speaker_ids)} speakers into Group A ({len(group_a_speakers)}) and Group B ({len(group_b_speakers)})."
    )

    # --- Step 4: Collect all entries for each group ---
    group_a_entries = []
    for spk_id in group_a_speakers:
        group_a_entries.extend(speakers_data[spk_id])

    group_b_entries = []
    for spk_id in group_b_speakers:
        group_b_entries.extend(speakers_data[spk_id])

    logger.info(
        f"Group A has {len(group_a_entries)} utterances. Group B has {len(group_b_entries)} utterances."
    )

    if not group_a_entries or not group_b_entries:
        logger.error(
            "One of the speaker groups has no utterances. Cannot perform split pairing. Aborting."
        )
        return

    # --- Step 5: Extract noisy data and clean paths for pairing ---
    # Group A Noisy Data + Other Info (uttid, wrd, len, noisy_path)
    group_a_noisy_data = [
        {
            "uttid": e["uttid"],
            "data": {k: v for k, v in e.items() if k != "path_emb_clean"},
        }
        for e in group_a_entries
    ]
    # Group B Clean Paths
    group_b_clean_paths = [e["path_emb_clean"] for e in group_b_entries]

    # Group B Noisy Data + Other Info
    group_b_noisy_data = [
        {
            "uttid": e["uttid"],
            "data": {k: v for k, v in e.items() if k != "path_emb_clean"},
        }
        for e in group_b_entries
    ]
    # Group A Clean Paths
    group_a_clean_paths = [e["path_emb_clean"] for e in group_a_entries]

    # --- Step 6: Shuffle the clean path lists ---
    random.shuffle(group_a_clean_paths)
    random.shuffle(group_b_clean_paths)
    logger.info("Shuffled clean paths within each group.")

    # --- Step 7: Create new pairings ---
    new_json_dict = {}
    num_pairs_a_noisy_b_clean = min(len(group_a_noisy_data), len(group_b_clean_paths))
    num_pairs_b_noisy_a_clean = min(len(group_b_noisy_data), len(group_a_clean_paths))

    logger.info(
        f"Creating {num_pairs_a_noisy_b_clean} pairs (Group A Noisy -> Group B Clean)."
    )
    for i in range(num_pairs_a_noisy_b_clean):
        noisy_info = group_a_noisy_data[i]
        uttid = noisy_info["uttid"]
        new_entry = noisy_info["data"].copy()  # Contains noisy path, wrd, len etc.
        new_entry["path_emb_clean"] = group_b_clean_paths[
            i
        ]  # Assign shuffled clean path from other group
        new_json_dict[uttid] = new_entry

    logger.info(
        f"Creating {num_pairs_b_noisy_a_clean} pairs (Group B Noisy -> Group A Clean)."
    )
    for i in range(num_pairs_b_noisy_a_clean):
        noisy_info = group_b_noisy_data[i]
        uttid = noisy_info["uttid"]
        # Check if uttid already exists (shouldn't if uttid are unique across speakers)
        if uttid in new_json_dict:
            logger.warning(
                f"Utterance ID {uttid} encountered again when pairing B->A. Overwriting previous A->B pair. This might indicate non-unique uttid across speakers."
            )
        new_entry = noisy_info["data"].copy()
        new_entry["path_emb_clean"] = group_a_clean_paths[
            i
        ]  # Assign shuffled clean path from other group
        new_json_dict[uttid] = new_entry

    total_pairs = len(new_json_dict)
    logger.info(f"Total utterances in the final speaker-split JSON: {total_pairs}")

    # --- Step 8: Write the new JSON file ---
    logger.info(
        f"Attempting to write {total_pairs} speaker-split paired entries to {output_json_path}"
    )
    try:
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(new_json_dict, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Successfully wrote speaker-split paired JSON to: {output_json_path}"
        )
    except IOError as e:
        logger.error(f"Failed to write JSON file {output_json_path}: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while writing {output_json_path}: {e}"
        )


def ljspeech_pairing(
    input_vbd_json_path: str, output_json_path: str, ljspeech_emb_folder: str
):
    """
    Reads a VBD JSON manifest, finds LJSpeech embeddings, and creates new
    pairings where noisy VBD samples are paired with randomly selected clean
    LJSpeech samples. Writes the result to a new JSON file.

    Arguments:
    ---------
    input_vbd_json_path : str
        Path to the original paired VBD training JSON file (e.g., "train.json.tmp").
    output_json_path : str
        Path where the LJSpeech-paired JSON file will be saved (e.g., "train.json").
    ljspeech_emb_folder : str
        Path to the directory containing LJSpeech audio embeddings (.pt files).
        Example: /data/LJSpeech-1.1/wav2vec2-base/audio_emb/
    """
    logger.info("Starting LJSpeech pairing process.")
    logger.info(f"Reading VBD noisy data from: {input_vbd_json_path}")
    logger.info(f"Reading LJSpeech clean embeddings from: {ljspeech_emb_folder}")

    # --- Step 1: Read the input VBD JSON file ---
    try:
        with open(input_vbd_json_path, "r", encoding="utf-8") as f:
            original_vbd_data = json.load(f)
        logger.info(
            f"Successfully read {len(original_vbd_data)} VBD entries from {input_vbd_json_path}"
        )
    except FileNotFoundError:
        logger.error(f"Input VBD JSON file not found: {input_vbd_json_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {input_vbd_json_path}: {e}")
        return
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while reading {input_vbd_json_path}: {e}"
        )
        return

    if not isinstance(original_vbd_data, dict) or not original_vbd_data:
        logger.error(
            "Input VBD JSON is not a non-empty dictionary. Aborting LJSpeech pairing."
        )
        return

    # --- Step 2: Extract noisy VBD data ---
    noisy_vbd_entries = []
    skipped_vbd_count = 0
    for uttid, entry_data in original_vbd_data.items():
        if isinstance(entry_data, dict) and "path_emb_noisy" in entry_data:
            # Keep all original VBD data except the clean path
            noisy_info = {k: v for k, v in entry_data.items() if k != "path_emb_clean"}
            noisy_vbd_entries.append({"uttid": uttid, "data": noisy_info})
        else:
            logger.warning(
                f"Skipping VBD entry {uttid} due to missing 'path_emb_noisy' or incorrect format."
            )
            skipped_vbd_count += 1

    num_noisy_vbd = len(noisy_vbd_entries)
    if skipped_vbd_count > 0:
        logger.warning(f"Skipped {skipped_vbd_count} VBD entries.")
    if num_noisy_vbd == 0:
        logger.error("No valid noisy VBD entries found. Aborting.")
        return
    logger.info(f"Extracted {num_noisy_vbd} noisy VBD entries.")

    # --- Step 3: Find LJSpeech embedding files ---
    try:
        ljspeech_emb_files = get_all_files(ljspeech_emb_folder, match_and=[".pt"])
        num_ljspeech = len(ljspeech_emb_files)
        logger.info(
            f"Found {num_ljspeech} LJSpeech embedding files in {ljspeech_emb_folder}"
        )
    except FileNotFoundError:
        logger.error(
            f"LJSpeech embedding folder not found: {ljspeech_emb_folder}. Cannot perform LJSpeech pairing."
        )
        return
    except Exception as e:
        logger.error(f"Error listing LJSpeech embedding files: {e}")
        return

    if num_ljspeech == 0:
        logger.error(
            f"No LJSpeech embedding files (.pt) found in {ljspeech_emb_folder}. Aborting."
        )
        return

    # --- Step 4: Select LJSpeech clean paths ---
    # We need exactly `num_noisy_vbd` clean paths.
    if num_ljspeech < num_noisy_vbd:
        logger.warning(
            f"Found fewer LJSpeech embeddings ({num_ljspeech}) than noisy VBD samples ({num_noisy_vbd}). "
            f"LJSpeech samples will be reused (sampled with replacement)."
        )
        # Sample with replacement to get the exact number needed
        selected_ljspeech_paths = random.choices(ljspeech_emb_files, k=num_noisy_vbd)
    else:
        logger.info(f"Selecting {num_noisy_vbd} unique LJSpeech embeddings.")
        # Shuffle the list and take the first num_noisy_vbd items
        random.shuffle(ljspeech_emb_files)
        selected_ljspeech_paths = ljspeech_emb_files[:num_noisy_vbd]

    # --- Step 5: Reconstruct the new dictionary with LJSpeech pairings ---
    new_json_dict = {}
    logger.info("Reconstructing JSON with VBD noisy -> LJSpeech clean pairings...")
    for i in range(num_noisy_vbd):
        vbd_entry = noisy_vbd_entries[i]
        uttid = vbd_entry["uttid"]
        new_entry_data = vbd_entry[
            "data"
        ].copy()  # Get the noisy VBD data (path, wrd, len)

        # Assign the selected LJSpeech path as the clean path
        new_entry_data["path_emb_clean"] = selected_ljspeech_paths[i]

        new_json_dict[uttid] = new_entry_data

    # --- Step 6: Write the new JSON file ---
    logger.info(
        f"Attempting to write {len(new_json_dict)} LJSpeech-paired entries to {output_json_path}"
    )
    try:
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(new_json_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully wrote LJSpeech-paired JSON to: {output_json_path}")
    except IOError as e:
        logger.error(f"Failed to write JSON file {output_json_path}: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while writing {output_json_path}: {e}"
        )


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    *filenames : tuple
        The file paths passed here should already exist
        in order for the preparation to be considered done.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True
