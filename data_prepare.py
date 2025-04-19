import json
import os
import torchaudio

from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_data(
    data_folder, model_id, save_json_train, save_json_valid, save_json_test
):
    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    emb_folder = os.path.join(data_folder, "VB+DMD", f"{model_id}")

    if not os.path.exists(emb_folder):
        logger.info("f{emb_folder} doesn't exist, computing audio embeddings")
        wav_folder = os.path.join(data_folder, "VB+DMD")

        from .get_wspr_emb import get_wspr_emb

        get_wspr_emb(
            input_dir=wav_folder,
            output_dir=emb_folder,
            model_name=model_id,
            device_arg="cuda",
            force_overwrite=True,
        )
    else:
        logger.info(f"{model_id} embeddings found in {emb_folder}")

    train_folder = os.path.join(emb_folder, "audio_emb", "train", "clean")
    valid_folder = os.path.join(emb_folder, "audio_emb", "valid", "clean")
    test_folder = os.path.join(emb_folder, "audio_emb", "test", "clean")

    # List files and create manifest from list
    logger.info(f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}")
    extension = [".pt"]
    wav_list_train = get_all_files(train_folder, match_and=extension)
    wav_list_valid = get_all_files(valid_folder, match_and=extension)
    wav_list_test = get_all_files(test_folder, match_and=extension)
    create_json(wav_list_train, save_json_train)
    create_json(wav_list_valid, save_json_valid)
    create_json(wav_list_test, save_json_test)


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
