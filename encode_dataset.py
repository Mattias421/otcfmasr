import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from speechbrain.inference.ASR import EncoderASR
from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper
from speechbrain.dataio.dataio import read_audio  # More robust audio reading

import torchaudio

# Define a mapping for common SpeechBrain ASR models
# Add more models here as needed
SPEECHBRAIN_MODELS = {
    "conformer-largescale": "speechbrain/asr-conformer-largescaleasr",
    "conformer-librispeech": "speechbrain/asr-conformer-transformerlm-librispeech",
    # "wav2vec2-base-960h": "speechbrain/asr-wav2vec2-commonvoice-en",
    "whisper-tiny": "openai/whisper-tiny",
    "wav2vec2-asr-base-960h": "pytorch/wav2vec2-asr-base-960h",
    # Add other relevant models if desired
}


def process_file(model, model_id, audio_path, output_emb_path, device):
    """
    Processes a single audio file to extract SpeechBrain encoder embedding and transcription.

    Args:
        model (EncoderDecoderASR): The loaded SpeechBrain ASR model.
        audio_path (str): Path to the input WAV file.
        output_emb_path (str): Path to save the output embedding (.pt).
        output_txt_path (str): Path to save the output transcription (.txt).
        device (str): The device to use ('cuda' or 'cpu').
    """
    try:
        signal = read_audio(audio_path)
        # Ensure signal is 2D [batch, time] as encode_batch expects batch dim
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        signal = signal.to(device)

        with torch.no_grad():
            if "openai" in model_id:
                mel = model._get_mel(signal)

                encoder_out = model.forward_encoder(mel)

            elif "pytorch" in model_id:
                with torch.inference_mode():
                    encoder_out, _ = model.extract_features(signal)
                encoder_out = encoder_out[-1]

            else:
                wav_lens = torch.tensor([1.0], device=device)
                # encoder_out shape: [batch, time_steps, features]
                encoder_out = model.encode_batch(signal, wav_lens)

        final_embedding = encoder_out.squeeze(0)

        # 4. Save embedding
        Path(output_emb_path).parent.mkdir(parents=True, exist_ok=True)
        # Save on CPU to avoid GPU memory issues when loading later
        torch.save(final_embedding.cpu(), output_emb_path)

        return True  # Indicate success

    except Exception as e:
        print(f"\n[!] Error processing {audio_path}: {e}")
        # Optional: Clean up partially created files
        if Path(output_emb_path).exists():
            try:
                os.remove(output_emb_path)
            except OSError:  # Handle potential race conditions or permission issues
                pass
        return False  # Indicate failure


def get_sb_features(
    input_dir, output_dir, model_id, savedir, device_arg, force_overwrite
):
    """
    Main function to orchestrate the dataset processing using SpeechBrain.

    Args:
        input_dir (str): Path to the root dataset directory (e.g., VB+DMD).
        output_dir (str): Path to the output directory (e.g., VB+DMD_sb).
        model_id (str): SpeechBrain model identifier (HuggingFace Hub ID or local path).
        savedir (str): Directory to save/load the downloaded SpeechBrain model.
        device_arg (str or None): Requested device ('cuda', 'cpu', or None for auto).
        force_overwrite (bool): Whether to overwrite existing output files.
    """
    input_base = Path(input_dir)
    output_base = Path(output_dir)
    # Consider renaming output dirs slightly to reflect SpeechBrain source
    output_emb_dir = output_base / "audio_emb"

    if not input_base.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    print(f"Loading SpeechBrain ASR model: {model_id}...")
    print(f"Model will be saved/loaded from: {savedir}")

    # Determine device
    device = device_arg
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU.")
        device = "cpu"
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    try:
        if "openai" in model_id:
            asr_model = Whisper(
                source=model_id,
                save_path=f"{savedir}/{model_id}",
                encoder_only=True,
                freeze=True,
                language="en",
            )
            asr_model.to(device)
            asr_model.eval()  # Set model to evaluation mode

        elif "pytorch" in model_id:
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            asr_model = bundle.get_model(
                dl_kwargs={"model_dir": savedir, "file_name": model_id.split("/")[-1]}
            ).to(device)

        else:
            # Load the SpeechBrain model
            asr_model = EncoderASR.from_hparams(
                source=model_id,
                savedir=f"{savedir}/{model_id}",
                run_opts={"device": device},
            )
            # Ensure model is on the correct device (from_hparams might handle this via run_opts, but explicit is safe)
            asr_model.to(device)
            asr_model.eval()  # Set model to evaluation mode

    except Exception as e:
        print(f"Error loading SpeechBrain model '{model_id}': {e}")
        print(
            "Please ensure 'speechbrain' and its dependencies are installed correctly."
        )
        print(f"Attempted to load from: {model_id}")
        return

    if input_dir.split("/")[-2] == "VB+DMD":
        # --- Define dataset structure ---
        # Adjust splits/conditions if your dataset differs
        splits = ["train", "test", "valid"]
        conditions = ["noisy", "clean"]
        # --- ---

        print(f"Starting processing from: {input_base}")
        print(f"Output will be saved to: {output_base}")

        total_files_processed = 0
        total_files_skipped = 0
        total_errors = 0

        for split in splits:
            for condition in conditions:
                current_input_dir = input_base / split / condition
                current_output_emb_dir = output_emb_dir / split / condition

                if not current_input_dir.is_dir():
                    print(
                        f"Warning: Directory not found, skipping: {current_input_dir}"
                    )
                    continue

                print(f"\nProcessing: {split}/{condition}")

                # Find all .wav files (adjust glob pattern if needed)
                wav_files = sorted(list(current_input_dir.glob("*.wav")))
                if not wav_files:
                    print(f"No .wav files found in {current_input_dir}")
                    continue

                # Create output directories for the current subset
                current_output_emb_dir.mkdir(parents=True, exist_ok=True)

                progress_bar = tqdm(
                    wav_files, desc=f"{split}/{condition}", unit="file", leave=False
                )
                for wav_path in progress_bar:
                    file_stem = wav_path.stem
                    output_emb_path = current_output_emb_dir / f"{file_stem}.pt"

                    if not force_overwrite and output_emb_path.exists():
                        total_files_skipped += 1
                        continue

                    progress_bar.set_postfix_str(
                        f"Processing {wav_path.name}", refresh=True
                    )

                    success = process_file(
                        asr_model,  # Pass the SpeechBrain model
                        model_id,
                        str(wav_path),
                        str(output_emb_path),
                        device,  # Pass the determined device
                    )

                    if success:
                        total_files_processed += 1
                    else:
                        total_errors += 1
                        # Optionally add a small delay or break on too many errors
                        # import time; time.sleep(0.1)

                    # Clear postfix for cleaner progress bar
                    progress_bar.set_postfix_str("", refresh=True)
                progress_bar.close()

    elif input_dir.split("/")[-2] == "LJSpeech-1.1":
        total_files_processed = 0
        total_files_skipped = 0
        total_errors = 0

        current_input_dir = input_base / "wavs"
        current_output_emb_dir = output_emb_dir

        print("\nProcessing:LJSpeech")

        # Find all .wav files (adjust glob pattern if needed)
        wav_files = sorted(list(current_input_dir.glob("*.wav")))

        # Create output directories for the current subset
        current_output_emb_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = tqdm(wav_files, desc="LJSpeech", unit="file", leave=False)
        for wav_path in progress_bar:
            file_stem = wav_path.stem
            output_emb_path = current_output_emb_dir / f"{file_stem}.pt"

            if not force_overwrite and output_emb_path.exists():
                total_files_skipped += 1
                continue

            progress_bar.set_postfix_str(f"Processing {wav_path.name}", refresh=True)

            success = process_file(
                asr_model,  # Pass the SpeechBrain model
                model_id,
                str(wav_path),
                str(output_emb_path),
                device,  # Pass the determined device
            )

            if success:
                total_files_processed += 1
            else:
                total_errors += 1
                # Optionally add a small delay or break on too many errors
                # import time; time.sleep(0.1)

            # Clear postfix for cleaner progress bar
            progress_bar.set_postfix_str("", refresh=True)
        progress_bar.close()

    breakpoint()
    print("\n--------------------")
    print("Processing Summary:")
    print(f"  Model Used: {model_id}")
    print(f"  Total files processed successfully: {total_files_processed}")
    print(f"  Total files skipped (already exist): {total_files_skipped}")
    print(f"  Total errors encountered: {total_errors}")
    print("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SpeechBrain encoder embeddings and transcriptions from an audio dataset."
    )
    parser.add_argument(
        "input_dir", type=str, help="Path to the root dataset directory (e.g., VB+DMD)."
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to the output directory (e.g., VB+DMD_sb)."
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="conformer-largescale",
        choices=list(SPEECHBRAIN_MODELS.keys()),
        help=f"Short key for the SpeechBrain model to use (default: conformer-largescale). "
        f"Available: {', '.join(SPEECHBRAIN_MODELS.keys())}",
    )
    parser.add_argument(
        "--model_id_override",
        type=str,
        default=None,
        help="Full SpeechBrain model ID (HuggingFace Hub ID or local path) to override the --model_key selection. e.g., 'speechbrain/asr-wav2vec2-commonvoice-en'",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="./pretrained_models",  # Changed default for clarity
        help="Directory to save/load downloaded SpeechBrain models (default: ./pretrained_models).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use ('cuda' or 'cpu'). Autodetects if not specified.",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping.",
    )

    args = parser.parse_args()

    # Determine the final model ID to use
    if args.model_id_override:
        model_identifier = args.model_id_override
        print(f"Using overridden model ID: {model_identifier}")
    elif args.model_key in SPEECHBRAIN_MODELS:
        model_identifier = SPEECHBRAIN_MODELS[args.model_key]
        print(f"Using model key '{args.model_key}': {model_identifier}")
    else:
        # This case should ideally not be reached due to 'choices' in parser, but good practice
        print(f"Error: Invalid model key '{args.model_key}' and no override provided.")
        exit(1)

    # Create the savedir if it doesn't exist
    Path(args.savedir).mkdir(parents=True, exist_ok=True)

    get_sb_features(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_id=model_identifier,
        savedir=args.savedir,
        device_arg=args.device,
        force_overwrite=args.force_overwrite,
    )
