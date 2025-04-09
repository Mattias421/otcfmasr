import whisper
import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def process_file(model, audio_path, output_emb_path, output_txt_path, device):
    """
    Processes a single audio file to extract Whisper embedding and transcription.

    Args:
        model: The loaded Whisper model.
        audio_path (str): Path to the input WAV file.
        output_emb_path (str): Path to save the output embedding (.pt).
        output_txt_path (str): Path to save the output transcription (.txt).
        device (str): The device to use ('cuda' or 'cpu').
    """
    try:
        # 1. Load audio
        #    Using transcribe directly simplifies loading and processing
        #    but we need mel separately for embed_audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # 2. Calculate log-Mel spectrogram for embedding
        #    Process the whole audio
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)

        # 3. Get audio embedding
        #    embed_audio returns shape (n_segments, n_ctx, d_model) or (n_segments, d_model)
        #    Let's run the encoder directly for potentially better full-file representation
        #    or use embed_audio and aggregate. Using embed_audio as per example.
        #    Result shape: [n_segments, D_MODEL]
        with torch.no_grad():
             # Pad or trim is not strictly needed here if we want the whole file's embedding
             # But let's keep mel calculation consistent
            audio_features = model.embed_audio(mel[None, :, :]) # Shape: [n_segments, D_MODEL]

            # Aggregate embeddings (e.g., mean pooling over segments)
            # If only one segment, this just removes the dimension
            # aggregated_embedding = audio_features.mean(dim=0) # Shape: [D_MODEL]

        # 4. Transcribe audio for text
        #    Using transcribe is robust as it handles language detection etc.
        #    It might recompute the mel spec internally, but it's convenient.
        #    Provide the path directly for simplicity.
        result = model.transcribe(audio_path)
        transcription = result['text']

        # 5. Save embedding
        #    Ensure output directory exists
        Path(output_emb_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(audio_features.cpu(), output_emb_path) # Save on CPU

        # 6. Save transcription
        #    Ensure output directory exists
        Path(output_txt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)

        return True # Indicate success

    except Exception as e:
        print(f"\n[!] Error processing {audio_path}: {e}")
        # Optional: Clean up partially created files if needed
        if Path(output_emb_path).exists():
            os.remove(output_emb_path)
        if Path(output_txt_path).exists():
            os.remove(output_txt_path)
        return False # Indicate failure

def main(args):
    """
    Main function to orchestrate the dataset processing.
    """
    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)
    output_emb_dir = output_base / "audio_emb"
    output_txt_dir = output_base / "txt"

    if not input_base.is_dir():
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    print(f"Loading Whisper model: {args.model_name}...")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU.")
        device = "cpu"
    elif device is None:
         device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    try:
        model = whisper.load_model(args.model_name, device=device, download_root="./.cache")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Please ensure 'openai-whisper' and its dependencies (like ffmpeg) are installed correctly.")
        return

    splits = ["train", "test", "valid"]
    conditions = ["noisy", "clean"]

    print(f"Starting processing from: {input_base}")
    print(f"Output will be saved to: {output_base}")

    total_files_processed = 0
    total_files_skipped = 0
    total_errors = 0

    for split in splits:
        for condition in conditions:
            current_input_dir = input_base / split / condition
            current_output_emb_dir = output_emb_dir / split / condition
            current_output_txt_dir = output_txt_dir / split / condition

            if not current_input_dir.is_dir():
                print(f"Warning: Directory not found, skipping: {current_input_dir}")
                continue

            print(f"\nProcessing: {split}/{condition}")

            # Find all .wav files
            wav_files = sorted(list(current_input_dir.glob("*.wav")))
            if not wav_files:
                print(f"No .wav files found in {current_input_dir}")
                continue

            # Create output dirs for the current split/condition batch
            # Doing it here avoids repeated checks in the loop
            current_output_emb_dir.mkdir(parents=True, exist_ok=True)
            current_output_txt_dir.mkdir(parents=True, exist_ok=True)

            progress_bar = tqdm(wav_files, desc=f"{split}/{condition}", unit="file")
            for wav_path in progress_bar:
                file_stem = wav_path.stem
                output_emb_path = current_output_emb_dir / f"{file_stem}.pt"
                output_txt_path = current_output_txt_dir / f"{file_stem}.txt"

                # Check if both output files already exist and skip if requested
                if not args.force_overwrite and output_emb_path.exists() and output_txt_path.exists():
                    total_files_skipped += 1
                    continue

                # Update progress bar description
                progress_bar.set_postfix_str(f"Processing {wav_path.name}", refresh=True)

                # Process the file
                success = process_file(model, str(wav_path), str(output_emb_path), str(output_txt_path), device)

                if success:
                    total_files_processed += 1
                else:
                    total_errors += 1

                # Clear postfix after processing
                progress_bar.set_postfix_str("", refresh=True)


    print("\n--------------------")
    print("Processing Summary:")
    print(f"  Total files processed successfully: {total_files_processed}")
    print(f"  Total files skipped (already exist): {total_files_skipped}")
    print(f"  Total errors encountered: {total_errors}")
    print("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Whisper embeddings and transcriptions from VB+DMD dataset.")
    parser.add_argument("input_dir", type=str, help="Path to the root VB+DMD dataset directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory (e.g., VB+DMD_wspr).")
    parser.add_argument("--model_name", type=str, default="base",
                        choices=["turbo", "tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en"],
                        help="Name of the Whisper model to use (default: base).")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                        help="Device to use ('cuda' or 'cpu'). Autodetects if not specified.")
    parser.add_argument("--force_overwrite", action="store_true",
                        help="Overwrite existing output files instead of skipping.")

    args = parser.parse_args()
    main(args)
