import os
import json
import jiwer
import re
from main import process_contract

# Point this to wherever your dataset and JSON live
DATASET_FOLDER = "test data"  
LABELS_FILE = "labels.json"

def main():
    labels_path = os.path.join(DATASET_FOLDER, LABELS_FILE)
    
    if not os.path.exists(labels_path):
        print(f"❌ Could not find {labels_path}. Did you generate the dataset?")
        return

    with open(labels_path, 'r', encoding='utf-8') as f:
        dataset_labels = json.load(f)

    total_wer = 0.0
    total_cer = 0.0
    processed_count = 0

    print(f"🚀 Starting Evaluation on {len(dataset_labels)} files...\n")

    for item in dataset_labels:
        filename = item["filename"]
        ground_truth = item["ground_truth"]
        
        # Skip files that haven't been manually transcribed yet
        if ground_truth == "[REQUIRES MANUAL TRANSCRIPTION]":
            print(f"⏭️ Skipping {filename} (No ground truth available)")
            continue
            
        file_path = os.path.join(DATASET_FOLDER, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {filename} is in the JSON but missing from the folder. Skipping.")
            continue

        print(f"⏳ Evaluating: {filename} (Quality: {item.get('quality', 'unknown')})")
        
        try:
            # Run your actual OCR pipeline
            ocr_output = process_contract(file_path, return_structured=False)
            
            # ---------------------------------------------------------
            # CLEANUP: Arabic Normalization & Noise Removal
            # ---------------------------------------------------------
            def normalize_arabic(text):
                # 1. Catch ALL dot variations (standard dots, underscores, and ellipses)
                text = re.sub(r'[\.\-_…]{2,}', ' ', text)
                # 2. Remove bullet points and asterisks
                text = re.sub(r'[•*\-]', ' ', text)
                # 3. Remove Arabic tatweel (elongation)
                text = re.sub(r'ـ', '', text)
                
                # 4. Strip Arabic Diacritics (Tashkeel - Fatha, Kasra, Damma, Tanween)
                tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
                text = re.sub(tashkeel, '', text)
                
                # 5. Unify Alef (Convert أ, إ, آ to simple ا)
                text = re.sub(r'[أإآ]', 'ا', text)
                
                # 6. Unify Ta Marbuta and Ha (Convert ة to ه)
                text = re.sub(r'ة', 'ه', text)
                
                # 7. Unify Ya and Alif Maksura (Convert ى to ي)
                text = re.sub(r'ى', 'ي', text)
                
                # 8. Standardize messy punctuation
                text = re.sub(r'([:،/()\[\]])', r' \1 ', text)
                
                # 9. Collapse all newlines and multiple spaces into a single space
                text = re.sub(r'\s+', ' ', text)
                
                return text.strip()

            # Apply the aggressive normalizer to BOTH the output and the ground truth
            ocr_output_clean = normalize_arabic(ocr_output)
            ground_truth_clean = normalize_arabic(ground_truth)
            
            if not ocr_output_clean: ocr_output_clean = "EMPTY_OUTPUT"
            if not ground_truth_clean: ground_truth_clean = "EMPTY_GROUND_TRUTH"

            # ========================================================
            # DEBUG PRINT: Let's see exactly what Jiwer is comparing
            # ========================================================
            print("\n--- GROUND TRUTH ---")
            print(ground_truth_clean[:500]) # Print first 500 chars to avoid terminal spam
            print("\n--- OCR OUTPUT ---")
            print(ocr_output_clean[:500])
            print("--------------------\n")

            # Calculate Error Rates on the cleaned text
            cer = jiwer.cer(ground_truth_clean, ocr_output_clean)
            wer = jiwer.wer(ground_truth_clean, ocr_output_clean)

            # Apply the aggressive normalizer to BOTH the output and the ground truth
            ocr_output_clean = normalize_arabic(ocr_output)
            ground_truth_clean = normalize_arabic(ground_truth)
            
            if not ocr_output_clean: ocr_output_clean = "EMPTY_OUTPUT"
            if not ground_truth_clean: ground_truth_clean = "EMPTY_GROUND_TRUTH"

            # Calculate Error Rates on the cleaned text
            cer = jiwer.cer(ground_truth_clean, ocr_output_clean)
            wer = jiwer.wer(ground_truth_clean, ocr_output_clean)
            
            total_cer += cer
            total_wer += wer
            processed_count += 1
            
            print(f"   -> CER: {cer * 100:.2f}% | WER: {wer * 100:.2f}%")
            
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    # Calculate and print final averages
    if processed_count > 0:
        avg_cer = total_cer / processed_count
        avg_wer = total_wer / processed_count
        avg_accuracy = max(0, (1 - avg_cer) * 100) # Prevents negative accuracy on catastrophic failures
        
        print("\n" + "=" * 50)
        print(f"📊 FINAL EVALUATION METRICS ({processed_count} files tested)")
        print("=" * 50)
        print(f"Average Word Error Rate (WER): {avg_wer * 100:.2f}%")
        print(f"Average Character Error Rate (CER): {avg_cer * 100:.2f}%")
        print(f"✅ Overall Dataset Accuracy: {avg_accuracy:.2f}%")
    else:
        print("\n⚠️ No files were successfully evaluated. Check your dataset.")

if __name__ == "__main__":
    main()