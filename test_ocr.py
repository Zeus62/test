import os
import glob
from main import process_contract

def main():
    test_folder = "test data" # Make sure your files are actually in here!
    
    # Added .pdf and .docx so you can test the full pipeline
    extensions = ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.pdf", "*.docx"]
    test_files = []
    for ext in extensions:
        test_files.extend(glob.glob(os.path.join(test_folder, ext)))
        
    if not test_files:
        print(f"❌ Couldn't find any files in '{test_folder}'. Check the folder name!")
        return
        
    print(f"🚀 Found {len(test_files)} files in '{test_folder}'. Firing up the Surya Engine...\n")
    
    output_file = "ocr_test_results.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, img_path in enumerate(test_files, 1):
            file_name = os.path.basename(img_path)
            print(f"⏳ [{idx}/{len(test_files)}] Processing: {file_name}...")
            
            try:
                result_text = process_contract(img_path, return_structured=False)
                
                f.write(f"{'='*60}\n")
                f.write(f"📄 DOCUMENT: {file_name}\n")
                f.write(f"{'='*60}\n")
                f.write(result_text + "\n\n")
                
                print(f"✅ Success! Extracted {len(result_text)} characters.")
                
            except Exception as e:
                print(f"❌ Pipeline crashed on {file_name}: {e}")
                f.write(f"{'='*60}\n")
                f.write(f"❌ FAILED DOCUMENT: {file_name}\n")
                f.write(f"Error: {e}\n\n")

    print(f"\n🎉 All done! Open '{output_file}' to review all the extracted text.")

if __name__ == "__main__":
    main()