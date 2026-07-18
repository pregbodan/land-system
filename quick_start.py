#!/usr/bin/env python3
"""
Quick Start Script for Land Matter Prediction System
Run this script to execute the complete pipeline
"""

import subprocess
import sys
from pathlib import Path
import time

def print_banner(text):
    """Print a nice banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    print(f"  Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"  ✓ Completed in {elapsed:.1f} seconds\n")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"  ✗ Failed after {elapsed:.1f} seconds")
        print(f"  Error: {result.stderr}")
        return False

def check_installation():
    """Check if required packages are installed"""
    print_banner("CHECKING INSTALLATION")
    
    required_packages = [
        'sklearn', 'xgboost', 'pandas', 'numpy',
        'nltk', 'flask', 'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed!")
    return True

def check_data():
    """Check if data directory has files"""
    data_dir = Path('data/raw')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print(f"❌ No data found in {data_dir}")
        print("Please add your PDF/DOCX judgment files to data/raw/")
        return False
    
    pdf_files = list(data_dir.glob('*.pdf'))
    docx_files = list(data_dir.glob('*.docx'))
    total_files = len(pdf_files) + len(docx_files)
    
    if total_files == 0:
        print(f"❌ No judgment files found in {data_dir}")
        print("Please add your PDF/DOCX judgment files to data/raw/")
        return False
    
    print(f"✓ Found {total_files} judgment files ({len(pdf_files)} PDF, {len(docx_files)} DOCX)")
    return True

def main():
    """Main function"""
    print_banner("LAND MATTER PREDICTION SYSTEM - QUICK START")
    
    # Step 1: Check installation
    if not check_installation():
        sys.exit(1)
    
    # Step 2: Check data
    print_banner("CHECKING DATA")
    if not check_data():
        sys.exit(1)
    
    # Step 3: Ask user what to do
    print_banner("SELECT OPERATION")
    print("1. Run complete pipeline (preprocessing + training)")
    print("2. Run preprocessing only")
    print("3. Run training only (requires preprocessed data)")
    print("4. Make prediction on new file")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Complete pipeline
        print_banner("RUNNING COMPLETE PIPELINE")
        
        # Preprocessing
        if not run_command(
            [sys.executable, 'src/main_preprocessing.py'],
            "Step 1/2: Preprocessing judgments"
        ):
            print("❌ Preprocessing failed!")
            sys.exit(1)
        
        # Training
        if not run_command(
            [sys.executable, 'src/main_training.py'],
            "Step 2/2: Training models"
        ):
            print("❌ Training failed!")
            sys.exit(1)
        
        print_banner("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("Models saved to: data/models/")
        print("Best model: data/models/best_model.pkl")
        print("\nTo make predictions, run:")
        print("  python src/predict.py --file path/to/judgment.pdf")
    
    elif choice == '2':
        # Preprocessing only
        print_banner("RUNNING PREPROCESSING")
        
        if run_command(
            [sys.executable, 'src/main_preprocessing.py'],
            "Preprocessing judgments"
        ):
            print_banner("✅ PREPROCESSING COMPLETED!")
            print("Preprocessed data saved to: data/processed/")
            print("\nNext step: Run training")
            print("  python src/main_training.py")
        else:
            print("❌ Preprocessing failed!")
            sys.exit(1)
    
    elif choice == '3':
        # Training only
        print_banner("RUNNING TRAINING")
        
        # Check if preprocessed data exists
        if not Path('data/processed/preprocessed_data.pkl').exists():
            print("❌ Preprocessed data not found!")
            print("Please run preprocessing first (option 1 or 2)")
            sys.exit(1)
        
        if run_command(
            [sys.executable, 'src/main_training.py'],
            "Training models"
        ):
            print_banner("✅ TRAINING COMPLETED!")
            print("Models saved to: data/models/")
        else:
            print("❌ Training failed!")
            sys.exit(1)
    
    elif choice == '4':
        # Make prediction
        print_banner("MAKE PREDICTION")
        
        # Check if model exists
        if not Path('data/models/best_model.pkl').exists():
            print("❌ Trained model not found!")
            print("Please run complete pipeline first (option 1)")
            sys.exit(1)
        
        file_path = input("Enter path to judgment file (PDF or DOCX): ").strip()
        
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
        
        run_command(
            [sys.executable, 'src/predict.py', '--file', file_path],
            f"Predicting outcome for {file_path}"
        )
    
    elif choice == '5':
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice. Please run again and select 1-5.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
