"""
Text extraction module for PDF and DOCX files
"""
import PyPDF2
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract text from various document formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc']
    
    def extract_from_pdf(self, pdf_path):
        """
        Extract text from PDF file using pdfplumber (primary) or PyPDF2 (fallback)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            str: Extracted text
        """
        text = ""
        
        try:
            # Method 1: pdfplumber (preferred - better text extraction)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            logger.info(f"Successfully extracted text from {pdf_path} using pdfplumber")
            
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}. Trying PyPDF2...")
            
            try:
                # Method 2: PyPDF2 (fallback)
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                logger.info(f"Successfully extracted text from {pdf_path} using PyPDF2")
                
            except Exception as e2:
                logger.error(f"Both extraction methods failed for {pdf_path}: {e2}")
                raise
        
        return text
    
    def extract_from_docx(self, docx_path):
        """
        Extract text from Word document
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            str: Extracted text
        """
        try:
            doc = Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info(f"Successfully extracted text from {docx_path}")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {docx_path}: {e}")
            raise
    
    def extract_with_ocr(self, pdf_path):
        """
        Extract text from scanned PDF using OCR
        Use this only if regular extraction fails or returns very little text
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            str: Extracted text via OCR
        """
        try:
            from pdf2image import convert_from_path
            
            logger.info(f"Attempting OCR extraction for {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            text = ""
            for i, image in enumerate(images):
                # Apply OCR to each page
                page_text = pytesseract.image_to_string(image)
                text += f"\n--- Page {i+1} ---\n{page_text}\n"
            
            logger.info(f"Successfully extracted text from {pdf_path} using OCR")
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            raise
    
    def extract(self, file_path):
        """
        Extract text from file (auto-detect format)
        
        Args:
            file_path: Path to document file
            
        Returns:
            str: Extracted text
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            text = self.extract_from_pdf(file_path)
            
            # Check if extraction was successful (enough text)
            if len(text.strip()) < 100:
                logger.warning(f"PDF extraction yielded little text. Trying OCR...")
                text = self.extract_with_ocr(file_path)
            
            return text
            
        elif suffix in ['.docx', '.doc']:
            return self.extract_from_docx(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def validate_extraction(self, text):
        """
        Validate that text extraction was successful
        
        Args:
            text: Extracted text
            
        Returns:
            tuple: (is_valid, message)
        """
        # Check minimum length
        if len(text) < 500:
            return False, "Text too short - extraction may have failed"
        
        # Check for gibberish (too many special characters)
        if len(text) > 0:
            special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
            if special_char_ratio > 0.4:
                return False, "Too many special characters - possible OCR error"
        
        # Check for common legal keywords
        keywords = ['court', 'plaintiff', 'defendant', 'judgment', 'appeal', 'land']
        if not any(keyword.lower() in text.lower() for keyword in keywords):
            return False, "Missing common legal keywords"
        
        return True, "Extraction successful"


def batch_extract(file_list, output_dir=None):
    """
    Extract text from multiple files
    
    Args:
        file_list: List of file paths
        output_dir: Optional directory to save extracted text files
        
    Returns:
        dict: {file_path: extracted_text}
    """
    extractor = TextExtractor()
    results = {}
    
    for file_path in file_list:
        try:
            logger.info(f"Processing: {file_path}")
            text = extractor.extract(file_path)
            
            is_valid, message = extractor.validate_extraction(text)
            if is_valid:
                results[str(file_path)] = text
                logger.info(f"✓ {file_path}: {message}")
                
                # Optionally save to text file
                if output_dir:
                    output_path = Path(output_dir) / f"{Path(file_path).stem}.txt"
                    output_path.write_text(text, encoding='utf-8')
            else:
                logger.warning(f"✗ {file_path}: {message}")
                results[str(file_path)] = None
                
        except Exception as e:
            logger.error(f"✗ Failed to process {file_path}: {e}")
            results[str(file_path)] = None
    
    logger.info(f"Extraction complete. Successful: {sum(1 for v in results.values() if v is not None)}/{len(file_list)}")
    
    return results


if __name__ == "__main__":
    # Example usage
    extractor = TextExtractor()
    
    # Test with a single file
    # text = extractor.extract("data/raw/sample_judgment.pdf")
    # is_valid, message = extractor.validate_extraction(text)
    # print(f"Validation: {message}")
    # print(f"Text length: {len(text)} characters")
    # print(f"First 500 characters:\n{text[:500]}")
    
    print("Text extraction module loaded successfully.")
    print("To use: from text_extraction import TextExtractor")
