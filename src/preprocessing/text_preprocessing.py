"""
Text preprocessing and cleaning module
"""
import re
import unicodedata
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Clean and standardize legal text"""
    
    def __init__(self):
        # Common OCR errors
        self.ocr_fixes = {
            'l/l': 'II',
            '0f': 'of',
            'tl1e': 'the',
            'witl1': 'with',
            'appellant/respondent': 'appellant respondent',
            'plaint iff': 'plaintiff',
            'defend ant': 'defendant',
        }
        
        # Legal term standardizations
        self.legal_term_patterns = {
            r'Land\s+Use\s+Act,?\s*1978': 'Land Use Act 1978',
            r'L\.U\.A\.': 'Land Use Act',
            r'Evidence\s+Act,?\s*2011': 'Evidence Act 2011',
            r'Supreme\s+Court\s+of\s+Nigeria': 'Supreme Court',
            r'S\.C\.N?\.?': 'Supreme Court',
            r'Certificate\s+of\s+Occupancy': 'Certificate of Occupancy',
            r'C\s*of\s*O\b': 'Certificate of Occupancy',
            r'C\.O\.': 'Certificate of Occupancy',
            r"Governor'?s\s+Consent": "Governor's Consent",
        }
    
    def normalize_unicode(self, text):
        """Normalize unicode characters"""
        return unicodedata.normalize('NFKD', text)
    
    def fix_ocr_errors(self, text):
        """Fix common OCR errors"""
        for wrong, correct in self.ocr_fixes.items():
            text = text.replace(wrong, correct)
        return text
    
    def standardize_quotes(self, text):
        """Standardize quotes and apostrophes"""
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text
    
    def remove_page_numbers(self, text):
        """Remove page number patterns"""
        patterns = [
            r'Page\s+\d+\s+of\s+\d+',
            r'Page\s+\d+',
            r'\d+\s+of\s+\d+',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
    
    def remove_headers_footers(self, text):
        """Remove common headers and footers"""
        patterns = [
            r'IN\s+THE\s+(SUPREME|HIGH|COURT\s+OF\s+APPEAL).*?\n',
            r'SUIT\s+NO\.?\s*:?\s*\S+\s*\n',
            r'HOLDEN\s+AT.*?\n',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
    
    def clean_whitespace(self, text):
        """Clean up excessive whitespace"""
        # Multiple spaces to single
        text = re.sub(r' +', ' ', text)
        # Multiple newlines to double
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Remove leading/trailing whitespace from lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()
    
    def remove_urls_emails(self, text):
        """Remove URLs and email addresses"""
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        return text
    
    def fix_punctuation_spacing(self, text):
        """Fix spacing around punctuation"""
        # Remove space before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        # Ensure space after punctuation (if followed by letter)
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        return text
    
    def standardize_legal_terms(self, text):
        """Standardize legal terminology"""
        for pattern, replacement in self.legal_term_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def clean(self, text):
        """
        Complete cleaning pipeline
        
        Args:
            text: Raw text from document
            
        Returns:
            str: Cleaned text
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for cleaning")
            return ""
        
        logger.info("Starting text cleaning...")
        
        # Apply cleaning steps in order
        text = self.normalize_unicode(text)
        text = self.fix_ocr_errors(text)
        text = self.standardize_quotes(text)
        text = self.remove_page_numbers(text)
        text = self.remove_headers_footers(text)
        text = self.remove_urls_emails(text)
        text = self.clean_whitespace(text)
        text = self.fix_punctuation_spacing(text)
        text = self.standardize_legal_terms(text)
        text = self.clean_whitespace(text)  # Final cleanup
        
        logger.info(f"Text cleaning complete. Length: {len(text)} characters")
        
        return text


class JudgmentParser:
    """Parse judgment into structured sections"""
    
    def parse_sections(self, text):
        """
        Parse judgment into structured sections
        
        Args:
            text: Cleaned judgment text
            
        Returns:
            dict: {section_name: section_text}
        """
        sections = {
            'title': '',
            'parties': '',
            'facts': '',
            'issues': '',
            'arguments': '',
            'analysis': '',
            'decision': ''
        }
        
        # Extract case title (usually first line or within first 500 chars)
        title_match = re.search(
            r'([A-Z][A-Z\s&]+)\s+V\.?\s+([A-Z][A-Z\s&]+)', 
            text[:500]
        )
        if title_match:
            sections['title'] = title_match.group(0)
        
        # Extract facts section
        facts_patterns = [
            r'FACTS?\s*:?\s*(.*?)(?=ISSUES?|ARGUMENTS?|SUBMISSIONS?|HELD|DECISION|$)',
            r'BACKGROUND\s*:?\s*(.*?)(?=ISSUES?|ARGUMENTS?|HELD|$)',
            r'The\s+facts.*?are\s+as\s+follows:?\s*(.*?)(?=ISSUES?|ARGUMENTS?|HELD)',
        ]
        
        for pattern in facts_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections['facts'] = match.group(1).strip()
                break
        
        # Extract issues
        issues_patterns = [
            r'ISSUES?\s+FOR\s+DETERMINATION\s*:?\s*(.*?)(?=ARGUMENTS?|SUBMISSIONS?|ANALYSIS|HELD|DECISION|$)',
            r'ISSUES?\s+FORMULATED\s*:?\s*(.*?)(?=ARGUMENTS?|SUBMISSIONS?|ANALYSIS|HELD|$)',
            r'ISSUES?\s*:?\s*(.*?)(?=ARGUMENTS?|SUBMISSIONS?|ANALYSIS|HELD)',
        ]
        
        for pattern in issues_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections['issues'] = match.group(1).strip()
                break
        
        # Extract decision/holding
        decision_patterns = [
            r'HELD\s*:?\s*(.*?)$',
            r'DECISION\s*:?\s*(.*?)$',
            r'(?:I|WE)\s+(?:THEREFORE\s+)?(?:HOLD|ORDER|DECLARE)\s+(?:THAT\s+)?(.*?)$',
            r'(?:THE\s+)?(?:APPEAL|APPLICATION)\s+(?:IS\s+)?(?:ALLOWED|DISMISSED)(.*?)$',
        ]
        
        for pattern in decision_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections['decision'] = match.group(0).strip()
                break
        
        # If no structured sections found, put everything in analysis
        if not any(sections.values()):
            sections['analysis'] = text
        
        return sections
    
    def extract_metadata(self, text):
        """
        Extract metadata like case number, date, court
        
        Args:
            text: Judgment text
            
        Returns:
            dict: Metadata fields
        """
        metadata = {}
        
        # Extract suit number
        suit_patterns = [
            r'(?:SUIT|SC|CA|HC)\s*(?:NO\.?|NUM\.?)\s*:?\s*([A-Z0-9/\-]+)',
            r'(?:APPEAL\s+NO\.?)\s*:?\s*([A-Z0-9/\-]+)',
        ]
        for pattern in suit_patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                metadata['suit_number'] = match.group(1)
                break
        
        # Extract date
        date_patterns = [
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                metadata['date'] = match.group(1)
                break
        
        # Extract court name
        court_patterns = [
            r'(SUPREME\s+COURT(?:\s+OF\s+NIGERIA)?)',
            r'((?:LAGOS|ABUJA|FCT|EKITI|OYO|KANO)\s+(?:STATE\s+)?HIGH\s+COURT)',
            r'(COURT\s+OF\s+APPEAL)',
        ]
        for pattern in court_patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                metadata['court'] = match.group(1).upper()
                break
        
        # Extract year from date or text
        year_match = re.search(r'(20\d{2})|(\[20\d{2}\])', text[:1000])
        if year_match:
            metadata['year'] = int(year_match.group(1) or year_match.group(2).strip('[]'))
        
        return metadata


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    parser = JudgmentParser()
    
    sample_text = """
    IN THE SUPREME COURT OF NIGERIA
    Page 1 of 45
    
    The appellant filed a suit seeking   declaration  of title to land...
    The C of O was issued without Governor's   Consent...
    """
    
    cleaned_text = preprocessor.clean(sample_text)
    print("Cleaned text:")
    print(cleaned_text)
    print("\n" + "="*50 + "\n")
    
    sections = parser.parse_sections(cleaned_text)
    print("Parsed sections:")
    for section, content in sections.items():
        if content:
            print(f"{section}: {content[:100]}...")
    
    print("\nText preprocessing module loaded successfully.")
