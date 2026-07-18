"""
Annotation module for labeling case outcomes and categories
"""
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaseAnnotator:
    """Annotate cases with outcomes and categories"""
    
    def __init__(self):
        self.outcome_success_keywords = [
            'appeal allowed',
            'appeal succeeds',
            'appeal is allowed',
            'appellant succeeds',
            'judgment for the plaintiff',
            'plaintiff succeeds',
            'declaration granted',
            'injunction granted',
            'damages awarded',
            'entitled to statutory right of occupancy',
            'in favour of the appellant',
            'in favor of the appellant',
            'relief granted',
            'appeal is hereby allowed',
        ]
        
        self.outcome_failure_keywords = [
            'appeal dismissed',
            'appeal fails',
            'appeal is dismissed',
            'appellant fails',
            'judgment for the defendant',
            'plaintiff fails',
            'claim dismissed',
            'against the appellant',
            'relief denied',
            'appeal is hereby dismissed',
        ]
    
    def extract_outcome(self, text, sections):
        """
        Determine case outcome: SUCCESS or FAILURE for plaintiff/appellant
        
        Args:
            text: Full judgment text
            sections: Parsed sections dict
            
        Returns:
            str: 'PLAINTIFF_SUCCESS', 'PLAINTIFF_FAILURE', or 'UNCLEAR'
        """
        # Focus on decision section and last 2000 characters
        decision_text = sections.get('decision', '')
        last_part = text[-2000:] if len(text) > 2000 else text
        combined_text = (decision_text + " " + last_part).lower()
        
        # Count keyword occurrences
        success_score = sum(
            keyword in combined_text 
            for keyword in self.outcome_success_keywords
        )
        
        failure_score = sum(
            keyword in combined_text 
            for keyword in self.outcome_failure_keywords
        )
        
        logger.debug(f"Success score: {success_score}, Failure score: {failure_score}")
        
        # Determine outcome
        if success_score > failure_score:
            return 'PLAINTIFF_SUCCESS'
        elif failure_score > success_score:
            return 'PLAINTIFF_FAILURE'
        else:
            return 'UNCLEAR'  # Manual review needed
    
    def categorize_land_dispute(self, text):
        """
        Categorize type of land dispute
        
        Args:
            text: Judgment text
            
        Returns:
            list: List of categories (can be multiple)
        """
        text_lower = text.lower()
        categories = []
        
        # Ownership dispute
        ownership_keywords = [
            'ownership', 'title to land', 'who owns', 'rightful owner',
            'declaration of title', 'proprietary interest'
        ]
        if any(keyword in text_lower for keyword in ownership_keywords):
            categories.append('OWNERSHIP_DISPUTE')
        
        # Inheritance
        inheritance_keywords = [
            'inherit', 'inheritance', 'succession', 'estate', 'deceased',
            'will', 'intestate', 'heir', 'beneficiary'
        ]
        if any(keyword in text_lower for keyword in inheritance_keywords):
            categories.append('INHERITANCE')
        
        # Boundary
        boundary_keywords = [
            'boundary', 'encroachment', 'demarcation', 'survey plan',
            'beacon', 'encroach'
        ]
        if any(keyword in text_lower for keyword in boundary_keywords):
            categories.append('BOUNDARY')
        
        # Trespass
        trespass_keywords = [
            'trespass', 'unlawful entry', 'illegal occupation',
            'trespasser', 'unlawful possession'
        ]
        if any(keyword in text_lower for keyword in trespass_keywords):
            categories.append('TRESPASS')
        
        # Governor's Consent
        consent_keywords = [
            "governor's consent", 'consent of the governor',
            'section 22', 'deemed consent'
        ]
        if any(keyword in text_lower for keyword in consent_keywords):
            categories.append('GOVERNORS_CONSENT')
        
        # Revocation
        revocation_keywords = [
            'revocation', 'revoke', 'revoked', 'section 28',
            'revoke the right of occupancy'
        ]
        if any(keyword in text_lower for keyword in revocation_keywords):
            categories.append('REVOCATION')
        
        # Compensation
        compensation_keywords = [
            'compensation', 'compulsory acquisition', 'adequate compensation',
            'market value', 'acquisition'
        ]
        if any(keyword in text_lower for keyword in compensation_keywords):
            categories.append('COMPENSATION')
        
        # Land Use Violation
        land_use_keywords = [
            'land use', 'development control', 'contravention',
            'planning permission', 'building approval'
        ]
        if any(keyword in text_lower for keyword in land_use_keywords):
            categories.append('LAND_USE_VIOLATION')
        
        # If no category found
        if not categories:
            categories.append('OTHER_LAND_MATTER')
        
        return categories
    
    def extract_legal_features(self, text):
        """
        Extract domain-specific legal features
        
        Args:
            text: Judgment text
            
        Returns:
            dict: Binary legal features
        """
        text_lower = text.lower()
        
        features = {
            # Land Use Act features
            'mentions_land_use_act': int('land use act' in text_lower),
            'has_certificate_of_occupancy': int(any(
                term in text_lower 
                for term in ['certificate of occupancy', 'c of o', 'c.o.']
            )),
            'has_governors_consent': int(
                "governor's consent" in text_lower or 
                'consent of the governor' in text_lower
            ),
            'mentions_section_22': int('section 22' in text_lower),
            'mentions_section_28': int('section 28' in text_lower),
            
            # Evidence Act features
            'mentions_evidence_act': int('evidence act' in text_lower),
            'has_documentary_evidence': int('documentary evidence' in text_lower),
            'has_survey_evidence': int(
                'survey' in text_lower and 
                ('plan' in text_lower or 'report' in text_lower)
            ),
            'mentions_section_84': int('section 84' in text_lower),
            
            # Tenure type
            'customary_tenure': int(
                'customary' in text_lower and 'tenure' in text_lower
            ),
            'statutory_tenure': int(
                'statutory' in text_lower and 
                ('tenure' in text_lower or 'right of occupancy' in text_lower)
            ),
            
            # Evidence types
            'has_witness_testimony': int('witness' in text_lower),
            'has_expert_evidence': int('expert' in text_lower),
            
            # Other features
            'involves_inheritance': int(
                'inherit' in text_lower or 'succession' in text_lower
            ),
            'has_boundary_issue': int('boundary' in text_lower),
            'mentions_trespass': int('trespass' in text_lower),
            'involves_fraud': int('fraud' in text_lower),
            'has_oral_evidence': int('oral evidence' in text_lower),
        }
        
        return features
    
    def annotate_case(self, text, sections):
        """
        Complete annotation of a single case
        
        Args:
            text: Full judgment text
            sections: Parsed sections
            
        Returns:
            dict: All annotations
        """
        outcome = self.extract_outcome(text, sections)
        categories = self.categorize_land_dispute(text)
        legal_features = self.extract_legal_features(text)
        
        annotations = {
            'outcome': outcome,
            'categories': categories,
            'primary_category': categories[0] if categories else 'OTHER_LAND_MATTER',
            **legal_features
        }
        
        logger.info(f"Annotated case - Outcome: {outcome}, Categories: {', '.join(categories)}")
        
        return annotations


def annotate_dataset(cases_data):
    """
    Annotate multiple cases
    
    Args:
        cases_data: List of dicts with 'text' and 'sections' keys
        
    Returns:
        list: List of annotated cases
    """
    annotator = CaseAnnotator()
    annotated_cases = []
    
    for i, case in enumerate(cases_data):
        try:
            annotations = annotator.annotate_case(
                case['text'], 
                case['sections']
            )
            
            annotated_case = {
                **case,
                **annotations
            }
            
            annotated_cases.append(annotated_case)
            
        except Exception as e:
            logger.error(f"Failed to annotate case {i}: {e}")
            continue
    
    logger.info(f"Annotated {len(annotated_cases)}/{len(cases_data)} cases")
    
    return annotated_cases


if __name__ == "__main__":
    # Example usage
    annotator = CaseAnnotator()
    
    sample_text = """
    The appellant seeks declaration of title to the land in question.
    The land is covered by Certificate of Occupancy No. XYZ/123.
    However, the transfer was made without obtaining Governor's Consent
    as required under Section 22 of the Land Use Act 1978.
    
    HELD: The appeal is hereby dismissed.
    """
    
    sample_sections = {
        'decision': 'The appeal is hereby dismissed.'
    }
    
    annotations = annotator.annotate_case(sample_text, sample_sections)
    
    print("Annotations:")
    for key, value in annotations.items():
        print(f"  {key}: {value}")
    
    print("\nAnnotation module loaded successfully.")
