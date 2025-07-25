import pandas as pd
import re
from collections import Counter

def analyze_terminology_gaps():
    """Analyze each dataset for missing critical terms"""
    
    # Define expected key terms for each category
    expected_terms = {
        "AI": ["machine learning", "neural network", "deep learning", "algorithm", 
               "classification", "regression", "supervised", "unsupervised", 
               "reinforcement", "nlp", "computer vision", "optimization", 
               "feature", "model", "training", "prediction", "ai"],
        
        "Image Processing": ["pixel", "filter", "convolution", "enhancement", 
                           "restoration", "compression", "morphology", "edge", 
                           "feature extraction", "histogram", "noise", "blur", 
                           "sharpening", "transform", "fourier", "wavelet"],
        
        "Distributed Systems": ["distributed", "consensus", "replication", 
                              "consistency", "partition", "fault tolerance", 
                              "scalability", "load balancing", "cluster", 
                              "node", "synchronization", "concurrent", "parallel"],
        
        "Networking & Cybersecurity": ["encryption", "authentication", "firewall", 
                                     "protocol", "vulnerability", "malware", 
                                     "threat", "attack", "defense", "privacy", 
                                     "secure", "cryptography", "penetration", "intrusion"],
        
        "Software Engineering": ["agile", "scrum", "methodology", "testing", 
                               "debugging", "refactoring", "version control", 
                               "requirements", "design patterns", "architecture", 
                               "maintenance", "deployment", "devops", "ci/cd", 
                               "quality assurance", "code review"]
    }
    
    datasets = {
        "AI": "/Users/toe/Htoo/TextClassification/data/ai_data.xlsx",
        "Image Processing": "/Users/toe/Htoo/TextClassification/data/image_processing_data.xlsx",
        "Distributed Systems": "/Users/toe/Htoo/TextClassification/data/distribution_data.xlsx",
        "Networking & Cybersecurity": "/Users/toe/Htoo/TextClassification/data/networking_cybersecurity_data.xlsx",
        "Software Engineering": "/Users/toe/Htoo/TextClassification/data/se_data.xlsx"
    }
    
    for category, file_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"TERMINOLOGY GAP ANALYSIS: {category}")
        print(f"{'='*60}")
        
        df = pd.read_excel(file_path)
        
        # Combine title and abstract text
        text_data = []
        if 'Title' in df.columns:
            text_data.extend(df['Title'].dropna().astype(str))
        if 'Abstract' in df.columns:
            text_data.extend(df['Abstract'].dropna().astype(str))
        
        all_text = ' '.join(text_data).lower()
        
        # Check for expected terms
        missing_terms = []
        present_terms = []
        
        for term in expected_terms[category]:
            # Use word boundaries to match whole words/phrases
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            matches = len(re.findall(pattern, all_text))
            
            if matches == 0:
                missing_terms.append(term)
            else:
                present_terms.append((term, matches))
        
        print(f"\nDataset size: {len(df)} papers")
        print(f"\nMISSING CRITICAL TERMS ({len(missing_terms)}/{len(expected_terms[category])}):")
        for term in missing_terms:
            print(f"  ❌ {term}")
        
        print(f"\nPRESENT TERMS (with frequency):")
        present_terms.sort(key=lambda x: x[1], reverse=True)
        for term, count in present_terms:
            print(f"  ✅ {term}: {count}")
        
        # Calculate coverage percentage
        coverage = (len(present_terms) / len(expected_terms[category])) * 100
        print(f"\nTERMINOLOGY COVERAGE: {coverage:.1f}%")
        
        if coverage < 70:
            print("⚠️  LOW COVERAGE - Needs significant improvement")
        elif coverage < 85:
            print("⚠️  MODERATE COVERAGE - Needs some improvement")
        else:
            print("✅ GOOD COVERAGE")

if __name__ == "__main__":
    analyze_terminology_gaps()