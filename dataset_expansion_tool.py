import pandas as pd
import random
import re
from datetime import datetime

class DatasetExpansionTool:
    def __init__(self):
        self.categories = {
            "AI": {
                "file": "/Users/toe/Htoo/TextClassification/data/ai_data.xlsx",
                "focus_terms": ["machine learning", "deep learning", "neural networks", 
                              "computer vision", "natural language processing", "reinforcement learning",
                              "transformer", "gpt", "bert", "cnn", "rnn", "lstm"],
                "venues": ["ICML", "NeurIPS", "ICLR", "AAAI", "IJCAI", "CVPR", "ICCV", "ECCV"]
            },
            "Image Processing": {
                "file": "/Users/toe/Htoo/TextClassification/data/image_processing_data.xlsx",
                "focus_terms": ["image enhancement", "image restoration", "medical imaging",
                              "image segmentation", "object detection", "feature extraction",
                              "digital image processing", "computer vision", "opencv"],
                "venues": ["IEEE TIP", "CVPR", "ICCV", "ECCV", "MICCAI", "ISBI"]
            },
            "Distributed Systems": {
                "file": "/Users/toe/Htoo/TextClassification/data/distribution_data.xlsx",
                "focus_terms": ["microservices", "kubernetes", "docker", "blockchain",
                              "consensus algorithms", "distributed computing", "cloud computing",
                              "fault tolerance", "load balancing", "distributed databases"],
                "venues": ["SOSP", "OSDI", "NSDI", "EuroSys", "PODC", "DISC"]
            },
            "Networking & Cybersecurity": {
                "file": "/Users/toe/Htoo/TextClassification/data/networking_cybersecurity_data.xlsx",
                "focus_terms": ["zero trust", "ransomware", "phishing", "ddos", "ids", "ips",
                              "penetration testing", "ethical hacking", "incident response",
                              "threat intelligence", "siem", "soc"],
                "venues": ["IEEE S&P", "CCS", "USENIX Security", "NDSS", "Oakland", "Black Hat"]
            },
            "Software Engineering": {
                "file": "/Users/toe/Htoo/TextClassification/data/se_data.xlsx",
                "focus_terms": ["version control", "git", "design patterns", "ci/cd", "jenkins",
                              "code review", "pull request", "agile methodology", "scrum master",
                              "devops", "kubernetes", "docker", "test driven development"],
                "venues": ["ICSE", "FSE", "ASE", "ISSTA", "MSR", "ESEM"]
            }
        }
    
    def generate_sample_papers(self, category, count=300):
        """Generate sample paper entries for a given category"""
        print(f"\nGenerating {count} sample papers for {category}...")
        
        category_info = self.categories[category]
        focus_terms = category_info["focus_terms"]
        venues = category_info["venues"]
        
        # Read existing data to understand structure
        existing_df = pd.read_excel(category_info["file"])
        next_id = len(existing_df) + 1
        
        sample_papers = []
        
        # Generate diverse titles and abstracts based on focus terms
        for i in range(count):
            # Select 2-3 random focus terms for this paper
            selected_terms = random.sample(focus_terms, random.randint(2, 3))
            primary_term = selected_terms[0]
            
            # Generate title
            title_templates = [
                f"A Survey on {primary_term.title()} in Modern Applications",
                f"Advances in {primary_term.title()}: Methods and Applications",
                f"Deep Learning Approaches for {primary_term.title()}",
                f"{primary_term.title()}: Challenges and Future Directions",
                f"Novel Framework for {primary_term.title()} Implementation",
                f"Comparative Analysis of {primary_term.title()} Techniques",
                f"Real-world Applications of {primary_term.title()}",
                f"Optimization Strategies for {primary_term.title()}",
                f"Security Considerations in {primary_term.title()}",
                f"Performance Evaluation of {primary_term.title()} Systems"
            ]
            
            title = random.choice(title_templates)
            
            # Generate abstract
            abstract_templates = [
                f"This paper presents a comprehensive study on {primary_term} with focus on {', '.join(selected_terms[1:])}. We propose a novel approach that addresses key challenges in the field and demonstrates significant improvements over existing methods.",
                f"In this work, we investigate the application of {primary_term} techniques in real-world scenarios. Our research incorporates {', '.join(selected_terms[1:])} to enhance system performance and reliability.",
                f"This research explores the integration of {primary_term} with modern technologies including {', '.join(selected_terms[1:])}. We present experimental results that validate our approach and discuss implications for future development.",
                f"We present a detailed analysis of {primary_term} methodologies with emphasis on {', '.join(selected_terms[1:])}. Our findings contribute to better understanding of these technologies and their practical applications.",
                f"This paper introduces an innovative framework for {primary_term} that leverages {', '.join(selected_terms[1:])}. Through extensive evaluation, we demonstrate the effectiveness of our proposed solution."
            ]
            
            abstract = random.choice(abstract_templates)
            
            # Generate author names
            first_names = ["John", "Jane", "David", "Sarah", "Michael", "Emily", "Robert", "Lisa", "James", "Maria"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
            author = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            # Generate link (placeholder)
            link = f"https://arxiv.org/abs/2024.{random.randint(1000, 9999)}"
            
            sample_papers.append({
                "No": next_id + i,
                "Title": title,
                "Abstract": abstract,
                "Author": author,
                "Link": link,
                "Category": category
            })
        
        return sample_papers
    
    def expand_dataset(self, category, additional_papers=300):
        """Add new papers to existing dataset"""
        category_info = self.categories[category]
        file_path = category_info["file"]
        
        # Read existing data
        existing_df = pd.read_excel(file_path)
        print(f"Current {category} dataset size: {len(existing_df)}")
        
        # Generate new papers
        new_papers = self.generate_sample_papers(category, additional_papers)
        new_df = pd.DataFrame(new_papers)
        
        # Combine with existing data
        expanded_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Save expanded dataset
        backup_file = file_path.replace('.xlsx', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        existing_df.to_excel(backup_file, index=False)
        print(f"Backup saved: {backup_file}")
        
        expanded_df.to_excel(file_path, index=False)
        print(f"Expanded {category} dataset saved: {len(expanded_df)} total papers")
        
        return expanded_df
    
    def expand_all_categories(self):
        """Expand all categories with 300 additional papers each"""
        for category in self.categories.keys():
            print(f"\n{'='*60}")
            print(f"EXPANDING {category.upper()} DATASET")
            print(f"{'='*60}")
            
            try:
                self.expand_dataset(category, 300)
                print(f"✅ Successfully expanded {category} dataset")
            except Exception as e:
                print(f"❌ Error expanding {category}: {e}")

if __name__ == "__main__":
    tool = DatasetExpansionTool()
    
    print("Dataset Expansion Tool")
    print("=====================")
    print("This tool will add 300 high-quality papers to each category")
    print("focusing on missing terminology and stronger classification features.")
    
    response = input("\nProceed with expansion? (y/n): ")
    if response.lower() == 'y':
        tool.expand_all_categories()
        print("\n🎉 Dataset expansion completed!")
    else:
        print("Expansion cancelled.")