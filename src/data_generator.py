"""
Smart Student Performance & Placement Prediction System
Data Generator Module: Generates realistic student academic, skill, and placement data.
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples: int = 1200, random_state: int = 42, add_noise_and_nulls: bool = True) -> pd.DataFrame:
    """
    Generates a realistic student placement dataset.
    
    Features:
    - Student_ID: Unique identifier (e.g., STU1001)
    - CGPA: Cumulative Grade Point Average (5.0 - 10.0)
    - Programming_Score: Coding proficiency test score (0 - 100)
    - Communication_Score: Verbal/Soft skills score (0 - 100)
    - Internships: Number of internships completed (0 - 5)
    - Projects: Number of major/minor technical projects (0 - 8)
    - Aptitude_Score: Quantitative and logical aptitude score (0 - 100)
    - Technical_Certifications: Certified technical credentials (0 - 6)
    - Work_Experience: Previous relevant industry experience ('Yes' or 'No')
    - Backlogs: Active or past academic backlogs (0 - 4)
    - Extracurricular_Activities: Participation in clubs/hackathons ('Yes' or 'No')
    - Placement_Status: Target label ('Placed' or 'Not Placed')
    """
    np.random.seed(random_state)
    
    student_ids = [f"STU{1000 + i}" for i in range(n_samples)]
    
    # Generate realistic feature distributions
    cgpa = np.clip(np.random.normal(loc=7.4, scale=1.1, size=n_samples), 5.0, 10.0)
    programming_score = np.clip(np.random.normal(loc=68.0, scale=14.0, size=n_samples) + (cgpa - 7.0) * 4.0, 30.0, 100.0)
    communication_score = np.clip(np.random.normal(loc=67.0, scale=13.0, size=n_samples), 35.0, 100.0)
    aptitude_score = np.clip(np.random.normal(loc=70.0, scale=12.5, size=n_samples) + (cgpa - 7.0) * 3.0, 35.0, 100.0)
    
    internships = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.30, 0.38, 0.20, 0.09, 0.03])
    projects = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n_samples, p=[0.05, 0.15, 0.30, 0.25, 0.15, 0.07, 0.03])
    technical_certifications = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.20, 0.35, 0.25, 0.12, 0.06, 0.02])
    
    # Work experience correlated with internships
    prob_work_exp = np.clip(0.1 + 0.15 * internships, 0.05, 0.85)
    work_exp_num = (np.random.rand(n_samples) < prob_work_exp).astype(int)
    work_experience = np.where(work_exp_num == 1, 'Yes', 'No')
    
    # Backlogs negatively correlated with CGPA
    prob_backlog = np.clip(0.6 - (cgpa - 5.0) * 0.12, 0.02, 0.7)
    backlogs = np.random.binomial(n=3, p=prob_backlog)
    
    extracurricular = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.55, 0.45])
    extracurricular_num = np.where(extracurricular == 'Yes', 1, 0)
    
    # Realistic Logistic Response Function (Log-odds calculation)
    # z = beta_0 + sum(beta_i * x_i)
    # Balanced intercept calibrated so ~56% placed, ~44% not placed with realistic academic variability
    z = (
        -21.8 +
        1.42 * cgpa +
        0.052 * programming_score +
        0.038 * communication_score +
        0.042 * aptitude_score +
        0.65 * internships +
        0.38 * projects +
        0.28 * technical_certifications +
        0.72 * work_exp_num +
        0.25 * extracurricular_num -
        1.15 * backlogs +
        np.random.normal(loc=0.0, scale=0.95, size=n_samples) # Realistic environmental noise
    )
    
    # Sigmoid function: p = 1 / (1 + exp(-z))
    probabilities = 1.0 / (1.0 + np.exp(-z))
    placement_status = np.where(probabilities >= 0.50, 'Placed', 'Not Placed')
    
    df = pd.DataFrame({
        'Student_ID': student_ids,
        'CGPA': np.round(cgpa, 2),
        'Programming_Score': np.round(programming_score, 1),
        'Communication_Score': np.round(communication_score, 1),
        'Aptitude_Score': np.round(aptitude_score, 1),
        'Internships': internships,
        'Projects': projects,
        'Technical_Certifications': technical_certifications,
        'Work_Experience': work_experience,
        'Backlogs': backlogs,
        'Extracurricular_Activities': extracurricular,
        'Placement_Status': placement_status
    })
    
    # Introduce a tiny amount of realistic null values & duplicates if requested (for preprocessing demonstration)
    if add_noise_and_nulls:
        # Add 12 duplicates
        duplicates = df.sample(n=12, random_state=random_state)
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Add a few missing values in continuous/categorical columns (approx 1.5%)
        mask_prog = np.random.rand(len(df)) < 0.015
        df.loc[mask_prog, 'Programming_Score'] = np.nan
        
        mask_comm = np.random.rand(len(df)) < 0.012
        df.loc[mask_comm, 'Communication_Score'] = np.nan
        
        mask_work = np.random.rand(len(df)) < 0.010
        df.loc[mask_work, 'Work_Experience'] = np.nan
    
    return df

def save_default_dataset(filepath: str = "data/student_placement_dataset.csv") -> str:
    """Generates and saves the default student placement dataset."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = generate_student_dataset(n_samples=1200, random_state=42, add_noise_and_nulls=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully created and saved at: {filepath} with shape {df.shape}")
    return filepath

if __name__ == "__main__":
    save_default_dataset()
