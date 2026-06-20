import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------------
# 1. LOAD OR CREATE DATASET
# -------------------------------
def load_or_create_data():
    csv_file = 'raw_skills.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(" Dataset loaded from 'raw_skills.csv'")
        return df
    else:
        print("  'raw_skills.csv' not found. Creating a sample dataset...")
        sample_data = {
            'job_title': [
                'AI Engineer', 'Cloud Architect', 'Data Scientist', 'DevOps Engineer',
                'Site Reliability Engineer', 'Machine Learning Engineer',
                'Backend Developer', 'Frontend Developer', 'Full Stack Developer',
                'Security Engineer', 'Solutions Architect', 'Product Manager',
                'AI Researcher', 'Database Administrator', 'Network Engineer', 'Technical Lead'
            ],
            'skills': [
                'Python, AI, Machine Learning, Deep Learning, TensorFlow, Research',
                'Python, Cloud Computing, AWS, Automation, DevOps, Linux',
                'Python, R, Machine Learning, Statistics, SQL, Data Visualization',
                'Python, Automation, CI/CD, Kubernetes, Docker, Cloud, Linux',
                'Python, Cloud, Monitoring, Automation, Linux, Troubleshooting',
                'Python, Machine Learning, Deep Learning, TensorFlow, NLP, Mathematics',
                'Java, Spring, REST APIs, SQL, Microservices, Git',
                'JavaScript, React, HTML, CSS, UX Design, Git',
                'JavaScript, Python, SQL, React, Node.js, Git, Cloud',
                'Security, Python, Network Security, Cryptography, Linux, DevOps',
                'Cloud, Architecture, Design Patterns, AWS, Azure, GCP, DevOps',
                'Product Strategy, Agile, Data Analysis, UX, Communication',
                'Python, Machine Learning, Research, Mathematics, Statistics, AI',
                'SQL, Database Design, Performance Tuning, NoSQL, Backup',
                'Network, TCP/IP, Routing, Security, Monitoring, Automation',
                'Leadership, Architecture, Design, Python, Cloud, Agile'
            ]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(csv_file, index=False)
        print(" Sample dataset created and saved as 'raw_skills.csv'")
        return df

# -------------------------------
# 2. BUILD TF-IDF VECTORS
# -------------------------------
def prepare_vectors(df):
    job_descriptions = df['skills'].fillna('').astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    job_vectors = vectorizer.fit_transform(job_descriptions)
    return vectorizer, job_vectors

# -------------------------------
# 3. GET USER SKILLS (UPDATED – accepts spaces OR commas)
# -------------------------------
def get_user_skills():
    print("\n Enter your skills (at least 3, separated by commas OR spaces):")
    while True:
        raw = input("> ").strip()
        if not raw:
            print(" No input. Please enter something.")
            continue

        # If input has commas, split by commas. Otherwise, split by spaces.
        if ',' in raw:
            skills = [s.strip().lower() for s in raw.split(',') if s.strip()]
        else:
            skills = [s.strip().lower() for s in raw.split() if s.strip()]

        if len(skills) < 3:
            print(f" Only {len(skills)} skill(s) entered. Need at least 3.")
            continue
        return skills

# -------------------------------
# 4. RECOMMEND TOP N
# -------------------------------
def recommend_jobs(user_skills, df, vectorizer, job_vectors, top_n=3):
    user_text = ', '.join(user_skills)
    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, job_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                'job_title': df.iloc[idx]['job_title'],
                'skills': df.iloc[idx]['skills'],
                'similarity': similarities[idx]
            })
    return results

# -------------------------------
# 5. DISPLAY RESULTS
# -------------------------------
def display_results(results):
    if not results:
        print("\n No matching jobs found. Try using more specific skills.")
        return
    print("\n" + "="*55)
    print(" YOUR TOP RECOMMENDED CAREER PATHS")
    print("="*55)
    for i, job in enumerate(results, 1):
        pct = job['similarity'] * 100
        print(f"\n{i}. {job['job_title']}  →  {pct:.1f}% match")
        print(f"   Skills needed: {job['skills']}")

# -------------------------------
# 6. MAIN
# -------------------------------
def main():
    print("\n TECH STACK RECOMMENDER")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df = load_or_create_data()
    vectorizer, job_vectors = prepare_vectors(df)
    user_skills = get_user_skills()
    results = recommend_jobs(user_skills, df, vectorizer, job_vectors, top_n=3)
    display_results(results)
    print("\n Done. Thank you for using the Tech Stack Recommender!\n")

if __name__ == "__main__":
    main()