from pyresparser import ResumeParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_resume_skills(resume_path):
    data = ResumeParser(resume_path).get_extracted_data()
    return " ".join(data.get("skills", []))

def calculate_match(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]

if __name__ == "__main__":
    resume = "sample_resume.pdf"
    
    job_description = """
    Looking for a Python developer with knowledge of machine learning,
    NLP, and data analysis.
    """

    resume_skills = extract_resume_skills(resume)
    score = calculate_match(resume_skills, job_description)

    print("Match Score:", score)
