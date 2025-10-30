# Resume-ranking-system

Developed an intelligent resume screening system that automatically evaluates and ranks resumes based on their similarity to a given job description, using natural language processing and machine learning techniques.

**Tools & Technologies:**

**Programming Language:** Python

**Libraries:**

*scikit-learn → TF-IDF Vectorizer, Cosine Similarity, KNN Classifier

*pandas, numpy → Data manipulation and preprocessing

*matplotlib, seaborn  → Visualization of similarity score distributions

**IDE/Environment:** Jupyter Notebook / VS Code

**Project Workflow:**

**Step 1:** Input Collection

Accepts:

  **Job Description (JD):** Text input

  **Candidate Resumes**: PDF/DOCX/Text files

**Step 2:** Text Preprocessing

Lowercasing, punctuation & number removal

Stopword elimination

Tokenization and lemmatization (via spaCy/NLTK)

**Step 3:** Feature Extraction

Converts JD and resumes into TF-IDF vectors representing text numerically

**Step 4:** Similarity Calculation

Computes Cosine Similarity between the job description and each resume

Higher similarity = better candidate match

**Step 5:** Grading System

Rule-based grading:

High Fit → Similarity > 0.35

Medium Fit → 0.20–0.35

Low Fit → < 0.20

KNN Classifier: Trained on similarity scores to predict grades for unseen resumes

**Step 6**: Ranking & Output

Ranks resumes based on similarity score

Displays:
Candidate | Similarity Score | Grade in tabular format
