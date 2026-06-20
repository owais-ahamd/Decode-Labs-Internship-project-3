# Tech Stack Recommender (Streamlit)

A production-style Streamlit web app that recommends job/role career paths based on the skills you enter. The app uses **TF‑IDF** to vectorize skill text and **cosine similarity** to match your input to a curated dataset of roles.

---

## Features

- **Premium, responsive UI** (mobile/tablet/desktop) with modern card layout
- **Fast startup** using Streamlit caching for dataset + vectorizer
- **Robust validation** for user input (requires at least 3 skills, supports comma or whitespace input)
- **Clear results**: top matches with confidence score + skill tags
- **Graceful fallbacks**: if `raw_skills.csv` doesn’t exist, the app creates a small sample dataset

---

## Technologies

- **Python**
- **Streamlit** (UI + caching)
- **pandas**, **numpy**
- **scikit-learn**
  - `TfidfVectorizer`
  - `cosine_similarity`

---

## Project Structure

```text
.
├── app_streamlit.py        # Streamlit application (Tech Stack Recommender)
├── raw_skills.csv         # Role dataset (job_title, skills)
└── README.md
```

---

## Installation

1) (Recommended) Create a virtual environment.

```bash
pip install streamlit pandas numpy scikit-learn
```

---

## Run the App

From the folder containing the files above:

```bash
streamlit run app_streamlit.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

---

## Usage

1. Enter your skills (comma-separated or space-separated), e.g.
   - `Python, SQL, AWS`
2. Click **Recommend**.
3. View the top role matches and their confidence score.

---

## Future Improvements

- Add a larger, editable dataset and admin upload UI
- Add synonym/normalization (e.g., “ML” → “Machine Learning”)
- Persist recommendation history (optional)
- Add more advanced ranking (e.g., hybrid matching with keyword boosts)

---

## Notes

- The repo includes other ML artifacts (Iris KNN) only if present in your folder. This app specifically powers the **Tech Stack Recommender**.

