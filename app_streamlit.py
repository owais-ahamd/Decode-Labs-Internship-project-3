import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Styling (premium dark theme)
# -------------------------------
THEME_CSS = """
<style>
  :root{
    --bg0:#070A12;
    --bg1:#0B1220;
    --card:#0F1A2E;
    --card2:#0C172A;
    --muted:#94A3B8;
    --text:#E2E8F0;
    --primary:#818CF8;
    --primary2:#60A5FA;
    --border:rgba(255,255,255,0.08);
    --shadow: 0 16px 60px rgba(0,0,0,0.45);
  }
  html, body { background: radial-gradient(1000px 500px at 10% 0%, rgba(129,140,248,0.18), transparent 55%), radial-gradient(900px 520px at 90% 10%, rgba(96,165,250,0.16), transparent 50%), var(--bg0) !important; }
  .stApp { background: transparent; }
  .title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--primary), var(--primary2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .subtitle { color: var(--muted); font-size: 14px; margin-top: 6px; }
  .card {
    background: linear-gradient(180deg, rgba(129,140,248,0.10), rgba(255,255,255,0.02));
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: var(--shadow);
    padding: 18px;
  }
  .pill {
    display:inline-flex;
    align-items:center;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(129,140,248,0.14);
    border: 1px solid rgba(129,140,248,0.25);
    color: #C7D2FE;
    font-weight: 700;
    font-size: 12px;
    margin: 6px 8px 0 0;
    user-select:none;
  }
  .btn-primary {
    background: linear-gradient(90deg, var(--primary), var(--primary2)) !important;
    color: white !important;
    border: none !important;
    font-weight: 800;
  }
  .divider { height: 1px; background: var(--border); margin: 14px 0; }
  .muted { color: var(--muted); }
  .score-bar {
    width: 100%;
    height: 10px;
    background: rgba(148,163,184,0.18);
    border-radius: 999px;
    overflow:hidden;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .score-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--primary2));
    border-radius: 999px;
    transition: width 420ms ease;
  }
  @media (max-width: 768px){
    .title { font-size: 26px !important; }
  }
</style>
"""


# -------------------------------
# Data / ML
# -------------------------------
@st.cache_data(show_spinner=False)
def load_or_create_data() -> pd.DataFrame:
    """Load raw_skills.csv if present; otherwise create a small sample dataset."""
    csv_file = "raw_skills.csv"
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)

    sample_data = {
        "job_title": [
            "AI Engineer",
            "Cloud Architect",
            "Data Scientist",
            "DevOps Engineer",
            "Site Reliability Engineer",
            "Machine Learning Engineer",
            "Backend Developer",
            "Frontend Developer",
            "Full Stack Developer",
            "Security Engineer",
            "Solutions Architect",
            "Product Manager",
            "AI Researcher",
            "Database Administrator",
            "Network Engineer",
            "Technical Lead",
        ],
        "skills": [
            "Python, AI, Machine Learning, Deep Learning, TensorFlow, Research",
            "Python, Cloud Computing, AWS, Automation, DevOps, Linux",
            "Python, R, Machine Learning, Statistics, SQL, Data Visualization",
            "Python, Automation, CI/CD, Kubernetes, Docker, Cloud, Linux",
            "Python, Cloud, Monitoring, Automation, Linux, Troubleshooting",
            "Python, Machine Learning, Deep Learning, TensorFlow, NLP, Mathematics",
            "Java, Spring, REST APIs, SQL, Microservices, Git",
            "JavaScript, React, HTML, CSS, UX Design, Git",
            "JavaScript, Python, SQL, React, Node.js, Git, Cloud",
            "Security, Python, Network Security, Cryptography, Linux, DevOps",
            "Cloud, Architecture, Design Patterns, AWS, Azure, GCP, DevOps",
            "Product Strategy, Agile, Data Analysis, UX, Communication",
            "Python, Machine Learning, Research, Mathematics, Statistics, AI",
            "SQL, Database Design, Performance Tuning, NoSQL, Backup",
            "Network, TCP/IP, Routing, Security, Monitoring, Automation",
            "Leadership, Architecture, Design, Python, Cloud, Agile",
        ],
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_file, index=False)
    return df


@st.cache_data(show_spinner=False)
def build_vectors(df: pd.DataFrame) -> Tuple[TfidfVectorizer, object]:
    job_descriptions = df["skills"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
    )
    job_vectors = vectorizer.fit_transform(job_descriptions)
    return vectorizer, job_vectors


def parse_user_skills(raw: str, min_skills: int = 3) -> Tuple[bool, list[str] | str]:
    if not raw or not raw.strip():
        return False, "Please enter at least 3 skills."

    # split by commas OR whitespace
    parts = [p.strip().lower() for p in re.split(r"[\s,]+", raw) if p.strip()]

    if len(parts) < min_skills:
        return False, f"Only {len(parts)} skill(s) entered. Need at least {min_skills}."

    # remove duplicates while preserving order
    seen = set()
    uniq = []
    for p in parts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    return True, uniq


def get_recommendations(user_skills: list[str], df: pd.DataFrame, vectorizer, job_vectors, top_n: int = 3):
    user_text = ", ".join(user_skills)
    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, job_vectors).flatten()

    # top indices sorted
    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score <= 0:
            continue
        results.append(
            {
                "job_title": df.iloc[idx]["job_title"],
                "skills": df.iloc[idx]["skills"],
                "similarity": round(score * 100, 1),
            }
        )
    return results


# -------------------------------
# UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="Tech Stack Recommender",
        page_icon="",
        layout="centered",
    )

    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align:center;'>
          <div class='title'> Tech Stack Recommender</div>
          <div class='subtitle'>Type your skills and get premium career-path matches using TF‑IDF + cosine similarity.</div>
        </div>
        <div class='divider'></div>
        """,
        unsafe_allow_html=True,
    )

    df = load_or_create_data()
    vectorizer, job_vectors = build_vectors(df)

    with st.form("skills_form", clear_on_submit=False):
        col1, col2 = st.columns([1.6, 1])
        with col1:
            skills_input = st.text_input(
                "Your skills",
                placeholder="e.g. Python, Linux, AWS",
                value="",
            )
        with col2:
            top_n = st.selectbox("Show top", options=[1, 2, 3, 4, 5], index=2, help="How many matches to show")
            submit = st.form_submit_button("🔍 Recommend", use_container_width=True)

    if not submit:
        st.markdown(
            """
            <div class='card' style='margin-top:16px;'>
              <b>Try this:</b> enter 3–8 skills you’ve worked with (comma or space separated).\n
              <div style='margin-top:10px;' class='muted'>Example: Python, SQL, AWS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    ok, parsed = parse_user_skills(skills_input, min_skills=3)
    if not ok:
        st.error(str(parsed))
        return

    skills_list = parsed

    with st.spinner("Finding your perfect match..."):
        results = get_recommendations(skills_list, df, vectorizer, job_vectors, top_n=int(top_n))

    if not results:
        st.warning("No matching jobs found. Try different skills (add more keywords).")
        return

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    cols = st.columns(1) if len(results) <= 2 else st.columns(2)
    for i, job in enumerate(results, 1):
        target_col = cols[i % 2] if len(cols) == 2 else cols[0]
        with target_col:
            score = float(job["similarity"])
            skills_for_tag = [s.strip() for s in str(job["skills"]).split(",") if s.strip()]

            st.markdown(
                f"""
                <div class='card' style='margin-top: 14px; background: linear-gradient(180deg, rgba(129,140,248,0.13), rgba(255,255,255,0.03));'>
                  <div style='display:flex; justify-content:space-between; align-items:flex-start; gap: 12px;'>
                    <div>
                      <div style='font-size:18px; font-weight: 850; color: var(--text); line-height:1.2;'>#{i} — {job['job_title']}</div>
                      <div class='muted' style='margin-top:6px; font-size: 13px;'>Match confidence</div>
                    </div>
                    <div class='pill' title='Similarity score from TF‑IDF vectors'>
                      {score:.1f}%
                    </div>
                  </div>

                  <div style='margin-top:12px;' class='score-bar'>
                    <div class='score-fill' style='width:{score:.1f}%;'></div>
                  </div>

                  <div style='margin-top:12px;'>
                    {''.join([f"<span class='pill' style='background: rgba(15,26,46,0.8); border-color: rgba(255,255,255,0.10); color: #CBD5E1;'>{re.sub(r'[^A-Za-z0-9\-\+\_ ]','', s)}</span>" for s in skills_for_tag[:10]])}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div style='text-align:center; margin-top: 26px; color: rgba(148,163,184,0.9); font-size: 12px;'>
          Powered by TF‑IDF & Cosine Similarity • Built for speed with Streamlit caching
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

