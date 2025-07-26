# filename: app.py

import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your insights as knowledge base
insights = [
    "The highest salaries are in the sales department.",
    "The highest paid employees are the ones working in sales as senior staff for nearly 20 years and the techniques get the most salary in development.",
    "The highest salary increase is in the research department.",
    "Senior staffs are the most paid in most departments, but in research, development, production and quality management the salaries are mostly based on accomplishments not titles.",
    "For most departments, salaries increase with tenure. Research and Development departments have higher top-end salaries compared to others. In contrast marketing and customer service show the lowest.",
    "Departments like Research, Finance, and Production show higher median salary growth. Research especially has a wide and positive salary growth spread, indicating compensation based on merits and accomplishments, while in customer service, marketing and quality management there appears to be demotion based on decreased productivity.",
    "Older employees generally have longer tenure and hold higher titles.",
    "Higher titles are predominantly held by older employees, suggesting title progression is age-linked, likely due to experience accumulation over years.",
    "There is an observed gap suggesting that not all employees progress in title at the same rate, potentially due to different career tracks, performance differences, or time in current role not aligning with age.",
    "Customer service has the lowest tenure most likely because of burnouts, while sales and development have the highest.",
    "Tenure increases with increasing salaries and higher titles. Managers in different departments have the highest tenure, in contrast lowest tenures are specifically with customer service department and staff title.",
    "Development engineers get consistent raises, likely promotions. In contrast, customer service and finance have very low or negative salary change which is most likely due to stagnation or demotion.",
    "It's clear that loyalty to the department (highest tenure) is in some cases rewarded by an increase in salary."
]

# Precompute embeddings for insights
insight_embeddings = model.encode(insights, convert_to_tensor=True)

# Streamlit UI
st.title("🧠 People Management Insights Q&A")

user_question = st.text_input("Ask a question about the company insights:")

if user_question:
    # Embed user question
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.cos_sim(question_embedding, insight_embeddings)[0]
    
    # Get highest similarity score
    max_idx = similarities.argmax()
    max_score = similarities[max_idx].item()
    
    # Threshold for matching (tune as needed)
    threshold = 0.5
    
    if max_score > threshold:
        st.success(insights[max_idx])
    else:
        st.warning("Further insights not supported till now.")

