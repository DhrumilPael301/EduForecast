import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# set page Configuration
st.set_page_config(
    layout="wide",
    page_title="Edu Forecast",
    page_icon="auctus tech logo.png"
)
df = pd.read_csv("StudentsPerformance.csv")
with st.sidebar:
    select = option_menu(
        "Main Menu",
        ["Home", "Prediction", "Dataset", "About"],
        icons=["house", "cpu", "table", "info-circle"],
        default_index=0,
        menu_icon="cast"
    )
# home

if select == "Home":

    st.title("Edu Forecast")
    st.subheader("AI-Based Academic Prediction Tool")

    st.write("""
Welcome to **EduForecast**, an AI-powered Academic Prediction System.

This platform analyzes student academic performance using demographic,
educational, and examination data to predict whether a student is likely
to perform well or may require additional academic support.
""")

    st.divider()

#metrics

    df_home = df.copy()

    df_home["Average_Score"] = (
        df_home["math score"] +
        df_home["reading score"] +
        df_home["writing score"]
    ) / 3

    total_students = len(df_home)
    male_students = (df_home["gender"] == "male").sum()
    female_students = (df_home["gender"] == "female").sum()
    avg_score = round(df_home["Average_Score"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👨‍🎓 Total Students", total_students)
    col2.metric("👦 Male Students", male_students)
    col3.metric("👧 Female Students", female_students)
    col4.metric("📊 Average Score", avg_score)

    st.divider()

    st.header("Dashboard Overview")

#row 1

    fig1, fig2 = st.columns(2)

    with fig1:
        st.subheader("Gender Distribution")
        gender = df_home["gender"].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(gender.index, gender.values)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Students")
        st.pyplot(fig)
    with fig2:
        st.subheader("Lunch Type Distribution")
        lunch = df_home["lunch"].value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            lunch.values,
            labels=lunch.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

    st.divider()

#row 2
    fig3, fig4 = st.columns(2)

    with fig3:
        st.subheader("Average Subject Scores")
        avg_scores = {
            "Math": df_home["math score"].mean(),
            "Reading": df_home["reading score"].mean(),
            "Writing": df_home["writing score"].mean()
        }
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(
            avg_scores.keys(),
            avg_scores.values()
        )
        ax.set_xlabel("Subjects")
        ax.set_ylabel("Average Score")
        st.pyplot(fig)
    with fig4:
        st.subheader("Test Preparation Course")
        prep = df_home["test preparation course"].value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            prep.values,
            labels=prep.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

    st.divider()

#quick insights
    st.subheader("Quick Insights")

    st.success(f"""
✔ Total Students : {total_students}

✔ Average Score : {avg_score}

✔ Male Students : {male_students}

✔ Female Students : {female_students}

✔ Subjects Analysed : Math, Reading & Writing

✔ AI Model : Random Forest Classifier
""")

    st.divider()
elif select == "Prediction":
    data = df.copy()
    st.title("Prediction")
    st.subheader("AI-Based Academic Prediction Tool")

    def score_to_level(score):
        if score < 50:
            return "Poor"
        elif score < 70:
            return "Average"
        elif score < 85:
            return "Good"
        else:
            return "Excellent"

    data["performance_level"] = data["math score"].apply(score_to_level)

    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_parent_edu = LabelEncoder()
    le_lunch = LabelEncoder()
    le_test_prep = LabelEncoder()
    le_target = LabelEncoder()

    data["gender_enc"] = le_gender.fit_transform(data["gender"])
    data["race_enc"] = le_race.fit_transform(data["race/ethnicity"])
    data["parent_edu_enc"] = le_parent_edu.fit_transform(data["parental level of education"])
    data["lunch_enc"] = le_lunch.fit_transform(data["lunch"])
    data["test_prep_enc"] = le_test_prep.fit_transform(data["test preparation course"])
    data["performance_level_enc"] = le_target.fit_transform(data["performance_level"])

    X = data[[
        "gender_enc", "race_enc", "parent_edu_enc",
        "lunch_enc", "test_prep_enc",
        "reading score", "writing score"
    ]]
    y = data["performance_level_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    med_reading = int(data["reading score"].median())
    med_writing = int(data["writing score"].median())

    st.divider()

    st.markdown("### 👤 Student Background Information")
    st.markdown(
        "ℹ️ Fill in all fields carefully — each option exactly matches "
        "the categories present in the dataset for accurate prediction."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🧑 Personal Details")
        gender = st.selectbox(
            "🚻 Gender",
            options=sorted(le_gender.classes_),
        )
        race = st.selectbox(
            "🌍 Race / Ethnicity Group",
            options=sorted(le_race.classes_),
        )
    with col2:
        st.markdown("🏫 Academic Background")
        parent_edu = st.selectbox(
            "🎓 Parental Level of Education",
            options=le_parent_edu.classes_,
        )
        lunch = st.selectbox(
            "🍱 Lunch Type",
            options=sorted(le_lunch.classes_),
        )
    st.markdown("")
    test_prep = st.selectbox(
        "📝 Did the Student Complete a Test Preparation Course?",
        options=sorted(le_test_prep.classes_),
    )
    st.divider()
    if st.button("🚀 Predict My Math Performance Level!", use_container_width=True):
        sample = pd.DataFrame({
            "gender_enc":[le_gender.transform([gender])[0]],
            "race_enc":[le_race.transform([race])[0]],
            "parent_edu_enc":[le_parent_edu.transform([parent_edu])[0]],
            "lunch_enc":[le_lunch.transform([lunch])[0]],
            "test_prep_enc":[le_test_prep.transform([test_prep])[0]],
            "reading score":[med_reading],
            "writing score":[med_writing],
        })
        probs  = model.predict_proba(sample)[0]
        result = pd.DataFrame({
            "Performance Level":le_target.classes_,
            "Confidence":probs
        }).sort_values("Confidence", ascending=False).reset_index(drop=True)
        top = result.iloc[0]
        st.divider()
        st.markdown("🎯 Prediction Result")
        st.subheader(top["Performance Level"])

elif select == "Dataset":

    st.title("Dataset Overview")
    st.write("Explore the student dataset used to train the Machine Learning model.")
    st.divider()
#dataset metrics
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    male_students = (df["gender"] == "male").sum()
    female_students = (df["gender"] == "female").sum()
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📄 Rows", total_rows)
    col2.metric("📑 Columns", total_columns)
    col3.metric("👦 Male Students", male_students)
    col4.metric("👧 Female Students", female_students)

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    st.divider()

elif select == "About":

    st.title("About EduForecast")
    st.write("Learn more about the project, dataset, machine learning model, and technologies used.")
    st.divider()

#project description
    st.header("Project Description")

    st.write("""
    EduForecast is an Artificial Intelligence and Machine Learning based
    Academic Prediction System developed to analyze student performance.

    The application predicts whether a student is likely to perform well
    or may require academic support using demographic information,
    educational background, and examination scores.

    This project demonstrates the practical implementation of Machine
    Learning for Educational Data Analysis.
    """)
    st.divider()

#project objective
    st.header("Project Objective")

    st.write("""
    • Predict student academic performance.

    • Analyze student learning behaviour.

    • Identify students requiring academic support.

    • Assist teachers in student evaluation.

    • Improve educational decision making.
    """)
    st.divider()

# dataset information
    st.header("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:

        st.metric("Total Records", len(df))
        st.metric("Male Students", (df["gender"] == "male").sum())
        st.metric("Female Students", (df["gender"] == "female").sum())

    with col2:

        st.metric("Features", len(df.columns))
        st.metric("Target Variable", "Performance")
        st.metric("ML Task", "Classification")

    st.divider()

# machine learning algorithm
    st.header("Machine Learning Algorithm")

    st.write("""
    Random Forest Classifier

    Random Forest is a supervised Machine Learning algorithm used for
    classification problems.

    It builds multiple decision trees and combines their predictions
    to produce accurate and reliable results.

    Reasons for selecting Random Forest:

    ✔ High Accuracy

    ✔ Fast Prediction

    ✔ Less Overfitting

    ✔ Handles Large Dataset

    ✔ Suitable for Classification Problems
    """)
    st.divider()

#technologies used
    st.header("Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - Matplotlib
    - Scikit-Learn
    - Streamlit Option Menu
    """)
    st.divider()

#advantages
    st.header("Advantages")

    st.write("""
    • Predicts student academic performance.

    • Easy to use interface.

    • Fast Machine Learning prediction.

    • Interactive dashboard.

    • Useful for teachers and educational institutions.
    """)
    st.divider()

#future scope
    st.header("Future Scope")

    st.write("""
    • Student Login System

    • Teacher Dashboard

    • Attendance Analysis

    • AI-Based Performance Report

    • College ERP Integration

    • Real-Time Academic Monitoring

    • Deep Learning Based Prediction
    """)
    st.divider()

#developer
    st.header("Developer")

    st.success("""
    Project Title : EduForecast

    AI-Based Academic Prediction Tool

    Developed By : Dhrumil Patel

    Developed Using Python, Streamlit and Machine Learning

    Machine Learning Algorithm : Random Forest Classifier
    """)
    st.divider()
