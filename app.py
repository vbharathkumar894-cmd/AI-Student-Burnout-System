
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Student Wellness AI",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------------------
# 2. LOAD DATA AND MODEL
# -----------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv("student_burnout_dataset.csv")

model, features = load_model()
df = load_data()

# -----------------------------------------
# 3. DASHBOARD HEADER
# -----------------------------------------

st.title("🎓 Student Wellness AI")
st.write(
    "AI-powered academic strain monitoring "
    "and early warning dashboard"
)

st.caption(
    "Educational demonstration using synthetic student data"
)

st.divider()

# -----------------------------------------
# 4. SIDEBAR
# -----------------------------------------

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    ["Overview", "Risk Prediction", "Dataset Analysis"]
)

st.sidebar.divider()

st.sidebar.info(
    "This tool provides educational risk estimates, "
    "not medical or psychological diagnoses."
)

# -----------------------------------------
# 5. OVERVIEW PAGE
# -----------------------------------------

if page == "Overview":

    st.header("📊 Dashboard Overview")

    total_students = len(df)

    risk_counts = (
        df["Burnout_Risk"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    low_count = risk_counts["Low"]
    medium_count = risk_counts["Medium"]
    high_count = risk_counts["High"]

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", total_students)
    c2.metric("Low Risk", low_count)
    c3.metric("Medium Risk", medium_count)
    c4.metric("High Risk", high_count)

    st.divider()

    # RISK DISTRIBUTION

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Distribution")

        chart_data = pd.DataFrame({
            "Risk Category": risk_counts.index,
            "Students": risk_counts.values
        })

        st.bar_chart(
            chart_data,
            x="Risk Category",
            y="Students"
        )

    with col2:

        st.subheader("Student Risk Percentages")

        percentage_data = pd.DataFrame({
            "Category": risk_counts.index,
            "Percentage": (
                risk_counts.values / total_students * 100
            )
        })

        st.dataframe(
            percentage_data,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # AVERAGE STUDENT STATISTICS

    st.subheader("📈 Student Statistics")

    a, b, c = st.columns(3)

    a.metric(
        "Average Study Hours",
        f"{df['Study_Hours'].mean():.2f}"
    )

    b.metric(
        "Average Sleep Hours",
        f"{df['Sleep_Hours'].mean():.2f}"
    )

    c.metric(
        "Average Stress Level",
        f"{df['Stress_Level'].mean():.2f}/10"
    )

# -----------------------------------------
# 6. RISK PREDICTION PAGE
# -----------------------------------------

elif page == "Risk Prediction":

    st.header("🔍 Student Risk Prediction")

    st.write("Enter student details below.")

    col1, col2 = st.columns(2)

    with col1:

        study_hours = st.slider(
            "Daily Study Hours",
            1, 12, 6
        )

        sleep_hours = st.slider(
            "Daily Sleep Hours",
            3, 9, 7
        )

        attendance = st.slider(
            "Attendance Percentage",
            40, 100, 75
        )

    with col2:

        breaks = st.slider(
            "Breaks Per Day",
            0, 6, 3
        )

        stress_level = st.slider(
            "Self-Reported Stress",
            1, 10, 5
        )

        workload = st.slider(
            "Assignment Workload",
            1, 10, 5
        )

    if st.button(
        "Predict Academic Strain Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{
            "Study_Hours": study_hours,
            "Sleep_Hours": sleep_hours,
            "Attendance": attendance,
            "Breaks": breaks,
            "Stress_Level": stress_level,
            "Workload": workload
        }])

        input_data = input_data[features]

        prediction = model.predict(input_data)[0]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == "Low":
            st.success("Low Academic Strain Risk")

        elif prediction == "Medium":
            st.warning("Medium Academic Strain Risk")

        else:
            st.error("High Academic Strain Risk")

        st.subheader("💡 Helpful Suggestions")

        suggestions = []

        if sleep_hours < 7:
            suggestions.append(
                "Maintain a regular and sufficient sleep schedule."
            )

        if study_hours > 8:
            suggestions.append(
                "Divide study sessions into smaller sessions."
            )

        if breaks < 2:
            suggestions.append(
                "Take regular breaks while studying."
            )

        if stress_level >= 7:
            suggestions.append(
                "Try relaxation activities and talk to someone you trust."
            )

        if attendance < 65:
            suggestions.append(
                "Identify barriers to attendance and plan improvements."
            )

        if workload >= 8:
            suggestions.append(
                "Divide large assignments into manageable tasks."
            )

        if not suggestions:
            suggestions.append(
                "Continue maintaining balanced daily routines."
            )

        for suggestion in suggestions:
            st.write("✅", suggestion)

        st.subheader("Student Input Summary")

        st.dataframe(input_data, use_container_width=True)

        st.caption(
            "The prediction is an educational estimate based on "
            "synthetic training labels, not a diagnosis."
        )

# -----------------------------------------
# 7. DATASET ANALYSIS PAGE
# -----------------------------------------

elif page == "Dataset Analysis":

    st.header("📁 Dataset Analysis")

    st.write("Explore the synthetic student dataset.")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Feature Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Risk Category Distribution")

    st.bar_chart(
        df["Burnout_Risk"].value_counts()
    )

    st.subheader("Study Hours Analysis")

    st.bar_chart(
        df.groupby("Burnout_Risk")["Study_Hours"].mean()
    )

    st.subheader("Download Dataset")

    st.download_button(
        "Download Student Dataset CSV",
        data=df.to_csv(index=False),
        file_name="student_burnout_dataset.csv",
        mime="text/csv"
    )

# -----------------------------------------
# 8. FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "Student Wellness AI | Python | Pandas | "
    "Scikit-learn | Streamlit"
)
