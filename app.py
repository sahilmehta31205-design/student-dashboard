
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Student Performance AI", layout="wide", page_icon="🎓")

@st.cache_data
def load_data():
    df = pd.read_csv("student_performance_updated_1000.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    def label(g):
        if g >= 85:   return "Excellent"
        elif g >= 70: return "Good"
        elif g >= 50: return "Average"
        else:         return "At Risk"
    df["performance_label"] = df["finalgrade"].apply(label)
    return df

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

df    = load_data()
model = load_model()

st.sidebar.title("🎓 Student Dashboard")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🔮 Predict", "📈 Analytics", "🏆 Feature Importance"])

if page == "📊 Overview":
    st.title("📊 Student Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Avg Final Grade", f"{df['finalgrade'].mean():.1f}")
    col3.metric("At Risk Students", len(df[df["performance_label"] == "At Risk"]))
    col4.metric("Excellent Students", len(df[df["performance_label"] == "Excellent"]))
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="performance_label", title="Performance Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x="finalgrade", nbins=20, title="Final Grade Distribution",
                            color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("📋 Student Data Table")
    st.dataframe(df[["name","gender","attendancerate","studyhoursperweek",
                      "previousgrade","finalgrade","performance_label"]].head(20),
                 use_container_width=True)

elif page == "🔮 Predict":
    st.title("🔮 Predict Student Performance")
    col1, col2 = st.columns(2)
    with col1:
        attendance  = st.slider("Attendance Rate (%)", 0, 100, 75)
        study_hours = st.slider("Study Hours / Week",  0,  40, 10)
    with col2:
        prev_grade  = st.slider("Previous Grade", 0, 100, 70)
        extra       = st.selectbox("Extracurricular Activities", [0, 1, 2, 3], index=1)
    if st.button("🚀 Predict Now", use_container_width=True):
        features    = np.array([[attendance, study_hours, prev_grade, extra]])
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features).max() * 100
        color = {"Excellent":"🟢","Good":"🔵","Average":"🟡","At Risk":"🔴"}
        st.success(f"{color.get(prediction,'⚪')} Predicted: **{prediction}**")
        st.metric("Confidence", f"{probability:.1f}%")
        tips = {
            "At Risk":   "⚠️ Increase attendance, study hours, and seek parental support.",
            "Average":   "📚 Try extracurricular activities and revise regularly.",
            "Good":      "👍 Keep it up! A bit more effort can push to Excellent.",
            "Excellent": "🌟 Outstanding! Maintain this performance."
        }
        st.info(tips[prediction])

elif page == "📈 Analytics":
    st.title("📈 Class Analytics")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="attendancerate", y="finalgrade", color="performance_label",
                         title="Attendance vs Final Grade")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(df, x="studyhoursperweek", y="finalgrade", color="performance_label",
                          title="Study Hours vs Final Grade")
        st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.box(df, x="performance_label", y="finalgrade", color="performance_label",
                  title="Grade Distribution by Performance Label")
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = px.bar(df.groupby("gender")["finalgrade"].mean().reset_index(),
                  x="gender", y="finalgrade", title="Average Grade by Gender", color="gender")
    st.plotly_chart(fig4, use_container_width=True)

elif page == "🏆 Feature Importance":
    st.title("🏆 Feature Importance")
    features = ["attendancerate","studyhoursperweek","previousgrade","extracurricularactivities"]
    importance_df = pd.DataFrame({
        "Feature":    features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig = px.bar(importance_df, x="Feature", y="Importance", color="Importance",
                 title="Feature Importance from Random Forest",
                 color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(importance_df, use_container_width=True)
