#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)

#######################
# Page config
st.set_page_config(
    page_title="Titanic Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.themes.enable("default")

#######################
# Load data
df = pd.read_csv("titanic.csv")

#######################
# Sidebar
with st.sidebar:
    st.title("Titanic Survival Analysis Dashboard")
    st.header("데이터 필터")

    pclass_filter = st.multiselect("Pclass 선택", [1, 2, 3], default=[1, 2, 3])
    sex_filter = st.multiselect("성별 선택", ["male", "female"], default=["male", "female"])
    embarked_filter = st.multiselect("탑승지 선택", ["C", "Q", "S"], default=["C", "Q", "S"])

    st.header("결측치 처리 옵션")
    missing_option = st.selectbox(
        "결측치 처리 방법 선택",
        ["제거", "평균 대체", "중앙값 대체", "최빈값 대체", "처리하지 않음"]
    )

    st.header("머신러닝 기법 선택")
    ml_method = st.multiselect(
        "사용할 ML 기법",
        ["분류(Classification)", "회귀(Regression)", "군집(Clustering)"]
    )

    run_analysis = st.button("분석 실행")

#######################
# Dashboard Layout
col = st.columns((1.5, 4.5, 2))

###############################################
# Column 1 : Summary
###############################################
with col[0]:
    st.subheader("요약 지표")

    total_passengers = len(df)
    survived_rate = df["Survived"].mean() * 100
    avg_age = df["Age"].mean()
    avg_fare = df["Fare"].mean()

    st.metric("전체 승객 수", f"{total_passengers:,}")
    st.metric("생존율", f"{survived_rate:.1f}%")
    st.metric("평균 나이", f"{avg_age:.1f} 세")
    st.metric("평균 요금 (Fare)", f"{avg_fare:.2f}")

    st.markdown("---")

    st.subheader("성별 생존율")
    sex_survival = df.groupby("Sex")["Survived"].mean() * 100
    st.write(pd.DataFrame({"생존율(%)": sex_survival.round(1)}))

    st.markdown("---")

    st.subheader("Pclass별 생존율")
    class_survival = df.groupby("Pclass")["Survived"].mean() * 100
    st.write(pd.DataFrame({"생존율(%)": class_survival.round(1)}))

###############################################
# Column 2 : Visualization
###############################################
with col[1]:
    st.subheader("시각화 분석")

    st.markdown("### 상관계수 히트맵")
    numeric_cols = ["Survived", "Age", "Fare", "SibSp", "Parch", "Pclass"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    st.markdown("### 연령 분포 (Age Histogram)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Age"], kde=True, bins=20, ax=ax2)
    st.pyplot(fig2)

    st.markdown("---")

    st.markdown("### Pclass × Sex 생존율 히트맵")
    pivot_table = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot_table, annot=True, cmap="Greens", fmt=".2f")
    st.pyplot(fig3)

###############################################
# Column 3 : ML + Details
###############################################
with col[2]:
    st.subheader("상세 분석 및 머신러닝 결과")

    st.markdown("### 생존/비생존 그룹 통계")
    group_stats = df.groupby("Survived")[["Age", "Fare", "SibSp", "Parch"]].mean()
    group_stats = group_stats.rename(index={0: "비생존", 1: "생존"})
    st.dataframe(group_stats)
    st.markdown("---")

    st.subheader("### 머신러닝 분석 결과")

    ############################
    # Classification
    ############################
    if "분류(Classification)" in ml_method:
        st.markdown("#### 분류 모델 (Logistic Regression)")

        X = df[["Pclass", "Age", "Fare", "SibSp", "Parch"]].copy()

        # 숫자형 변환
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.mean())

        y = df["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.write("Accuracy:", round(accuracy_score(y_test, preds), 3))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, preds))
        st.text(classification_report(y_test, preds))

        st.markdown("---")

    ############################
    # Regression
    ############################
    if "회귀(Regression)" in ml_method:
        st.markdown("#### 회귀 모델 (Fare 예측)")

        reg_df = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]].copy()
        reg_df = reg_df.apply(pd.to_numeric, errors="coerce").dropna()

        X = reg_df.drop("Fare", axis=1)
        y = reg_df["Fare"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)
        pred = reg_model.predict(X_test)

        rmse = mean_squared_error(y_test, pred, squared=False)
        mae = mean_absolute_error(y_test, pred)

        st.write("RMSE:", round(rmse, 3))
        st.write("MAE:", round(mae, 3))

        st.markdown("---")

    ############################
    # Clustering
    ############################
    if "군집(Clustering)" in ml_method:
        st.markdown("#### 군집 모델 (K-Means)")

        cluster_data = df[["Age", "Fare", "Pclass", "SibSp", "Parch"]].copy()
        cluster_data = cluster_data.apply(pd.to_numeric, errors="coerce").dropna()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_data)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled)

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(pca_data)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        st.pyplot(fig)

        cluster_summary = pd.DataFrame({
            "클러스터": labels,
            "Age": cluster_data["Age"].values,
            "Fare": cluster_data["Fare"].values
        }).groupby("클러스터").mean()

        st.dataframe(cluster_summary)
