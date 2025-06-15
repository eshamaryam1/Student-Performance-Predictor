# 🎓 Student Performance Predictor 📊

A web-based interactive tool to analyze and predict student performance based on demographic and academic indicators, built using **Python, Streamlit**, and **scikit-learn**.

---

## 🚀 Live Demo
Try the app live on Hugging Face Spaces:  
👉 [Student Performance Predictor](https://huggingface.co/spaces/eshamaryam/Student-Performance-Predictor)

---

## 📁 About the Dataset

This project uses the **StudentsPerformance.csv** dataset, which contains the following fields:

- Gender
- Race/ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Math score
- Reading score
- Writing score

Each row represents one student's academic results and demographic information.

---

## 🔍 Key Features

✅ Perform Exploratory Data Analysis (EDA)  
✅ Visualize trends and distributions with Seaborn & Matplotlib  
✅ Train a regression model to predict math scores  
✅ Real-time predictions based on user input  
✅ Clean, user-friendly Streamlit interface  

---

## 🧪 Exploratory Data Analysis (EDA)

The EDA includes:

- Summary statistics
- Histograms and box plots
- Correlation heatmaps
- Group-wise average scores
- Outlier detection
- Pairwise relationships

> All visualizations are integrated in the app for easy exploration!

---

## 🤖 Machine Learning

I used a **Linear Regression model** to predict a student’s math score based on other features. The model is trained with:

- Encoded categorical features (LabelEncoder)
- Normalized numeric values (StandardScaler)
- Model evaluation metrics:
  - R² Score
  - RMSE (Root Mean Squared Error)

---

## 🛠️ Tech Stack

| Tool            | Purpose                      |
|-----------------|------------------------------|
| Python          | Core programming language    |
| Pandas          | Data manipulation            |
| Seaborn/Matplotlib | Visualizations            |
| Scikit-learn    | Machine Learning             |
| Streamlit       | Web interface & deployment   |

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/eshamaryam1/Student-Performance-Predictor.git
cd Student-Performance-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run streamlit_app.py
```
---

## 🙋‍♀️ Author
### Esha Maryam
📍 Data Science Student 
<br> 
📬 [eshamaryam19em@gmail.com]

---

## 📝 License
This project is licensed under the MIT License.
