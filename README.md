# ✈️ FlyPredict – Flight Fare Prediction App  

🚀 A Machine Learning powered web application built using **Streamlit** to predict **flight ticket prices** and compare multiple regression models in real time.

🔗 **Live App:**  
👉 https://flypredict.streamlit.app/

---

## 📌 Project Overview  

**FlyPredict** is an end-to-end **Predictive Analytics & Machine Learning project** that predicts airline ticket prices using real-world features such as:

- Airline  
- Source & Destination  
- Departure & Arrival Time  
- Number of Stops  
- Travel Class  
- Journey Duration  
- Days Left for Booking  

This project demonstrates:
- Complete ML pipeline from preprocessing to deployment  
- Multi-model comparison  
- Interactive visual dashboards  
- A production-ready Streamlit application  

---

## 📘 Notebook Workflow  

This project follows a structured machine learning workflow:

1. **Importing Libraries** – Essential Python libraries imported  
2. **Loading the Dataset** – Cleaned flight dataset loaded  
3. **Data Preprocessing** – Handling missing values & removing unwanted columns  
4. **Feature Engineering** – Encoding categorical features  
5. **Exploratory Data Analysis (EDA)** – Data visualization & pattern analysis  
6. **Splitting Data** – Train-test split  
7. **Model Training** – Training multiple regression models  
8. **Model Evaluation** – Performance evaluation using metrics  
9. **Model Comparison** – Selecting the best performing model  
10. **Feature Importance** – Identifying key influencing factors  
11. **Predictions** – Manual input-based prediction  
12. **Conclusion** – Final insights and recommendations  

---

## 🧠 Machine Learning Models Used  

| Model | Description |
|--------|-------------|
| ✅ Random Forest Regressor | High-accuracy ensemble model |
| ✅ K-Nearest Neighbors (KNN) | Pattern-based prediction |
| ✅ Decision Tree Regressor | Rule-based learning |
| ✅ Linear Regression | Baseline comparison model |

✅ All models are optimized for:
- Small file size  
- Fast prediction  
- GitHub & Streamlit Cloud compatibility  

---

## ✅ Key Features  

✅ Real-time flight fare prediction  
✅ Comparison of 4 ML models  
✅ Interactive dark-themed UI  
✅ Route, airline, class & timing based prediction  
✅ Market insights & analytics  
✅ Optimized model size for deployment  
✅ Streamlit Cloud ready  

---

## 🗂️ Project Structure  

```text
FlyPredict/
│
├── Config/
│ └── feature_order.json
│
├── Dataset/
│ └── Clean_Dataset.csv
│
├── Models/
│ ├── model_rf.pkl
│ ├── model_dt.pkl
│ ├── model_knn.pkl
│ ├── model_lr.pkl
│ ├── encoders.pkl
│ └── scaler.pkl
│
├── app.py
├── FlightFarePrediction.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack  

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Plotly  
- Joblib  

---

## 📊 Visualizations Included  

- Airline Market Share  
- Flight Class Distribution  
- Price Distribution by Airline  
- Booking Day vs Price Trend  
- Duration vs Price Scatter Plot  
- Model-wise Price Comparison  

---

## ▶️ How to Run Locally  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/FlyPredict.git
cd FlyPredict
```

### 2️⃣ Create Virtual Environment (Optional)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
streamlit run app.py
```

## 👨‍💻 Author

Ayush Kumar
🎓 B.Tech CSE – Lovely Professional University
📊 Aspiring Data Analyst & Machine Learning Enthusiast

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!

## 📝 License

This project is licensed under the MIT License.
