# 🚗 Car Price Prediction using Multiple Linear Regression

A Machine Learning project that predicts the estimated price of a used car based on user inputs such as engine capacity, kilometers driven, ownership, car age, brand, and fuel type.

---

## 📌 Project Overview

This project uses **Multiple Linear Regression** to predict car prices. The dataset is cleaned, preprocessed, encoded, scaled, and then used to train a machine learning model. Users can enter car details, and the model predicts the estimated price.

---

## 🚀 Features

- Handle Missing Values
- Feature Engineering (Car Age)
- Ownership Mapping
- One-Hot Encoding
- Feature Scaling using StandardScaler
- Multiple Linear Regression Model
- User Input Based Prediction
- Model Evaluation using Actual vs Predicted Scatter Plot
- Save Model using Joblib

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Dataset Features

| Feature | Description |
|---------|-------------|
| Brand | Car Brand |
| Fuel Type | Petrol/Diesel/Electric |
| Engine Capacity | Engine size (CC) |
| KM Driven | Total kilometers driven |
| Ownership | 1st, 2nd, 3rd, or 4th Owner |
| Car Age | Calculated using Manufacturing Year |
| Price | Target Variable |

---

## 🔄 Project Workflow

1. Load Dataset
2. Handle Missing Values
3. Feature Engineering (Car Age)
4. Ownership Mapping
5. One-Hot Encoding
6. Train-Test Split
7. Feature Scaling
8. Train Multiple Linear Regression Model
9. Predict Car Price
10. Visualize Actual vs Predicted Price
11. Save Model using Joblib

---

## 📊 Visualization

The project includes a scatter plot comparing:

- Actual Car Price
- Predicted Car Price

This helps visualize the model's prediction performance.

---

## 💻 User Input Example

```text
Enter Engine Capacity (CC): 1498
Enter KM Driven: 25000
Enter Ownership (1-4): 1
Enter Car Age: 3
Enter Brand: Honda
Enter Fuel Type: Petrol
```

### Output

```text
Predicted Car Price: ₹10,68,962.31
```

---

## 📁 Project Structure

```
Car-Price-Prediction-ML/
│
├── README.md
├── requirements.txt
├── .gitignore
├── car_dataset.csv
├── Car_Price_Prediction.py
├── car_price_model.joblib
├── scaler.joblib
```

---

## 🎯 Learning Outcomes

Through this project, I learned:

- Data Cleaning
- Missing Value Handling
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Multiple Linear Regression
- Model Prediction
- Data Visualization using Seaborn
- Saving ML Models using Joblib

---

## 👨‍💻 Author

**Suraj Tiwari**

GitHub: https://github.com/YOUR_USERNAME

LinkedIn: https://www.linkedin.com/in/YOUR_LINKEDIN/

---

⭐ If you found this project useful, don't forget to give it a Star!