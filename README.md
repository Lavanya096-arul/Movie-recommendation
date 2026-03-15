# 🎬 Movie Recommendation System

## 📌 Overview

The **Movie Recommendation System** is a machine learning-based application that suggests movies similar to a selected movie.
It uses content-based filtering techniques to analyze movie features and recommend related movies to users.

The application provides an interactive web interface built using **Streamlit**, allowing users to easily select a movie and receive recommendations instantly.

---

## 🚀 Features

* Content-based movie recommendation
* Interactive user interface
* Fast movie similarity computation
* Dataset-based movie suggestions
* Simple and intuitive web application

---

## 🧠 Technology Stack

* **Programming Language:** Python
* **Framework:** Streamlit
* **Libraries Used:**

  * Pandas
  * Scikit-learn
  * NumPy
  * Pickle

---

## 📂 Project Structure

```
movie-recommendation/
│
├── app.py
│__load.py
│
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
│
└── README.md
```

---

## 📊 Dataset

This project uses the **TMDB 5000 Movie Dataset**, which contains:

* Movie titles
* Genres
* Cast information
* Crew details
* Keywords
* Movie descriptions

These features are processed to calculate similarity between movies.

---

## ⚙️ How the Recommendation Works

1. Movie data is loaded from the dataset.
2. Important features such as genres, keywords, cast, and overview are combined.
3. Text vectorization is applied using **CountVectorizer**.
4. **Cosine similarity** is computed between movies.
5. When a user selects a movie, the system returns the most similar movies.

---

## ▶️ Running the Application

### Step 1: Install dependencies

```
pip install pandas scikit-learn streamlit numpy
```

### Step 2: Run the Streamlit app

```
streamlit run app2.py
```

### Step 3: Open in Browser

```
http://localhost:8501
```

---

## 📸 Example Output

Example recommendation for the movie **Avatar**:

* Apollo 18
* Beowulf
* The Helix... Loaded
* The American
* The Adventures of Pluto Nash

---

## 💡 Future Improvements

* Add movie posters using TMDB API
* Improve recommendation accuracy
* Deploy the application online
* Add user rating-based recommendations

---

## 👨‍💻 Author

**Lavanya**

---

## 📜 License

This project is created for educational and learning purposes.
