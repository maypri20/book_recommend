## Book Recommendation System

A Content-Based Book Recommendation System** built using Python, Streamlit, and Scikit-learn, designed to suggest similar books based on **genre, author, rating, and publication year.  
The system computes Cosine Similarity between books after feature engineering (One-Hot Encoding + Standard Scaling) to generate personalized recommendations.


Presentation Link: https://docs.google.com/presentation/d/1MnuaB9OxC-V0bp87MpTXUwattlxt06ytp41ZPH7dhkE/edit?usp=sharing


##  Features

- 🔍 Search for a book by partial or full title  
- 📖 Get **top N similar books** based on feature similarity  
- 🧠 Uses **One-Hot Encoding** and **StandardScaler** for feature transformation  
- 📏 Calculates **Cosine Similarity** to find related books  
- 🧾 Displays book details: title, author, genre, rating, and Goodreads/OpenLibrary link  
- 🖼️ Shows cover image (when available)  
- 📊 Includes an interactive **EDA Dashboard**:
  - Genre & Author distribution  
  - Rating histogram  
  - Publication trends (century & decade)  
  - Summary metrics  

---

##  Project Workflow

```
flowchart LR
A[Raw Dataset (Goodreads & OpenLibrary)] --> B[Data Cleaning & Preprocessing]
B --> C[Feature Engineering (Encoding + Scaling)]
C --> D[Cosine Similarity Matrix]
D --> E[Recommendation Function]
E --> F[Streamlit Web App + Dashboard]

| Category             | Tools & Libraries                                               |
| -------------------- | --------------------------------------------------------------- |
| **Language**         | Python                                                          |
| **Framework**        | Streamlit                                                       |
| **Data Handling**    | Pandas, NumPy                                                   |
| **Machine Learning** | Scikit-learn (OneHotEncoder, StandardScaler, cosine_similarity) |
| **Visualization**    | Altair, Matplotlib, Seaborn                                     |
| **Dataset Sources**  | Goodreads & OpenLibrary                                         |
```
Project By:
Priyanka Marmath
Görlitz, Germany
