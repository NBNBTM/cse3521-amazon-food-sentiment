# Amazon Food Review Sentiment Analysis

CSE 3521 · Autumn 2026 · Group Theta

A team project that classifies food review text as **negative** (1–2 stars), **neutral** (3 stars), or **positive** (4–5 stars), and compares model predictions in a Streamlit application.

## Current status

The initial app is deployed on **Streamlit Community Cloud**: [Open the live prototype](https://hgbwwg2ig7z6vw2xbgxjz9.streamlit.app/).

**Demo** mode trains Bag-of-Words + Multinomial Naive Bayes and TF-IDF + Logistic Regression on 18 invented food reviews for interface demonstrations. Select **Team models** and **Naive Bayes** to use the fitted pipeline trained on Amazon reviews.

The repository is **public**, as required by the project specification (pp. 1, 4, 8), and is owned and administered by **NBNBTM**. Anyone can view the code; direct write access remains limited to the owner and accepted collaborators. No teammates have been invited yet.

## Team

| Member | Primary role |
| --- | --- |
| Abdullah Felemban | Project Lead / Data Engineer |
| Ziwei Li | Data / ML Engineer |
| Lindsey Yang | App Engineer; repository owner |
| Jeffery Lin | Analyst / Writer |

## Run locally

Use Python **3.12**. From this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

The current Mac already has `.venv` installed. Double-click `run_app.command`, or run:

```sh
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local address printed by Streamlit. Choose an example or enter the review text and an optional summary, select models, and click **Analyze review**. The app makes no external API requests and has no review-upload or persistence feature. Model probabilities are uncalibrated.

## Model handoff

See [models/README.md](models/README.md). The feature/model owner should export the **entire fitted preprocessing + classifier pipeline**, using the pinned inference dependencies and string class names `negative`, `neutral`, `positive`:

- `models/naive_bayes.joblib`
- `models/logistic_regression.joblib`

The app sends `Summary + " " + Text` to the fitted pipeline. An omitted summary becomes an empty string. Use this format for training and exclude `Score` from the input. Team mode reports missing or incompatible files. Load team artifacts from trusted sources because joblib can execute Python code.

The deep model will need a separate adapter after the team supplies it. This prototype does not complete the required deep/modern model, full model comparison, or M3 report.

## Data and reproducibility

Group Theta uses the instructor-assigned **seed 49** to draw 50,000 reviews from the full Amazon Fine Food Reviews dataset. Generate `data/group_sample_seed49.csv` using the steps in [data/README.md](data/README.md), then share that file within the team for all project stages. The charter reports 46,391 reviews after removing duplicate texts and a sample positive-class proportion of 78.1%. Recalculate held-out evaluation metrics from the agreed split.

The [data and EDA notebook](notebooks/01_data_and_eda.ipynb) and [feature and baseline notebook](notebooks/02_features_and_nb_baseline.ipynb) use `data/Reviews.csv` to reproduce the initial analysis and Naive Bayes model. The baseline notebook removes duplicates before the stratified 80/20 split and fits features on the training data only. Reserve the test set for final evaluation. Demo pipelines are fitted in memory by `get_demo_models()`.

`requirements.txt` pins the direct runtime dependencies. `requirements-lock.txt` captures the complete tested Python environment. Use the lock file for an identical dependency set:

```sh
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

## Course basis and remaining work

- `Week 6.2/lecture06.pdf`, PDF pp. 4–10: Bag-of-Words, fixed vocabulary, n-grams, TF-IDF.
- `Week 9/lecture09.pdf`, PDF pp. 12–16: Naive Bayes and text/sentiment classification.
- `group_project.pdf`, p. 2: at least two feature representations, two classical models, one deep/modern model, evaluation, and a Streamlit app with two trained models.
- M1 is due Nov 10 at 11:59 pm under the professor's [updated deadline and group-seed announcement](https://osu.instructure.com/courses/208628/discussion_topics/4312999). `group_project.pdf`, pp. 4–5, lists M2 on Nov 15 and M3 on Nov 22 at 11:59 pm.
- `group_project.pdf`, pp. 6–7: final report/slides due Nov 29 at 11:59 pm; live demo date is **See Carmen**, not verified here.
- `group_project.pdf`, p. 8: repository organization, reproducibility, and continuing genuine contribution history.

Remaining team work: train and evaluate the other models on the shared split, export compatible artifacts, and complete the required report and screenshots.

## Deployment

Live prototype: https://hgbwwg2ig7z6vw2xbgxjz9.streamlit.app/

Streamlit Community Cloud uses repository `NBNBTM/cse3521-amazon-food-sentiment`, branch `main`, entrypoint `app.py`, and Python 3.12. Demo mode displays **DEMO ONLY**. Select **Team models** to exercise the exported model.

References: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest), [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html). Course materials above are read-only local references and are not republished in the repository.
