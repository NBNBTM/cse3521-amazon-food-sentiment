# Amazon Food Review Sentiment Analysis

CSE 3521 · Autumn 2026 · Group Theta

A team project that classifies food review text as **negative** (1–2 stars), **neutral** (3 stars), or **positive** (4–5 stars), and compares model predictions in a Streamlit application.

## Current status

The app is deployed on **Streamlit Community Cloud**. [Open the live app](https://hgbwwg2ig7z6vw2xbgxjz9.streamlit.app/).

**Demo** mode fits two classifiers to 18 invented reviews and displays **DEMO ONLY**. **Team models** compares fitted Naive Bayes and Logistic Regression pipelines trained on Amazon Fine Food Reviews. A feedforward neural network becomes available when its fitted pipeline is present.

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

Open the local address printed by Streamlit and select **Team models**. Enter the review text and an optional summary, or choose an example. Select the models to compare and click **Analyze review**. The results show each model's predicted sentiment and uncalibrated class probabilities side by side.

## Model handoff

See [models/README.md](models/README.md). The feature/model owner should export the **entire fitted preprocessing + classifier pipeline**, using the pinned inference dependencies and string class names `negative`, `neutral`, `positive`:

- `models/naive_bayes.joblib`
- `models/logistic_regression.joblib`
- `models/feedforward_nn.joblib`

The app sends `Summary + " " + Text` to the fitted pipeline. An omitted summary becomes an empty string. Use this format for training and exclude `Score` from the input. Team mode lists the exported files available to the app and reports loading or prediction errors. Load team artifacts from trusted sources because joblib can execute Python code.

## Data and reproducibility

Download the full Amazon Fine Food Reviews dataset using [data/README.md](data/README.md) and put the original `Reviews.csv` in `data/`. The training scripts draw the 50,000-review **seed-49** sample, remove duplicate texts, and use the same stratified split across models.

Run the validation and final evaluation scripts from the repository root. See [the training guide](training/README.md) for the script protocol and output files.

`requirements.txt` pins the direct runtime dependencies. `requirements-lock.txt` captures the complete tested Python environment. Use the lock file for an identical dependency set:

```sh
python -m pip install -r requirements-lock.txt
python training/run_tuning.py data/Reviews.csv
python training/run_final.py data/Reviews.csv models
python -m unittest discover -s tests -v
```

## Course basis and remaining work

- `Week 6.2/lecture06.pdf`, PDF pp. 4–10: Bag-of-Words, fixed vocabulary, n-grams, TF-IDF.
- `Week 9/lecture09.pdf`, PDF pp. 12–16: Naive Bayes and text/sentiment classification.
- `group_project.pdf`, p. 2: at least two feature representations, two classical models, one deep/modern model, evaluation, and a Streamlit app with two trained models.
- See the [M3 assignment on Carmen](https://osu.instructure.com/courses/208628/assignments/5746369) for its submission deadline.
- `group_project.pdf`, pp. 6–7: final report/slides due Nov 29 at 11:59 pm; live demo date is **See Carmen**, not verified here.
- `group_project.pdf`, p. 8: repository organization, reproducibility, and continuing genuine contribution history.

## Deployment

Live prototype: https://hgbwwg2ig7z6vw2xbgxjz9.streamlit.app/

Streamlit Community Cloud uses repository `NBNBTM/cse3521-amazon-food-sentiment`, branch `main`, entrypoint `app.py`, and Python 3.12.

References: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest), [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html). Course materials above are read-only local references and are not republished in the repository.
