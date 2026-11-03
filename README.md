# Amazon Food Review Sentiment Analysis

CSE 3521 · Autumn 2026 · Group 8

A team project that classifies food review text as **negative** (1–2 stars), **neutral** (3 stars), or **positive** (4–5 stars), and compares model predictions in a Streamlit application.

## Current status

The initial app is runnable locally. **Demo** mode trains Bag-of-Words + Multinomial Naive Bayes and TF-IDF + Logistic Regression on 18 invented food reviews. These are interface demonstrations, not Amazon-trained project models, held-out evaluation, or evidence of predictive quality. **Team models** mode accepts complete, trusted fitted pipelines; no official artifacts are included yet.

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

Open the local address printed by Streamlit. Choose an example or paste a review, select models, and click **Analyze review**. The app makes no external AI/API requests and has no review-upload or persistence feature. Probabilities are uncalibrated model outputs, not a guarantee that the prediction is correct.

## Model handoff

See [models/README.md](models/README.md). The feature/model owner should export the **entire fitted preprocessing + classifier pipeline**, using the pinned inference dependencies and string class names `negative`, `neutral`, `positive`:

- `models/naive_bayes.joblib`
- `models/logistic_regression.joblib`

The app forwards the entered string unchanged. Agree on `Summary + Text` composition before training, consistent with the charter. Never use `Score` as a feature. Team mode reports missing/incompatible files rather than substituting demo predictions. Only trusted team artifacts may be loaded; joblib can execute Python code.

The deep model will need a separate adapter after the team supplies it. This prototype does not complete the required deep/modern model, full model comparison, or M3 report.

## Data and reproducibility

See [data/README.md](data/README.md). The instructor's shared 50,000-review sample is not available in this checkout. Obtain the assigned sample from the team/instructor; do not substitute a fresh download or treat the charter's reported sample statistics as independently verified here.

The charter plans duplicate removal before a stratified 80/20 split, shared across models, with feature fitting on training data only. Cross-validation/tuning belongs within the training partition; reserve the test set for final evaluation. Training commands and real-data results will be added by the responsible teammates when implemented. Demo pipelines are fitted in memory by `get_demo_models()`; they require no download.

`requirements.txt` pins the direct runtime dependencies. `requirements-lock.txt` captures the complete tested Python environment. Use the lock file for an identical dependency set:

```sh
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

## Course basis and remaining work

- `Week 6.2/lecture06.pdf`, PDF pp. 4–10: Bag-of-Words, fixed vocabulary, n-grams, TF-IDF.
- `Week 9/lecture09.pdf`, PDF pp. 12–16: Naive Bayes and text/sentiment classification.
- `group_project.pdf`, p. 2: at least two feature representations, two classical models, one deep/modern model, evaluation, and a Streamlit app with two trained models.
- `group_project.pdf`, pp. 3–5: M1 due Nov 8; M2 Nov 15; M3 Nov 22, each 11:59 pm. The app is advance preparation for M3, not an M1 deliverable requirement.
- `group_project.pdf`, pp. 6–7: final report/slides due Nov 29 at 11:59 pm; live demo date is **See Carmen**, not verified here.
- `group_project.pdf`, p. 8: repository organization, reproducibility, and continuing genuine contribution history.

Remaining team work: obtain/clean the assigned data, train/evaluate official models on the shared split, deliver compatible artifacts, add the deep model, complete the required report/screenshots, and deploy and verify the live Streamlit URL. No Carmen submission or cloud deployment has been performed.

## Deployment preparation

For Streamlit Community Cloud, use repository `NBNBTM/cse3521-amazon-food-sentiment`, branch `main`, entrypoint `app.py`, and Python 3.12. Review repository access and app viewer permissions separately before deploying. This repository setting does not establish a deployed app's viewer privacy. No Cloud account connection or live deployment is included in this initial setup.

References: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest), [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html). Course materials above are read-only local references and are not republished in the repository.

## Collaboration and AI assistance

Lindsey owns repository access/settings; Abdullah remains the team's Project Lead. Use short task branches and pull requests for review. Each teammate should commit their own actual contributions; do not manufacture commit history or attribute AI output to another member. Accepted collaborators can push changes; no collaborator invitations have been sent.

Codex assisted with initial app implementation, configuration, developer documentation, and tests on 2026-11-03. The project specification (p. 9) allows implementation assistance and requires a one-sentence disclosure in the final report. Team analysis, interpretation, and report writing remain the students' responsibility. Review and understand this implementation before using it in the course deliverable.
