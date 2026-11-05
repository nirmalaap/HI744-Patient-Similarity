# HI 744 Patient Similarity Retrieval

## Methods
- TF-IDF with smoothed IDF, sublinear term frequency, and L2 normalization.
- BM25 with `k1=1.5` and `b=0.75`.
- Precision@5 and Recall@5 against the dataset's `similar_patients` field.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

NLTK may download the English stopword and WordNet data the first time it runs.

## Run
```bash
python main.py PATH_TO_INPUT_DATA_FOLDER
```

For the supplied archive, unzip `PMC_Patients.zip` and point the command to the
folder containing the JSON files, for example:
```bash
python main.py data/PMC_Patients
```

## Outputs
- `outputs/tfidf_matrix.npz`: sparse TF-IDF matrix.
- `outputs/tfidf_metadata.json`: row-to-patient UID and column-to-term mapping.
- `outputs/patient_similarities.json`: top five patients from TF-IDF and BM25.
- `outputs/evaluation_results.json`: per-patient precision@5 and recall@5.
- `outputs/evaluation_summary.json`: average metrics and top-five overlap.

## Evaluation note
The supplied `similar_patients` field may refer to UIDs not present in the
provided folder. Such references remain in the ground-truth denominator for
Recall@5, as specified in the assignment. Patients with no listed relevant
patients are excluded from the recall average.
