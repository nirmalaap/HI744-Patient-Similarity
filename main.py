
"""HI 744 patient similarity retrieval assignment.

Run:
    python main.py PATH_TO_INPUT_DATA_FOLDER
"""

import argparse
from pathlib import Path

from src.data_loader import load_dataset
from src.preprocessing import preprocess_records
from src.tfidf_model import build_tfidf
from src.bm25_model import build_bm25_index
from src.similarity import retrieve_all
from src.evaluation import evaluate_all
from src.utils import save_json, ensure_dir


def main():

    # 1. Read command-line arguments
    parser = argparse.ArgumentParser(
        description="Patient similarity retrieval"
    )

    parser.add_argument(
        "input_folder",
        help="Folder containing one JSON file per patient"
    )

    parser.add_argument(
        "--output-folder",
        default="outputs",
        help="Output directory"
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=50000
    )

    args = parser.parse_args()

    # 2. Create output directory
    output_dir = Path(args.output_folder)
    ensure_dir(output_dir)

    # 3. Load patient records
    print("\nLoading patient records...")

    records = load_dataset(args.input_folder)

    print(f"Loaded {len(records):,} patient records")

    # 4. Preprocess clinical text
    print("\nPreprocessing patient records...")

    processed = preprocess_records(records)

    uids = [r["patient_uid"] for r in records]

    # 5. Build TF-IDF matrix
    print("\nBuilding TF-IDF matrix...")

    tfidf_matrix, vectorizer = build_tfidf(
        processed,
        max_features=args.max_features
    )

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Save TF-IDF matrix
    from scipy.sparse import save_npz

    save_npz(
        output_dir / "tfidf_matrix.npz",
        tfidf_matrix
    )

    # Save metadata to identify patients and terms
    save_json(
        output_dir / "tfidf_metadata.json",
        {
            "patient_uids": uids,
            "feature_names": vectorizer.get_feature_names_out().tolist(),
            "shape": list(tfidf_matrix.shape),
        }
    )

    print("TF-IDF matrix saved.")

    # 6. Build BM25 index
    print("\nBuilding BM25 index...")

    bm25_index = build_bm25_index(processed)

    # 7. Retrieve top 5 similar patients
    print("\nRetrieving top 5 similar patients...")

    retrieval = retrieve_all(
        tfidf_matrix=tfidf_matrix,
        bm25_index=bm25_index,
        records=records,
        processed_texts=processed,
    )

    save_json(
        output_dir / "patient_similarities.json",
        retrieval
    )

    print("Patient similarity results saved.")

    # 8. Evaluate retrieval performance
    print("\nEvaluating retrieval performance...")

    evaluation, summary = evaluate_all(
        records,
        retrieval
    )

    save_json(
        output_dir / "evaluation_results.json",
        evaluation
    )

    save_json(
        output_dir / "evaluation_summary.json",
        summary
    )

    # 9. Display average evaluation results
    print("\n" + "=" * 50)
    print("AVERAGE EVALUATION RESULTS")
    print("=" * 50)

    for method in ("tfidf", "bm25"):

        metrics = summary[method]

        print(f"\n{method.upper()}:")

        print(
            f"Precision@5: "
            f"{metrics['mean_precision_5']:.4f}"
        )

        print(
            f"Recall@5: "
            f"{metrics['mean_recall_5']:.4f}"
        )

        print(
            f"Patients included in recall average: "
            f"{metrics['patients_in_recall_average']:,}"
        )

        print(
            f"Patients excluded from recall average: "
            f"{metrics['patients_excluded_from_recall']:,}"
        )

    # 10. Compare TF-IDF and BM25
    comparison = summary["comparison"]

    print("\nMETHOD COMPARISON:")

    print(
        f"TF-IDF and BM25 Top-5 Overlap: "
        f"{comparison['mean_top5_overlap']:.4f}"
    )

    # 11. Final output location
    print("\n" + "=" * 50)

    print("ASSIGNMENT PIPELINE COMPLETED SUCCESSFULLY")

    print("=" * 50)

    print(f"\nOutputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
