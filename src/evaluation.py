"""Precision@5 and Recall@5 evaluation against supplied similar_patients."""
def evaluate_all(records, retrieval, k=5):
    results = {}
    methods = ("tfidf", "bm25")
    precision_values = {method: [] for method in methods}
    recall_values = {method: [] for method in methods}
    excluded_recall = 0
    overlap_values = []

    for record in records:
        uid = record["patient_uid"]
        relevant = set(record.get("_similar_patients", {}).keys())
        results_for_patient = {
            "patient_id": str(record["patient_id"])
        }

        for method in methods:
            ranked = retrieval[uid][method][:k]
            relevant_retrieved = len(set(ranked) & relevant)
            precision = relevant_retrieved / float(k)
            results_for_patient[method] = {"precision_5": precision}

            precision_values[method].append(precision)
            if relevant:
                recall = relevant_retrieved / float(len(relevant))
                results_for_patient[method]["recall_5"] = recall
                recall_values[method].append(recall)
            else:
                results_for_patient[method]["recall_5"] = None

        overlap = len(set(retrieval[uid]["tfidf"]) & set(retrieval[uid]["bm25"])) / float(k)
        results_for_patient["top5_overlap"] = overlap
        overlap_values.append(overlap)
        results[uid] = results_for_patient

        if not relevant:
            excluded_recall += 1

    summary = {}
    for method in methods:
        summary[method] = {
            "mean_precision_5": (
                sum(precision_values[method]) / len(precision_values[method])
                if precision_values[method] else 0.0
            ),
            "mean_recall_5": (
                sum(recall_values[method]) / len(recall_values[method])
                if recall_values[method] else 0.0
            ),
            "patients_in_precision_average": len(precision_values[method]),
            "patients_in_recall_average": len(recall_values[method]),
            "patients_excluded_from_recall": excluded_recall,
        }

    summary["comparison"] = {
        "mean_top5_overlap": (
            sum(overlap_values) / len(overlap_values) if overlap_values else 0.0
        ),
        "overlap_definition": "intersection of TF-IDF and BM25 top-5 divided by 5",
    }
    return results, summary
