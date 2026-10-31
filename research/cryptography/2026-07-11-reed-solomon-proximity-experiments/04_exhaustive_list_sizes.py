from pathlib import Path
import csv

import matplotlib.pyplot as plt

from rs_core import all_words, build_codebook, list_size_summary, make_params

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

cases = [(5, 4, 2), (7, 5, 2), (7, 6, 3)]
summary_rows = []
witness_rows = []
histogram_rows = []

for q, n, k in cases:
    params = make_params(q, n, k)
    _, codebook = build_codebook(params)
    words = all_words(q, n)
    radii = range(n + 1)
    maxima, witnesses, histograms = list_size_summary(words, codebook, radii)

    for t in radii:
        summary_rows.append({
            "q": q,
            "n": n,
            "k": k,
            "rate": params.rate,
            "minimum_distance": params.distance,
            "unique_radius": params.unique_radius,
            "johnson_radius_real": params.johnson_radius_real,
            "johnson_radius_integer": params.johnson_radius_integer,
            "capacity_radius_integer": params.capacity_radius_integer,
            "radius_t": t,
            "relative_radius": t / n,
            "worst_list_size_L_t": maxima[t],
        })
        witness_rows.append({
            "q": q,
            "n": n,
            "k": k,
            "radius_t": t,
            "worst_list_size_L_t": maxima[t],
            "witness_received_word": " ".join(map(str, witnesses[t].tolist())),
        })
        for list_size, number_of_words in sorted(histograms[t].items()):
            histogram_rows.append({
                "q": q,
                "n": n,
                "k": k,
                "radius_t": t,
                "list_size": list_size,
                "number_of_received_words": number_of_words,
            })

for filename, rows in [
    ("04_list_size_summary.csv", summary_rows),
    ("04_worst_received_words.csv", witness_rows),
    ("04_list_size_histograms.csv", histogram_rows),
]:
    path = OUT / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

plt.figure(figsize=(9, 5.5))
for q, n, k in cases:
    selected = [row for row in summary_rows if (row["q"], row["n"], row["k"]) == (q, n, k)]
    plt.plot(
        [row["radius_t"] for row in selected],
        [row["worst_list_size_L_t"] for row in selected],
        marker="o",
        label=f"q={q}, n={n}, k={k}",
    )
plt.yscale("log")
plt.xlabel("Hamming radius t")
plt.ylabel("Worst-case list size L_t (log scale)")
plt.title("Exact worst-case Reed–Solomon list sizes")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "04_worst_list_size_by_radius.png", dpi=180)
plt.close()

print(OUT / "04_list_size_summary.csv")
print(OUT / "04_worst_list_size_by_radius.png")
