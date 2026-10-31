from pathlib import Path
import csv

import numpy as np

from rs_core import build_codebook, make_params, nearest_codewords, polynomial_string

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

params = make_params(q=7, n=6, k=3)
messages, codebook = build_codebook(params)

codebook_path = OUT / "03_codebook_q7_n6_k3.csv"
with codebook_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["codeword_id", "a0", "a1", "a2", "polynomial", *[f"p({x})" for x in params.evaluation_points]])
    for index, (message, codeword) in enumerate(zip(messages, codebook)):
        writer.writerow([index, *map(int, message), polynomial_string(message.tolist()), *map(int, codeword)])

source_index = 3 * 7 * 7 + 2 * 7 + 4  # message [3,2,4] in lexicographic base-7 order
sent = codebook[source_index].copy()
received = sent.copy()
received[1] = (received[1] + 1) % 7
received[4] = (received[4] + 3) % 7
minimum, nearest = nearest_codewords(received, codebook)

report_path = OUT / "03_received_word_report.txt"
lines = [
    "Experiment 3: build the whole RS(7,6,3) codebook and corrupt a word",
    "=" * 73,
    f"Number of message polynomials: q^k = 7^3 = {len(messages)}",
    f"Minimum distance: d = n-k+1 = {params.distance}",
    f"Unique-decoding radius: {params.unique_radius}",
    "",
    f"Sent message coefficients: {messages[source_index].tolist()}",
    f"Sent codeword: {sent.tolist()}",
    f"Received word after two coordinate changes: {received.tolist()}",
    f"Distance to the code: {minimum}",
    f"Number of nearest codewords: {len(nearest)}",
]
for index in nearest[:20]:
    lines.append(
        f"  id={int(index)}, coefficients={messages[index].tolist()}, "
        f"polynomial={polynomial_string(messages[index].tolist())}, codeword={codebook[index].tolist()}"
    )
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(codebook_path)
print(report_path)
