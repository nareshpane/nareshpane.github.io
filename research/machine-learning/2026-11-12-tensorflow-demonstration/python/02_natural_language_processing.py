"""Classify entirely synthetic university requests; hold out template families.

Run: python 02_natural_language_processing.py --quick --output ../outputs/nlp
Independent full program. NumPy baseline/data functions do not import TensorFlow.
No personal data. These templates are a toy language environment, not a benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import platform
import string
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

CLASSES = ["IT", "Library", "Housing", "Transportation"]
FAMILIES = {
    "train": [0, 1, 2, 3, 4, 5], "validation": [6, 7], "test": [8, 9]
}
TEMPLATES = [
    "{prefix} Can someone help because {intent}? {suffix}",
    "I tried the usual advice, but {intent}. {prefix} What should I do next? {suffix}",
    "{prefix} My concern is this: {intent}. Could your office look into it? {suffix}",
    "Please route this request to the right team. {intent}. {suffix} {prefix}",
    "{prefix} I was told to ask here about this problem: {intent}. {suffix}",
    "When I contacted the service desk I explained that {intent}. {suffix} {prefix}",
    "Following up on an unresolved issue — {intent}. Guidance about the next step would help. {suffix} {prefix}",
    "Does this office handle the following situation? {intent}. I need a practical way forward. {prefix} {suffix}",
    "Subject: assistance requested. The situation I am facing is that {intent}. Who takes responsibility for resolving this? {suffix} {prefix}",
    "A colleague suggested escalating this matter after several unsuccessful attempts. {intent}. Please explain the available options. {prefix} {suffix}",
]
INTENTS = [
    ["my laptop cannot join the campus wifi inside the library", "the login page rejects my account password",
     "the printer driver stopped working in my residence", "I cannot open the course portal from my phone",
     "my university email is not receiving messages", "I need to reset multifactor authentication",
     "the network connection drops near the bus stop", "a software installation needs administrator access",
     "my browser reports an error when I sign in", "the wireless signal works but the internet does not"],
    ["I need to locate a journal article for my course", "the book renewal option is unavailable",
     "I want to borrow a research book from another university", "the catalogue lists a missing reference text",
     "I need help finding historical newspaper archives", "I cannot reserve the reading room for a study group",
     "I need advice about citing a database source", "my library loan has an unexpected overdue charge",
     "I want to access scholarly resources from my residence", "I need a librarian to help with a literature search"],
    ["the heater in my residence room has stopped working", "a water leak is damaging my dormitory wall",
     "I need to discuss a room transfer before next term", "the shared kitchen has a broken appliance",
     "the lock on my student apartment door is damaged", "the residence laundry machine keeps my payment",
     "I want clarification of my housing deposit refund", "the study area in my residence has no working lights",
     "I need maintenance for a noisy ventilation fan", "my student accommodation contract has the wrong dates"],
    ["the campus shuttle did not arrive at the library stop", "I need the weekend bus timetable",
     "the transit pass on my student card is not accepted", "I want the accessible transport booking procedure",
     "the bike parking area is closed without directions", "I need a route between the residence and the laboratory",
     "the evening shuttle schedule conflicts with my course", "I need to replace a lost campus bus pass",
     "a detour has removed the usual bus stop", "I want information about late night campus transport"],
]
# Evaluation changes intent wording as well as holding out whole wrapper families.
# Shared words such as access, account, room, card, library, and residence occur in
# several classes; the requested action, rather than the named building, is target.
EVALUATION_INTENTS = [
    ["the wireless service on my computer fails while I study in a reading room", "I have been locked out of my university account",
     "a residence computer cannot communicate with the shared printer", "I see an access error instead of my online course",
     "my campus mailbox has stopped delivering email", "the sign in verification device needs replacement",
     "the internet fails beside the transport waiting area", "I cannot get permission to install this application",
     "my web login produces a blank screen", "joining the network does not let me reach any websites"],
    ["a publication needed for my research cannot be found in the catalogue", "I want to extend the due date on borrowed books",
     "I am requesting an interlibrary loan for a course text", "a reference volume should be on the shelf but is absent",
     "I need archival press sources for a history assignment", "our group needs to book a library discussion room",
     "I need guidance on bibliography entries for electronic resources", "a charge was added to my borrowing account",
     "I need scholarly reading material while away from campus", "I want expert help searching academic literature"],
    ["the temperature in my dormitory is too low because the radiator is broken", "water is coming through my accommodation ceiling",
     "I would like a different student residence room next semester", "the communal cooker in student housing is defective",
     "the door to my accommodation cannot be secured", "the shared washing machine charged me but did not start",
     "I want to recover the security deposit on my room", "the residence common study room needs electrical maintenance",
     "the ventilation in my student apartment rattles all night", "the dates on my accommodation agreement need correcting"],
    ["the university bus bypassed the stop outside the library", "I need departure times for campus transport on Sunday",
     "my student transit card is rejected when I board", "I need assistance arranging an accessible campus journey",
     "there is no information about the closed bicycle storage area", "how can I travel from student housing to the science building",
     "I need a later bus after my class finishes", "I cannot find my pass for the university shuttle",
     "road works have displaced my usual transit pickup location", "I need a safe university transport option after dark"],
]
PREFIXES = ["Hello.", "Hi there.", "Thanks for reading.", "This is rather urgent.", "I may have contacted the wrong office.", "Sorry if this has already been answered."]
SUFFIXES = ["I am on campus today.", "This has happened twice.", "An online answer was not clear.", "Please reply when possible.", "A friend has the same difficulty.", "I will need assistance before Friday."]

CHALLENGES = [
    ("Wifi is broken in the library; I do not need help finding a book.", 0),
    ("I can log in, but need a librarian to locate an article about computer networks.", 1),
    ("My residence network works. The heater does not work at all.", 2),
    ("The bus pass in my account is valid, but no shuttle stops near my residence.", 3),
    ("The library is closed and I cannot reach the transport desk about my missing bus.", 3),
    ("My room is cold and I also cannot access email; please repair the heater first.", 2),
    ("Nothing is working and I need help.", None),
    ("I cannot connect to wifii; the campus netwrok keeps failing.", 0),
]


# SECTION 1 — Family-separated, deduplicated synthetic data.
def tokenize(text: str) -> list[str]:
    """Match Keras lower_and_strip_punctuation for these ASCII examples."""
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()


def generate_messages(name: str, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    texts, labels, family_ids, seen = [], [], [], set()
    intent_sets = INTENTS if name == "train" else EVALUATION_INTENTS
    for label, intents in enumerate(intent_sets):
        attempts = 0
        while sum(y == label for y in labels) < per_class:
            attempts += 1
            if attempts > per_class * 100:
                raise ValueError("Requested more unique messages than the finite generator can supply")
            family = int(rng.choice(FAMILIES[name]))
            text = TEMPLATES[family].format(intent=rng.choice(intents), prefix=rng.choice(PREFIXES), suffix=rng.choice(SUFFIXES))
            # Modest typographical variation, using an independent deterministic draw.
            if rng.random() < .08:
                text = text.replace("campus", "campuss").replace("library", "libary")
            normalized = " ".join(tokenize(text))
            if normalized in seen:
                continue
            seen.add(normalized)
            texts.append(text)
            labels.append(label)
            family_ids.append(family)
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], np.asarray(labels, dtype=np.int64)[order], np.asarray(family_ids)[order]


def make_corpus(seed: int = 42, quick: bool = False):
    counts = (40, 12, 12) if quick else (240, 60, 60)
    seeds = np.random.SeedSequence(seed).generate_state(3)
    corpus = {name: generate_messages(name, count, int(s)) for name, count, s in zip(FAMILIES, counts, seeds)}
    sets = [set(" ".join(tokenize(t)) for t in corpus[k][0]) for k in FAMILIES]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert not (set(FAMILIES["train"]) & set(FAMILIES["test"]))
    return corpus


# SECTION 2 — A conventional count-based multinomial Naive Bayes baseline.
def fit_baseline(texts, labels, max_tokens=3000):
    counts = Counter(w for text in texts for w in tokenize(text))
    words = sorted(counts, key=lambda w: (-counts[w], w))[:max_tokens - 2]
    vocabulary = ["", "[UNK]"] + words
    index = {word: i for i, word in enumerate(vocabulary)}
    word_counts = np.ones((4, len(vocabulary)), dtype=np.float64) # Laplace alpha=1.
    class_counts = np.bincount(labels, minlength=4).astype(float)
    for text, label in zip(texts, labels):
        for word in tokenize(text):
            word_counts[label, index.get(word, 1)] += 1
    # Padding never appears in a document; remove its artificial smoothing mass.
    word_counts[:, 0] = 0
    likelihood = np.log(word_counts[:, 1:] / word_counts[:, 1:].sum(axis=1, keepdims=True))
    return {"vocabulary": vocabulary, "log_likelihood": likelihood,
            "log_prior": np.log(class_counts / class_counts.sum())}


def baseline_predict(model, texts):
    index = {word: i for i, word in enumerate(model["vocabulary"])}
    logits = np.tile(model["log_prior"], (len(texts), 1))
    for i, text in enumerate(texts):
        for word in tokenize(text):
            logits[i] += model["log_likelihood"][:, index.get(word, 1) - 1]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def classification_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((4, 4), dtype=int)
    np.add.at(matrix, (labels, predicted), 1)
    per_class = []
    for k in range(4):
        tp = int(matrix[k, k]); fp = int(matrix[:, k].sum() - tp); fn = int(matrix[k].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        per_class.append({"label": CLASSES[k], "precision": precision, "recall": recall, "f1": f1,
                          "support": int(matrix[k].sum())})
    # ECE is descriptive and bin-dependent; it is not a calibration guarantee.
    confidence = probabilities.max(axis=1)
    ece = 0.
    for low in np.linspace(0, .9, 10):
        selected = (confidence >= low) & (confidence < low + .1 if low < .9 else confidence <= 1)
        if selected.any():
            ece += selected.mean() * abs((predicted[selected] == labels[selected]).mean() - confidence[selected].mean())
    return {"accuracy": float((predicted == labels).mean()),
            "macro_f1": float(np.mean([m["f1"] for m in per_class])),
            "cross_entropy": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
            "ece_10_equal_width_bins": float(ece), "confusion_matrix": matrix.tolist(), "per_class": per_class}


def write_confusion(path, matrix, title):
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 450" role="img"><title>{html.escape(title)}</title><rect width="600" height="450" fill="#fff8f2"/><g font-family="sans-serif" fill="#26332f"><text x="30" y="30" font-size="19">{html.escape(title)}</text><text x="225" y="60">Predicted category</text><text x="10" y="100">True</text>']
    for j, label in enumerate(CLASSES):
        svg.append(f'<text x="{190+j*95}" y="90" text-anchor="middle" font-size="12">{label}</text>')
    for i, row in enumerate(matrix):
        svg.append(f'<text x="135" y="{145+i*75}" text-anchor="end" font-size="12">{CLASSES[i]}</text>')
        for j, value in enumerate(row):
            svg.append(f'<rect x="{145+j*95}" y="{105+i*75}" width="90" height="70" fill="{"#d2e9cc" if i==j else "#f4e5df"}"/><text x="{190+j*95}" y="{145+i*75}" text-anchor="middle">{value}</text>')
    path.write_text("".join(svg) + "</g></svg>", encoding="utf-8")


# SECTION 3 — Train-only vocabulary, masked embeddings, and pooled classifier.
def build_model(tf, training_texts):
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=3000, output_sequence_length=48,
                                                  standardize="lower_and_strip_punctuation", split="whitespace")
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(tf.constant(training_texts)).batch(32))
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoded = vectorizer(inputs)
    x = tf.keras.layers.Embedding(len(vectorizer.get_vocabulary()), 24, mask_zero=True)(encoded)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)  # Respects padding mask.
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(.15)(x)
    logits = tf.keras.layers.Dense(4)(x)
    return tf.keras.Model(inputs, logits, name="university_request_classifier"), vectorizer


def run(args):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    corpus = make_corpus(args.seed, args.quick)
    train_text, train_y, _ = corpus["train"]
    val_text, val_y, _ = corpus["validation"]
    test_text, test_y, _ = corpus["test"]
    baseline = fit_baseline(train_text, train_y)
    baseline_prob = baseline_predict(baseline, test_text)
    model, vectorizer = build_model(tf, train_text)
    assert set(vectorizer.get_vocabulary()[2:]) <= set(w for text in train_text for w in tokenize(text))

    def dataset(texts, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((tf.constant(texts), labels))
        if training:
            ds = ds.shuffle(len(texts), seed=args.seed, reshuffle_each_iteration=True)
        return ds.batch(32).prefetch(tf.data.AUTOTUNE)

    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"], jit_compile=False)
    start = time.perf_counter()
    history = model.fit(dataset(train_text, train_y, True), validation_data=dataset(val_text, val_y),
                        epochs=2 if args.quick else 35, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                                   tf.keras.callbacks.TerminateOnNaN()])
    logits = model.predict(tf.constant(test_text), verbose=0)
    probabilities = tf.nn.softmax(logits).numpy()
    assert probabilities.shape == (len(test_text), 4) and np.isfinite(probabilities).all()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "requests.keras")
    restored = tf.keras.models.load_model(args.output / "requests.keras", compile=False)
    np.testing.assert_allclose(restored.predict(tf.constant(test_text[:4]), verbose=0), logits[:4], atol=1e-6)
    predicted = probabilities.argmax(axis=1)
    errors = sorted([i for i in range(len(test_y)) if predicted[i] != test_y[i]],
                    key=lambda i: -probabilities[i].max())
    challenge_text = [text for text, _ in CHALLENGES]
    challenge_prob = tf.nn.softmax(model.predict(tf.constant(challenge_text), verbose=0)).numpy()
    neural_metrics = classification_metrics(test_y, probabilities)
    result = {"provenance": "actual TensorFlow training on synthetic messages", "seed": args.seed,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "tensorflow": tf.__version__},
              "configuration": {"template_families": FAMILIES, "counts": {k: len(v[0]) for k,v in corpus.items()},
                                "sequence_length": 48, "embedding_dimension": 24, "vocabulary_size": len(vectorizer.get_vocabulary())},
              "neural_metrics": neural_metrics, "baseline_metrics": classification_metrics(test_y, baseline_prob),
              "history": history.history, "seconds": time.perf_counter() - start, "reload_verified": True,
              "misclassified_examples": [{"text": test_text[i], "true_label": CLASSES[test_y[i]],
                                           "predicted_label": CLASSES[predicted[i]], "probabilities": probabilities[i].tolist()} for i in errors[:20]],
              "challenge_set": [{"text": text, "routing_policy_label": None if y is None else CLASSES[y],
                                 "probabilities": p.tolist()} for (text,y),p in zip(CHALLENGES,challenge_prob)]}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "vocabulary.json").write_text(json.dumps(vectorizer.get_vocabulary(), indent=2), encoding="utf-8")
    write_confusion(args.output / "neural_confusion.svg", neural_metrics["confusion_matrix"], "Held-out template families: neural classifier")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="160/48/48 requests, at most two epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/nlp")
    args = parser.parse_args()
    try:
        run(args)
    except ModuleNotFoundError as exc:
        raise SystemExit("Install requirements.txt in an isolated environment first.") from exc
