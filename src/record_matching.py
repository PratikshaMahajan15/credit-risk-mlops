"""Record matching with PySpark: find duplicate customer records.

Credit bureaus receive the same person from many lenders, with typos, different name order,
missing fields and so on. This is a small version of that problem:

  1. make a fictional customer file where the true identity of every record is known
  2. clean and standardise the fields (lower case, no punctuation, sorted name words)
  3. "block": only compare records that share a blocking key, so we never compare every pair
  4. score each candidate pair (name similarity, date of birth, pincode, city)
  5. call it a match above a threshold, and measure precision, recall and the comparisons saved

All data is fictional. Run from the project root:
    python -m src.record_matching
"""
import json
import random
import sys

import pandas as pd

from src.config import INTERIM_DIR, REPORTS_DIR

FIRST = ["Aarav", "Vivaan", "Aditya", "Rohan", "Karan", "Neha", "Priya", "Anjali", "Sneha", "Pooja",
         "Rahul", "Amit", "Sanjay", "Meera", "Kavita", "Rakesh", "Deepa", "Nikhil", "Isha", "Manoj"]
LAST = ["Sharma", "Patil", "Deshmukh", "Joshi", "Kulkarni", "Mehta", "Iyer", "Nair", "Gupta", "Singh",
        "Reddy", "Chavan", "Jadhav", "Bhosale", "Pawar", "Shinde", "Verma", "Khan", "Das", "Rao"]
CITIES = ["Pune", "Mumbai", "Nagpur", "Nashik", "Thane", "Aurangabad"]
PIN_BASE = {"Pune": 411000, "Mumbai": 400000, "Nagpur": 440000, "Nashik": 422000,
            "Thane": 400600, "Aurangabad": 431000}


def _typo(word, rng):
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["drop", "swap", "double", "replace"])
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    return word[:i] + rng.choice("aeiou") + word[i + 1:]


def make_customers(n_entities=3000, dup_share=0.35, seed=7):
    """Fictional customers. About `dup_share` of people appear again as a messy second record."""
    rng = random.Random(seed)
    rows, rid = [], 0
    for eid in range(n_entities):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        city = rng.choice(CITIES)
        pin = PIN_BASE[city] + rng.randrange(0, 60)
        dob = f"{rng.randrange(1960, 2002)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
        rows.append({"record_id": rid, "entity_id": eid, "name": f"{first} {last}",
                     "dob": dob, "city": city, "pincode": str(pin)})
        rid += 1
        if rng.random() < dup_share:
            name = f"{first} {last}"
            style = rng.choice(["typo", "order", "initial", "case", "typo2"])
            if style == "typo":
                name = f"{_typo(first, rng)} {last}"
            elif style == "typo2":
                name = f"{first} {_typo(last, rng)}"
            elif style == "order":
                name = f"{last}, {first}"
            elif style == "initial":
                name = f"{first[0]}. {last}"
            else:
                name = f"  {first.upper()}  {last.lower()} "
            d_dob = dob if rng.random() > 0.15 else ""                      # sometimes missing
            d_pin = pin if rng.random() > 0.15 else pin + rng.choice([-1, 1])  # sometimes off by one
            d_city = city if rng.random() > 0.1 else ""
            rows.append({"record_id": rid, "entity_id": eid, "name": name, "dob": d_dob,
                         "city": d_city, "pincode": str(d_pin)})
            rid += 1
    # some people who merely share a surname/city (hard negatives)
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def get_spark():
    from src.spark_utils import get_spark as _get
    return _get("record-matching", memory="2g")


def standardise(sdf):
    """Clean the fields and build blocking keys."""
    from pyspark.sql import functions as F

    name = F.lower(F.regexp_replace(F.col("name"), r"[^A-Za-z ]", " "))
    name = F.trim(F.regexp_replace(name, r"\s+", " "))
    words = F.array_sort(F.split(name, " "))            # word order no longer matters
    sdf = sdf.withColumn("name_clean", F.array_join(words, " "))
    sdf = sdf.withColumn("last_word", F.element_at(words, -1))
    sdf = sdf.withColumn("first_initial", F.substring(F.element_at(words, 1), 1, 1))
    sdf = sdf.withColumn("dob", F.when(F.col("dob") == "", None).otherwise(F.col("dob")))
    sdf = sdf.withColumn("city", F.lower(F.trim(F.when(F.col("city") == "", None).otherwise(F.col("city")))))
    sdf = sdf.withColumn("pin3", F.substring(F.col("pincode"), 1, 3))
    # three blocking keys: a pair only needs to share one of them
    sdf = sdf.withColumn("block_a", F.concat_ws("|", F.soundex(F.col("last_word")), F.col("pin3")))
    sdf = sdf.withColumn("block_b", F.concat_ws("|", F.substring(F.col("last_word"), 1, 3), F.substring(F.col("dob"), 1, 4)))
    sdf = sdf.withColumn("block_c", F.col("dob"))          # same date of birth, whatever the spelling
    return sdf


def candidate_pairs(sdf):
    from pyspark.sql import functions as F

    a, b = sdf.alias("a"), sdf.alias("b")
    on_a = (F.col("a.block_a") == F.col("b.block_a")) & (F.col("a.record_id") < F.col("b.record_id"))
    on_b = (F.col("a.block_b") == F.col("b.block_b")) & F.col("a.dob").isNotNull() & (F.col("a.record_id") < F.col("b.record_id"))
    on_c = (F.col("a.block_c") == F.col("b.block_c")) & F.col("a.block_c").isNotNull() & (F.col("a.record_id") < F.col("b.record_id"))
    pairs = a.join(b, on_a | on_b | on_c)
    return pairs.select(
        F.col("a.record_id").alias("id_a"), F.col("b.record_id").alias("id_b"),
        F.col("a.entity_id").alias("entity_a"), F.col("b.entity_id").alias("entity_b"),
        F.col("a.name_clean").alias("name_a"), F.col("b.name_clean").alias("name_b"),
        F.col("a.first_initial").alias("init_a"), F.col("b.first_initial").alias("init_b"),
        F.col("a.last_word").alias("last_a"), F.col("b.last_word").alias("last_b"),
        F.col("a.dob").alias("dob_a"), F.col("b.dob").alias("dob_b"),
        F.col("a.city").alias("city_a"), F.col("b.city").alias("city_b"),
        F.col("a.pincode").alias("pin_a"), F.col("b.pincode").alias("pin_b"),
    ).dropDuplicates(["id_a", "id_b"])


def score_pairs(pairs, threshold=0.75):
    from pyspark.sql import functions as F

    longest = F.greatest(F.length("name_a"), F.length("name_b")).cast("double")
    lev_sim = 1 - F.levenshtein("name_a", "name_b") / F.when(longest > 0, longest).otherwise(1.0)
    # "A. Nair" vs "Aditya Nair": one side only has an initial, so treat same surname + same initial as a strong match
    has_initial = F.col("name_a").rlike(r"(^| )[a-z]( |$)") | F.col("name_b").rlike(r"(^| )[a-z]( |$)")
    name_sim = F.when(has_initial & (F.col("last_a") == F.col("last_b")), F.greatest(lev_sim, F.lit(0.9))).otherwise(lev_sim)
    dob_match = F.when(F.col("dob_a").isNull() | F.col("dob_b").isNull(), None) \
                 .otherwise((F.col("dob_a") == F.col("dob_b")).cast("double"))
    pin_diff = F.abs(F.col("pin_a").cast("int") - F.col("pin_b").cast("int"))
    pin_sim = F.when(pin_diff == 0, 1.0).when(pin_diff <= 1, 0.7).when(pin_diff <= 5, 0.3).otherwise(0.0)
    city_match = F.when(F.col("city_a").isNull() | F.col("city_b").isNull(), None) \
                  .otherwise((F.col("city_a") == F.col("city_b")).cast("double"))
    init_ok = (F.col("init_a") == F.col("init_b")).cast("double")

    scored = (pairs.withColumn("name_sim", name_sim).withColumn("dob_match", dob_match)
              .withColumn("pin_sim", pin_sim).withColumn("city_match", city_match)
              .withColumn("init_ok", init_ok))
    # weighted evidence; a missing field simply carries no weight
    parts = [("name_sim", 0.55), ("dob_match", 0.25), ("pin_sim", 0.12), ("city_match", 0.08)]
    num = sum(F.coalesce(F.col(c) * w, F.lit(0.0)) for c, w in parts)
    den = sum(F.when(F.col(c).isNotNull(), F.lit(w)).otherwise(F.lit(0.0)) for c, w in parts)
    scored = scored.withColumn("match_score", num / den)
    # two records with different, known dates of birth are different people, whatever the name says
    dob_conflict = F.col("dob_match") == 0.0
    return scored.withColumn(
        "is_match", (F.col("match_score") >= threshold) & (F.col("init_ok") == 1.0) & ~F.coalesce(dob_conflict, F.lit(False)))


def evaluate(spark, customers_sdf, scored):
    """Precision / recall against the known identities, and how many comparisons blocking saved."""
    from pyspark.sql import functions as F

    n = customers_sdf.count()
    all_pairs = n * (n - 1) // 2
    cand = scored.count()
    pred = scored.filter("is_match")
    tp = pred.filter(F.col("entity_a") == F.col("entity_b")).count()
    fp = pred.filter(F.col("entity_a") != F.col("entity_b")).count()
    # true duplicate pairs in the whole file
    per_entity = customers_sdf.groupBy("entity_id").count()
    true_pairs = per_entity.select(F.sum(F.col("count") * (F.col("count") - 1) / 2)).first()[0] or 0
    fn = int(true_pairs) - tp
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / true_pairs if true_pairs else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (tp and precision + recall) else 0.0
    cand_true = scored.filter(F.col("entity_a") == F.col("entity_b")).count()
    return {
        "records": int(n), "true_duplicate_pairs": int(true_pairs),
        "all_possible_pairs": int(all_pairs), "candidate_pairs_after_blocking": int(cand),
        "comparisons_saved_pct": round(100 * (1 - cand / all_pairs), 2),
        "blocking_recall": round(cand_true / true_pairs, 4) if true_pairs else None,
        "true_positives": int(tp), "false_positives": int(fp), "false_negatives": int(fn),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    }


def main(n_entities=3000, threshold=0.75, reports_dir=REPORTS_DIR):
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    customers = make_customers(n_entities)
    customers.to_csv(INTERIM_DIR / "customers_synthetic.csv", index=False)
    print(f"Fictional customer file: {len(customers):,} records for {customers['entity_id'].nunique():,} people")

    spark = get_spark()
    sdf = spark.createDataFrame(customers.astype(str))
    sdf = sdf.withColumn("record_id", sdf["record_id"].cast("long")).withColumn("entity_id", sdf["entity_id"].cast("long"))
    clean = standardise(sdf)
    scored = score_pairs(candidate_pairs(clean), threshold).cache()
    metrics = evaluate(spark, sdf, scored)
    metrics["threshold"] = threshold

    print("\nResults")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    with open(reports_dir / "record_matching_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nExamples of matched pairs:")
    scored.filter("is_match").select("name_a", "name_b", "dob_a", "dob_b", "pin_a", "pin_b", "match_score") \
          .orderBy("match_score").show(8, truncate=False)
    print("Examples of missed true duplicates:")
    scored.filter("NOT is_match AND entity_a = entity_b") \
          .select("name_a", "name_b", "dob_a", "dob_b", "pin_a", "pin_b", "match_score").show(5, truncate=False)
    spark.stop()
    return metrics


if __name__ == "__main__":
    main()
