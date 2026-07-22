# %% [markdown]
# # An NLP Analysis of Consumer Financial Complaints
# ### Digital-payment complaints: themes, sentiment, priority, and products
#
# **Methods:** text mining, topic modelling, domain-specific sentiment analysis (FinBERT) benchmarked
# against a generic baseline, temporal analysis, and a product-type breakdown.
# **Stack:** Python, pandas, scikit-learn, NLTK, Hugging Face Transformers, matplotlib.
#
# ---
#
# ### Overview
#
# This notebook works through a digital-payment complaints dataset in a structured way:
#
# - **Building the vocabulary** from the complaint text (Section 3).
# - **Grouping complaints into themes** with topic modelling (Section 4).
# - **Measuring sentiment with a finance-domain model** rather than a generic one, and showing why it
#   matters (Section 5).
# - **Tracking how the complaint mix changed over time** (Section 6).
# - **Ranking the themes with a Complaint Priority Index** (Section 7).
# - **Breaking complaints down by product type** (Section 8).
#
# ### The data
#
# Public complaints from the **US CFPB Consumer Complaint Database**, product category *"Money
# transfer, virtual currency, or money service"*, where the customer wrote a narrative. 4,282
# complaints, 2017 to 2024. Fully public data (a US federal agency publishes it); no confidential or
# employer data is involved, and company names are deliberately excluded from the analysis so the
# focus stays on problems, not providers.

# %% [markdown]
# ## 1. Setup

# %%
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.figsize": (9, 5), "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "font.size": 11,
})
INK = "#0f172a"
PRIMARY = "#2563eb"
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#64748b"]

# %% [markdown]
# ## 2. The Dataset
#
# Before modelling, look at the raw shape: size, time span, and one real record so we know what the
# text is like. CFPB masks personal details as runs of X.

# %%
df = pd.read_csv("data/complaints_money_transfer.csv")
df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
df["year"] = df["Date received"].dt.year
# Outcome label used later: did the complaint win relief, or just an explanation?
df["relief"] = df["Company response to consumer"].isin(
    ["Closed with monetary relief", "Closed with non-monetary relief"]).astype(int)
print(f"Rows: {len(df):,}")
print(f"Span: {df['Date received'].min().date()} to {df['Date received'].max().date()}")
print(f"Unique issue labels (CFPB-provided): {df['Issue'].nunique()}")
print("\nExample narrative:\n", df["Consumer complaint narrative"].iloc[0][:500], "...")

# %%
# Volume by year, and the CFPB-provided issue labels (a useful sanity check on what we later model)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
by_year = df.groupby("year").size()
a1.bar(by_year.index, by_year.values, color=PRIMARY)
a1.set_title("Complaint volume by year"); a1.set_xlabel(""); a1.set_ylabel("Complaints")
top_issue = df["Issue"].value_counts().head(8).sort_values()
a2.barh(range(len(top_issue)), top_issue.values, color=PRIMARY)
a2.set_yticks(range(len(top_issue))); a2.set_yticklabels([i[:38] for i in top_issue.index], fontsize=9)
a2.set_title("Most common issues (CFPB labels)"); a2.set_xlabel("Complaints")
plt.tight_layout(); plt.savefig("figures/01_overview.png", dpi=120, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 3. Building the Vocabulary
#
# Topic models do not read words, they read a numeric matrix. Building that matrix well is most of
# the work, and every choice below is deliberate:
#
# - **Cleaning:** lowercase, strip the `XXXX` redactions (otherwise they become the most frequent
#   "word"), and keep letters only.
# - **Stop-words, in three layers:** (1) standard English stop-words, (2) generic filler that is
#   frequent but meaningless here (`account`, `money`, `would`), and (3) **company and brand names**,
#   removed on purpose so the model groups complaints by *problem* rather than rediscovering which
#   provider was involved, which we already know from a column.
# - **N-grams (1 to 2):** so multi-word concepts survive ("wire transfer", "fell victim").
# - **`min_df=5`** drops words in fewer than 5 documents (typos, noise); **`max_df=0.6`** drops words
#   in more than 60% of documents (too generic to separate topics).
# - **TF-IDF weighting:** up-weights words distinctive to a complaint, down-weights the ubiquitous.

# %%
brand_stop = {
    "paypal", "coinbase", "venmo", "cashapp", "cash app", "chase", "wells", "fargo", "wells fargo",
    "bank america", "america", "zelle", "citibank", "chase bank", "cash", "app", "bank",
    "moneygram", "western union", "western", "union", "wu", "binance", "gemini", "robinhood",
    "chime", "remitly", "square", "skrill", "revolut", "wise", "kraken", "stripe", "affirm",
    "klarna", "varo", "sofi", "dave", "netspend",
}
custom_stop = set(stopwords.words("english")) | brand_stop | {
    "xxxx", "xx", "xxxxxxxx", "account", "money", "company", "would", "said", "told",
    "also", "get", "got", "us", "im", "ive", "dont", "didnt", "day", "days",
}

def clean(text):
    text = str(text).lower()
    text = re.sub(r"x{2,}", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df["clean"] = df["Consumer complaint narrative"].apply(clean)
vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.6,
                             stop_words=list(custom_stop), ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean"])
vocab = vectorizer.get_feature_names_out()
print(f"Matrix: {X.shape[0]:,} complaints x {X.shape[1]:,} vocabulary terms")
print("Sample of the vocabulary:", ", ".join(list(vocab[::150])))

# %% [markdown]
# ## 4. Selecting the Number of Topics
#
# The complaints are grouped into themes with NMF (non-negative matrix factorisation). Seven topics
# gave the clearest, most usable separation: detailed enough to act on, without splitting into
# overlapping slivers. Each topic is then labelled by reading its most distinctive words.

# %%
K = 7
nmf = NMF(n_components=K, random_state=42, init="nndsvd", max_iter=400)
W = nmf.fit_transform(X)
H = nmf.components_
df["topic"] = W.argmax(axis=1)

def top_words(t, n=9):
    return [vocab[i] for i in H[t].argsort()[::-1][:n]]

LABELS = {
    0: "Fraud & unauthorized transactions",
    1: "Uncategorised (templated text)",
    2: "Cheque deposits & cashing",
    3: "Wire transfers",
    4: "Investment scams",
    5: "Accessing / withdrawing funds & support",
    6: "Account limitations & appeals",
}
for t in range(K):
    print(f"[{LABELS[t]}]  n={(df['topic']==t).sum()}")
    print("   top words:", ", ".join(top_words(t)))

# %% [markdown]
# Topic 1 is a small pocket of near-identical, templated narratives rather than a genuine theme. It is
# labelled as uncategorised and left out of the theme breakdowns.

# %% [markdown]
# ## 5. Domain-Specific Sentiment Analysis
#
# ### 5a. A Generic Baseline Model
#
# VADER is the usual first reach: a rule-based lexicon that scores text from -1 to +1. It was tuned
# on social media, and on formal complaint language it makes a specific, revealing mistake.

# %%
sia = SentimentIntensityAnalyzer()
df["vader"] = df["Consumer complaint narrative"].apply(lambda t: sia.polarity_scores(str(t))["compound"])

# The tell: VADER calls a large share of complaints POSITIVE
vader_pos_share = (df["vader"] >= 0.05).mean()
print(f"VADER labels {vader_pos_share*100:.0f}% of these complaints 'positive'.")
# Look at what it thinks are the most positive complaints
print("\nA complaint VADER scored strongly POSITIVE:")
print(" ", df.sort_values("vader", ascending=False)["Consumer complaint narrative"].iloc[0][:280].replace("\n", " "), "...")

# %% [markdown]
# Read that "positive" complaint: it opens *"Dear Support Team, I hope this finds you well... I have
# always valued..."* The customer is angry but polite, and VADER scores the politeness. On this
# corpus, polite framing systematically fools it, and the earlier "account limitations & appeals"
# topic (full of formal appeal letters) is exactly where this concentrates. A generic tool is
# measuring etiquette, not sentiment.
#
# ### 5b. A Finance-Domain Model (FinBERT)
#
# **FinBERT** is a BERT model fine-tuned on financial text. It reads whole sentences in context, so
# "I hope you are well but my funds are frozen" is understood as negative. Scores for all 4,282
# complaints were precomputed once (about 5 minutes on CPU) and cached; the loop below regenerates
# them if the cache is missing.

# %%
import os
CACHE = "data/finbert_scores.csv"
if os.path.exists(CACHE):
    fb = pd.read_csv(CACHE)
    print("Loaded cached FinBERT scores.")
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
    texts = df["Consumer complaint narrative"].astype(str).tolist()
    pos, neg, neu = [], [], []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            p = torch.softmax(mdl(**enc).logits, dim=1).numpy()
        pos += list(p[:, 0]); neg += list(p[:, 1]); neu += list(p[:, 2])
    fb = pd.DataFrame({"finbert_pos": pos, "finbert_neg": neg, "finbert_neu": neu})
    fb["finbert_score"] = fb["finbert_pos"] - fb["finbert_neg"]
    fb.to_csv(CACHE, index=False)

df["finbert"] = fb["finbert_score"].values

# %%
# Head to head: distributions, and how often they disagree
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
a1.hist(df["vader"], bins=40, alpha=0.7, label="VADER (generic)", color="#94a3b8")
a1.hist(df["finbert"], bins=40, alpha=0.7, label="FinBERT (finance)", color=PRIMARY)
a1.set_title("FinBERT sees complaints as far more negative"); a1.set_xlabel("negative  <->  positive"); a1.legend()
# disagreement: VADER positive but FinBERT negative
flip = ((df["vader"] >= 0.05) & (df["finbert"] <= -0.1)).sum()
a2.bar(["VADER\ncalls positive", "FinBERT\ncalls positive"], [vader_pos_share*100, (df["finbert"] >= 0.05).mean()*100], color=["#94a3b8", PRIMARY])
a2.set_title(f"{flip} complaints VADER calls positive,\nFinBERT correctly calls negative")
a2.yaxis.set_major_formatter(PercentFormatter()); a2.set_ylabel("Share of complaints")
plt.tight_layout(); plt.savefig("figures/03_sentiment_models.png", dpi=120, bbox_inches="tight"); plt.show()
print(f"On {flip} complaints ({flip/len(df)*100:.0f}%) the two models flatly disagree, and FinBERT is right on inspection.")

# %% [markdown]
# ## 6. Trends Over Time
#
# A single snapshot hides movement. Two temporal views, using FinBERT sentiment and the theme
# assignments: how the theme mix shifted, and whether complaints got angrier.

# %%
themes_to_track = [0, 5, 3, 4]   # fraud, funds-access, wire, investment scams
yr = df[df["year"].between(2018, 2023)]   # trim sparse end-years
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
share = (yr.groupby("year")["topic"].value_counts(normalize=True).unstack().fillna(0) * 100)
for i, t in enumerate(themes_to_track):
    a1.plot(share.index, share[t], "o-", lw=2, color=PALETTE[i], label=LABELS[t][:26])
a1.set_title("Theme mix over time (% of that year's complaints)")
a1.yaxis.set_major_formatter(PercentFormatter()); a1.legend(fontsize=8); a1.set_xlabel("")
sent_yr = yr.groupby("year")["finbert"].mean()
a2.plot(sent_yr.index, sent_yr.values, "o-", lw=2, color="#dc2626")
a2.set_title("Average complaint sentiment by year (FinBERT)"); a2.set_xlabel(""); a2.set_ylabel("Avg sentiment")
plt.tight_layout(); plt.savefig("figures/04_over_time.png", dpi=120, bbox_inches="tight"); plt.show()
print("Investment-scam share by year (%):")
print((yr.groupby("year")["topic"].apply(lambda s: (s == 4).mean()*100)).round(1))

# %% [markdown]
# ## 7. The Complaint Priority Index
#
# Sentiment and topics describe the data. A business still has to decide *what to fix first*, and
# volume alone is a poor guide (a huge theme people barely mind is lower priority than a smaller one
# that leaves customers furious and unresolved). So I define a transparent composite index:
#
# > **CPI = 100 x (0.4 x volume + 0.4 x negativity + 0.2 x unresolved-rate)**, each component
# > min-max normalised across themes.
#
# The reasoning: a theme deserves attention in proportion to how many customers it hits, how negative
# their experience is (domain sentiment), and how rarely their complaint ends in any relief. The
# weights are a stated, adjustable judgement, not a hidden default. This is the analyst's
# contribution on top of the models: turning measurements into a ranked decision.

# %%
def minmax(s):
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

cpi = (df[df["topic"] != 1].groupby("topic")
       .agg(volume=("topic", "size"), sentiment=("finbert", "mean"), relief_rate=("relief", "mean"))
       .reset_index())
cpi["CPI"] = (0.4*minmax(cpi["volume"]) + 0.4*minmax(-cpi["sentiment"]) + 0.2*minmax(1-cpi["relief_rate"])) * 100
cpi["theme"] = cpi["topic"].map(LABELS)
cpi = cpi.sort_values("CPI", ascending=False)

THEME_COL = {0: "#dc2626", 2: "#059669", 3: "#d97706", 4: "#7c3aed", 5: "#2563eb", 6: "#64748b"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.barh(cpi["theme"][::-1], cpi["CPI"][::-1], color=[THEME_COL[int(t)] for t in cpi["topic"][::-1]])
a1.set_title("Themes ranked by CPI"); a1.set_xlabel("Complaint Priority Index (0-100)")
a1.tick_params(labelsize=9)
a2.scatter(cpi["sentiment"], cpi["volume"], s=cpi["CPI"]*6,
           c=[THEME_COL[int(t)] for t in cpi["topic"]], edgecolor=INK, alpha=0.85)
for _, r in cpi.iterrows():
    a2.annotate(LABELS[int(r["topic"])][:20], (r["sentiment"], r["volume"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8)
a2.set_title("Priority map (bubble size = CPI, colour = theme)")
a2.set_xlabel("Sentiment (angrier <-- )"); a2.set_ylabel("Complaints")
plt.tight_layout(); plt.savefig("figures/05_priority_map.png", dpi=120, bbox_inches="tight"); plt.show()

print(cpi[["theme", "volume", "sentiment", "relief_rate", "CPI"]].round(3).to_string(index=False))
print("\nRanked by raw volume :", list(cpi.sort_values("volume", ascending=False)["theme"]))
print("Ranked by CPI        :", list(cpi["theme"]))

# %% [markdown]
# **What the index surfaces.** Ranking by CPI rather than volume changes the decision: *accessing and
# withdrawing funds* overtakes the larger fraud theme (it is the most negative and among the least
# resolved), and *investment scams*, the smallest theme by count, rises to third because it is highly
# negative and almost never resolved. Volume alone would have buried both.

# %% [markdown]
# ## 8. Complaints by Product Type
#
# The themes describe *what went wrong*. The CFPB's product field describes *what kind of product* the
# complaint is about, a second, independent lens. Grouping by product shows where the volume and the
# anger sit, and where the investment scams concentrate.

# %%
import matplotlib
sp_map = {"Mobile or digital wallet": "Mobile / digital wallet",
          "Domestic (US) money transfer": "Domestic transfer",
          "Virtual currency": "Virtual currency",
          "International money transfer": "International transfer"}
df["product"] = df["Sub-product"].map(sp_map).fillna("Other")
prod = (df.groupby("product").agg(n=("relief", "size"), sentiment=("finbert", "mean"),
        relief=("relief", "mean")).reset_index().sort_values("n"))

norm = matplotlib.colors.Normalize(vmin=-0.42, vmax=-0.26)
colors = [matplotlib.cm.Reds(0.25 + 0.75 * (1 - norm(s))) for s in prod["sentiment"]]  # angrier = darker red
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(prod["product"], prod["n"], color=colors)
for y, (v, s) in enumerate(zip(prod["n"], prod["sentiment"])):
    ax.text(v + 15, y, f"{v:,}  ({s:.2f})", va="center", fontsize=9)
ax.set_title("Complaints by product type (bar colour = sentiment, redder is angrier)")
ax.set_xlabel("complaints"); ax.set_xlim(0, prod["n"].max() * 1.18)
plt.tight_layout(); plt.savefig("figures/07_by_product.png", dpi=120, bbox_inches="tight"); plt.show()

print(prod.assign(relief_pct=(prod["relief"] * 100).round(1), sent=prod["sentiment"].round(3))
      [["product", "n", "sent", "relief_pct"]].to_string(index=False))
vc_share = (df[df["topic"] == 4]["product"] == "Virtual currency").mean() * 100
print(f"\nAmong investment-scam complaints, {vc_share:.0f}% are about virtual currency.")

# %% [markdown]
# **The finding.** Mobile wallets carry the most complaints, but **mobile wallets and virtual currency
# draw the most negative language** (both around -0.39), while international transfers are the calmest
# (about -0.29). And the two lenses connect: nearly **half of all investment-scam complaints are about
# virtual currency**, so the crypto-era scam story shows up clearly when the theme and product views are
# crossed.

# %% [markdown]
# ## 9. Summary and Recommendations
#
# The pipeline turns 4,282 unread complaints into one decision-ready picture:
#
# - **Fraud and unauthorized transactions is the biggest theme** (~1,760 complaints) and strongly
#   negative. The distinct, smaller **investment-scam** theme is among the angriest of all. Together
#   they mark where the money-anger and the reputational risk concentrate, so that is where prevention
#   and faster, clearer complaint handling pay off first.
# - **Accessing and withdrawing funds is the sharpest finding, and only the domain model sees it.** It
#   is the second-largest theme and the single most negative one under FinBERT. A generic sentiment
#   tool made it look mild, because these complaints are often politely worded appeals to get funds
#   released; read with the right tool it becomes a clear priority.
# - **Wire transfers and cheque deposits** are mid-sized, consistently negative operational clusters.
# - **Over time**, the investment-scam share drifted upward across 2019 to 2023, and those scams
#   concentrate in virtual currency, the crypto-era fraud story.
#
# Two points about method run through the whole analysis:
#
# 1. **The choice of sentiment model changes the answer.** A generic tool called 38% of these
#    complaints positive and made the most-negative theme look mild; the finance-domain model
#    disagreed on 25% of complaints and reordered the priorities a business would act on.
# 2. **The Complaint Priority Index turns description into a decision.** Ranking themes by volume,
#    domain sentiment, and how rarely they are resolved reorders what a simple count would show.
#
# A written report of these results is in `report.md`, and an interactive version is in
# `dashboard/index.html`.
#
# ### Limitations
#
# FinBERT is trained on financial news, not complaints specifically, so it is a strong proxy rather
# than ground truth; NMF gives soft themes read with judgement (one cluster here is genuinely noise);
# and a complaints database reflects people motivated enough to file formally, not all customers.
#
# *Data: public CFPB Consumer Complaint Database. No confidential or employer data was used.*
