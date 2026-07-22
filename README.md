# An NLP Analysis of Consumer Financial Complaints

### ▶ [Open the interactive dashboard](https://danihyder.github.io/financial-complaints-nlp/dashboard/)

A domain-aware NLP study of digital-payment complaints. The point is not that sentiment and topic
modelling can be run, it is that the **modelling decisions are made explicitly and defended**: how
the vocabulary is built, how the number of topics is chosen, and why a generic sentiment tool is the
wrong instrument for financial text.

**Skills used:** text mining, topic modelling, domain-specific sentiment modelling benchmarked against
a baseline, temporal analysis, a priority index, and a product-type breakdown.
**Stack:** Python, pandas, scikit-learn (TF-IDF, NMF), NLTK (VADER), Hugging Face Transformers
(FinBERT), matplotlib.

> Built on fully public data. No confidential or employer data is used. Company names are
> deliberately excluded so the analysis is about problems, not providers.

---

## What makes this more than a tutorial

Most complaint-NLP projects stop at "here are some topics and here is the sentiment." The value is in
the decisions and in going past description, and this project shows its work on six of them:

1. **Vocabulary construction is deliberate.** Three layers of stop-words (standard, generic filler,
   and company names removed on purpose), n-grams so "wire transfer" survives, and `min_df` / `max_df`
   thresholds chosen to cut noise and over-common words. Nothing is dumped into a black box.
2. **The themes come from topic modelling.** After testing a range of options, seven topics gave the
   clearest, most usable separation, granular enough to act on without fragmenting into overlapping
   slivers.
3. **Sentiment is done for the domain.** VADER, the usual baseline, labels 38% of these complaints
   *positive*, because angry customers write politely ("Dear Support Team, I hope this finds you
   well..."). **FinBERT**, a model fine-tuned on financial text, disagrees on 25% of complaints and is
   correct on inspection. Tool choice changes the business conclusion, not just the numbers.
4. **An original decision metric.** I define a **Complaint Priority Index (CPI)** combining volume,
   domain sentiment, and how rarely a theme is resolved. It reorders priorities that volume alone
   misses, moving fund-access above fraud and lifting investment scams from smallest theme to third.
5. **A second lens: product type.** Breaking complaints down by payment product shows virtual currency
   and mobile wallets draw the angriest complaints, and that investment scams concentrate in crypto.
6. **The findings are read critically.** One weak, templated topic is openly flagged as uncategorised
   rather than dressed up, and the sentiment anomaly (a generic tool misreading polite complaints) is
   explained with evidence.

## Headline findings

- **The politeness trap:** a generic sentiment tool calls 38% of these complaints positive and
  misranks them; a finance-domain model corrects it on 25% of complaints.
- **The Complaint Priority Index reorders the picture:** *accessing and withdrawing funds* is the top
  priority (highest volume-weighted anger, rarely resolved), and *investment scams* rise to third
  despite being the smallest theme by count.
- **Crypto concentrates the scams:** virtual currency and mobile wallets draw the angriest complaints,
  and 46% of all investment-scam complaints are about virtual currency.

## Deliverables

| Asset | Description |
|---|---|
| `report.md` | Analysis report including executive summary, methodology, findings, recommendations |
| **[Interactive dashboard](https://danihyder.github.io/financial-complaints-nlp/dashboard/)** (`dashboard/index.html`) | A **self-contained interactive dashboard**. |
| `complaints-nlp-analysis.ipynb` | code, explanations, charts |
| `notebook.py` | Plain script |

## Key visual: the priority map

Each theme placed by volume against sentiment, bubble size = CPI, coloured by theme.

![Priority map](figures/05_priority_map.png)

## The data

Public complaints from the **US CFPB Consumer Complaint Database**, product category *"Money transfer,
virtual currency, or money service"* with a written narrative. 4,282 complaints, 2017 to 2024,
accessed via the `BEE-spoke-data/consumer-finance-complaints` mirror on Hugging Face. The CFPB is a
US federal agency that publishes these complaints publicly, so no confidentiality applies.

## Run it yourself

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
jupyter notebook complaints-nlp-analysis.ipynb
```

FinBERT scores are cached in `data/finbert_scores.csv`, so the notebook runs top to bottom in
seconds.

## Files

| File | What it is |
|---|---|
| `report.md` | Analysis report |
| `dashboard/index.html` | Self-contained interactive dashboard |
| `build_dashboard.py` | Regenerates the dashboard from the data |
| `complaints-nlp-analysis.ipynb` | The analysis: explanations, code, charts |
| `notebook.py` | Plain script  |
| `data/complaints_money_transfer.csv` | The public complaint subset |
| `data/finbert_scores.csv` | Cached FinBERT sentiment scores |
| `figures/` | Exported charts |

## Limitations

FinBERT is trained on financial news, not complaints specifically, so it is a strong proxy rather than
ground truth; NMF gives soft themes read with judgement (one cluster here is genuinely noise); and a
complaints database reflects people motivated enough to file formally, not all customers. A production
version would fine-tune a sentiment model on labelled complaint text and tune the topic count further.

---

*Data: public CFPB Consumer Complaint Database. No confidential or employer data was used.*
