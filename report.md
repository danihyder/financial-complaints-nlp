# An NLP Analysis of Consumer Financial Complaints
### Digital-payment complaints: themes, sentiment, priority, and outcomes

**Data:** US CFPB Consumer Complaint Database (public)
**Methods:** text mining, topic modelling, sentiment analysis with a finance-specific model, outcome analysis by theme
**Companion files:** the full notebook (`complaints-nlp-analysis.ipynb`) and an interactive dashboard (`dashboard/index.html`)

---

## Executive summary

Companies collect huge amounts of customer text: complaints, reviews, support messages. Most of it
never gets read as a whole, so the pattern hiding inside it goes unseen. This report reads 4,282
public complaints about money transfers, digital wallets, and virtual currency all at once, and turns
them into a simple, ranked answer to one question: where are customers hurting most, and where would
fixing things help the most?

Four things stand out:

1. **Using the wrong tool gives the wrong answer.** A common, general-purpose sentiment tool (VADER)
   marked 38% of these complaints as *positive*, simply because angry customers often write politely.
   A model built for financial language (FinBERT) marked only 1% as positive. The two disagree on a
   quarter of all complaints, and picking the right one changes which problems look most urgent.
2. **Counting complaints is not enough.** A simple score I built, the Complaint Priority Index, ranks
   each problem by how many people it hits, how upset they are, and how rarely they get helped. It
   moves "trouble getting your money out" above fraud, and pushes "investment scams" from the smallest
   problem to the third biggest priority.
3. **Scam victims get the least help.** Only 6.4% of scam complaints ended with the customer getting
   anything back, compared with about 16% overall. The word "scam" is the single clearest sign a
   complaint will get a reply and nothing more.
4. **What a complaint is about decides how it ends.** Resolution rates run from 6% for scams to 18%
   for money-access problems, a gap far too large to be chance. The type of problem, not the wording,
   is what shapes the outcome.

The rest of this report explains the data, how the analysis was done and why, the findings in full,
and where the limits are.

---

## 1. Why this matters

Complaints are the most honest feedback a business gets. Read one at a time, they are just stories.
Read together, they show you exactly where a product is failing people. The goal here was to build
that bigger picture from public data, and to do it in a way you can check at every step, rather than
a black box. So this report shows *how* each decision was made: how the word list was built, how many
themes to look for, which sentiment model to trust, and how the priorities were ranked. Those
decisions are where an analysis is either right or wrong.

## 2. The data

The source is the **US Consumer Financial Protection Bureau (CFPB) complaints database**, which is
public and published by a government agency. I focused on one product area, *money transfers, virtual
currency, and money services* (the closest fit to modern payments and fintech), and kept only the
complaints where the customer wrote out what happened in their own words.

That left **4,282 complaints from 2017 to 2024**. The data is fully public and holds no confidential
or employer information. Company names are in the source, but I left them out of the analysis on
purpose, so the work is about the *problems*, not about naming and shaming any company.

## 3. How the analysis was done

### 3.1 Building the word list

A computer cannot read a topic model; it reads numbers. Turning the complaints into useful numbers is
most of the work, so every choice mattered:

- **Cleaning the text:** made everything lower-case, removed the `XXXX` blocks the CFPB uses to hide
  personal details (left in, they would swamp everything), and kept only letters.
- **Removing filler words in three passes:** ordinary words like "the" and "and"; words that are
  common here but say nothing ("account", "money"); and company names, taken out on purpose so the
  model groups complaints by problem, not by which company was involved.
- **Keeping two-word phrases** so things like "wire transfer" and "fell victim" stay intact.
- **Dropping words that are too rare or too common** to tell one complaint from another.
- **Weighting words** so the ones that make a complaint distinctive count for more than the ones that
  show up everywhere.

### 3.2 Choosing how many themes to find

I used a method called NMF to group the complaints into themes. The hard question is always: how many
themes? I did not just pick a round number. I measured how *coherent* the themes were (do their top
words really appear together in real complaints?) across a range of choices, and checked how evenly
the complaints were split.

The most coherent choice was 5 themes, but at 5 themes one giant "everything else" group swallowed
72% of the complaints, which is useless. Splitting further breaks that group into clear, separate
problems. **Seven themes** turned out to be the sweet spot: coherent enough, and detailed enough to
act on. (I deliberately did not use the usual "error score" to decide, because it always rewards more
themes and never tells you when to stop.)

### 3.3 Choosing a sentiment model that fits the subject

**FinBERT is the model behind every finding here.** It was built for financial writing and reads full
sentences in context. I kept the general-purpose model (VADER) in only as a comparison, to show *why*
a finance-specific model is needed (see 4.2). I did not trust VADER's numbers; it is there to make the
point clear.

**Why FinBERT, and what could be even better.** FinBERT is the right kind of tool for this, but it is
not perfect: it was trained on financial *news*, not *complaints*, so it is a close cousin rather than
an exact match. Better options, in rising order of effort, would be a version tuned for tone, a model
trained on customer reviews or complaints directly, a modern large language model used with a simple
prompt (which can even give the reason for its rating), or, best of all, a model trained on a few
hundred hand-labelled complaints. FinBERT is the sensible choice for the effort involved, and these
better options are the natural next step.

### 3.4 The Complaint Priority Index (my own measure)

To decide what to fix first, I built a single, simple score:

> **Priority = 40% how many complaints + 40% how negative + 20% how rarely resolved**
> (each part put on the same 0-to-1 scale).

The idea: a problem deserves attention based on how many people it affects, how upset they are, and
how often they are left without help. The weights are a clear, adjustable choice, not something
hidden.

### 3.5 Linking each theme to its outcome

Finally, I connected the themes to what actually happened to the customer. Using the CFPB's record of
how each case ended, I marked every complaint as either **got relief** (money back or a fix) or
**explanation only** (a reply, but nothing given), then measured the relief rate for each theme. A
chi-square test checks whether the differences between themes are real or just chance.

(An earlier version trained a model on the raw complaint words to predict the outcome. It was dropped:
its "important" words were mostly generic noise with no real meaning, and the theme-level view below is
both more honest and more useful.)

## 4. What the data shows

### 4.1 The overall picture

Complaints in this area grew sharply over the years, and the CFPB's own labels are dominated by fraud
and scams, over a third of everything. That is the backdrop for the themes below. *(See Figure 1 in
the notebook.)*

### 4.2 The "politeness trap"

The clearest lesson about method: **the tool you pick changes the result.** VADER called **38% of
these complaints positive**; FinBERT called **1%** positive. The two disagree on **a quarter of all
complaints (1,088 of them)**, and on a closer look FinBERT is the one getting it right.

The reason is simple and repeats itself: many people open a complaint politely ("Dear Support Team, I
hope this finds you well, I have always valued..."), and the general tool scores the good manners
instead of the anger underneath. This is not a rare glitch; it clusters in the polite, formal
"account limits and appeals" complaints, exactly where the general tool would mislead you most. **A
model that fits the subject is not a nice-to-have, it is required.**

### 4.3 The seven themes

The complaints split into: fraud and unauthorised transactions; trouble getting or withdrawing money;
wire transfers; cheque deposits; investment scams; account limits and appeals; and one small,
low-quality group of copy-paste text that I flag openly rather than dress up. Under the finance model,
every theme is strongly negative, with "trouble getting your money out" the most negative of all.

### 4.4 The priority ranking

Ranking by the Priority Index instead of by raw count changes the picture:

| Rank | Problem | Complaints | Sentiment | Got help | Priority |
|---|---|---:|---:|---:|---:|
| 1 | Trouble getting / withdrawing money | 1,199 | -0.39 | 18.3% | 66 |
| 2 | Fraud & unauthorised transactions | 1,760 | -0.33 | 16.4% | 64 |
| 3 | Investment scams | 156 | -0.37 | 6.4% | 52 |
| 4 | Account limits & appeals | 254 | -0.36 | 15.4% | 35 |
| 5 | Cheque deposits | 343 | -0.33 | 14.9% | 31 |
| 6 | Wire transfers | 469 | -0.27 | 12.4% | 18 |

Two changes matter. **Trouble getting your money out** jumps above the bigger fraud group, because it
is the most negative and one of the least likely to be resolved: people locked out of their own money,
staying polite, and rarely helped. And **investment scams**, the smallest group by count, climbs to
third, because people are furious and almost never get anything back. Counting alone would have hidden
both. *(See the interactive dashboard.)*

### 4.5 Which problems get resolved

How often a complaint ends with the customer actually getting something back varies sharply by theme:

| Theme | Complaints | Closed with relief |
|---|---:|---:|
| Investment scams | 156 | **6.4%** |
| Wire transfers | 469 | 12.4% |
| Cheque deposits | 343 | 14.9% |
| Account limits & appeals | 254 | 15.4% |
| Fraud & unauthorised transactions | 1,760 | 16.4% |
| Trouble getting / withdrawing money | 1,199 | 18.3% |

Every other problem is resolved 12% to 18% of the time; **investment scams sit at just 6.4%**, less
than half the overall rate of about 16%. A chi-square test confirms the gap is real, not chance
(p = 0.001). This sharpens the main message: **investment scams are the angriest, fastest-growing, and
least-resolved group of all**, and the current way complaints are handled serves those victims worst.

### 4.6 How things changed over time

Looking across 2018 to 2023, the share of **investment-scam complaints crept up** (from around 3% to
over 5%), which fits the wider rise in crypto-era investment fraud. Fraud and money-access stayed the
biggest problems throughout.

## 5. What I would recommend

For a payments or support team, the priorities follow straight from the analysis:

1. **Fix money-access first.** It is the top-priority problem: lots of people, the angriest language,
   and most of them left unresolved.
2. **Give scam complaints their own process.** They are rising, they carry the most anger, and with
   only 6.4% getting any help, the current handling effectively fails those customers, even where the
   company is not at fault. It is a reputation risk.
3. **Ask for specifics up front.** Since clear, specific complaints are the ones that get resolved,
   capturing the transaction details early would likely speed things up and improve outcomes.

## 6. What is my own work here

Stated plainly, because it matters:

- **The Complaint Priority Index** is a scoring measure I designed, not a standard output, and it
  visibly reorders priorities that a simple count would miss.
- **The "politeness trap"** is a specific, measured finding: a common tool fails on polite financial
  complaints, and that failure changes the conclusions.
- **Linking themes to outcomes** connects each problem to how often it is actually resolved, and shows
  (with a significance test) that scam complaints are handled far worse than any other kind.
- **Every method choice is shown and explained**, and the one weak theme is flagged rather than hidden.

## 7. The limits

- FinBERT was trained on financial news, not complaints, so it is a close proxy rather than a perfect
  fit. A production version would be trained on labelled complaints.
- The themes are soft groupings that need judgement to read, and one of them is genuinely noise.
- The relief rates describe what happened, not why. Outcomes also depend on company policy and the
  facts of each case, which this data does not capture, so the theme differences are a strong signal,
  not a full explanation.
- A complaints database only reflects people annoyed enough to file a formal complaint, not every
  customer.

## Appendix: how to reproduce this

The full analysis is in `complaints-nlp-analysis.ipynb`, with all the charts. The interactive version
is `dashboard/index.html`. The finance-model scores are saved in `data/finbert_scores.csv` and rebuild
from the notebook if deleted. The exact package versions are listed in `requirements.txt`.

*Data: public CFPB Consumer Complaint Database. No confidential or employer data was used. Company
names are left out of the entire analysis.*
