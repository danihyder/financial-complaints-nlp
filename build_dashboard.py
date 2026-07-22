"""Builds a self-contained INTERACTIVE dashboard app (Tableau/Power BI style) from the complaint
analysis: live filters (year range, sentiment model, outcome), click-a-theme cross-filtering, and
KPIs + a Complaint Priority Index that recompute client-side.

Outputs:
  dashboard/index.html          -- full standalone app (open in any browser / GitHub Pages)
  dashboard/_artifact_body.html -- body-only fragment (for the claude.ai Artifact preview)

Row-level aggregatable data is embedded as JSON; all filtering/aggregation happens in-browser.
No employer data; company names excluded from the analysis.
"""
import re, os, json, numpy as np, pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from plotly.offline import get_plotlyjs

os.makedirs("dashboard", exist_ok=True)

# ---------- data + features ----------
df = pd.read_csv("data/complaints_money_transfer.csv")
df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
df["year"] = df["Date received"].dt.year
df["finbert"] = pd.read_csv("data/finbert_scores.csv")["finbert_score"].values
df["relief"] = df["Company response to consumer"].isin(
    ["Closed with monetary relief", "Closed with non-monetary relief"]).astype(int)

brand = {"paypal","coinbase","venmo","cashapp","cash app","chase","wells","fargo","wells fargo",
    "bank america","america","zelle","citibank","chase bank","cash","app","bank","moneygram",
    "western union","western","union","wu","binance","gemini","robinhood","chime","remitly","square",
    "skrill","revolut","wise","kraken","stripe","affirm","klarna","varo","sofi","dave","netspend"}
stop = set(stopwords.words("english")) | brand | {"xxxx","xx","xxxxxxxx","account","money","company",
    "would","said","told","also","get","got","us","im","ive","dont","didnt","day","days"}
def clean(t):
    t = str(t).lower(); t = re.sub(r"x{2,}", " ", t); t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
df["clean"] = df["Consumer complaint narrative"].apply(clean)

tf = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.6, stop_words=list(stop), ngram_range=(1,2))
X = tf.fit_transform(df["clean"])
df["topic"] = NMF(n_components=7, random_state=42, init="nndsvd", max_iter=400).fit_transform(X).argmax(1)
LABELS = {0:"Fraud & unauthorized txns",2:"Cheque deposits",3:"Wire transfers",
          4:"Investment scams",5:"Accessing / withdrawing funds",6:"Account limitations"}
sia = SentimentIntensityAnalyzer()
df["vader"] = df["Consumer complaint narrative"].apply(lambda t: sia.polarity_scores(str(t))["compound"])

# embed the full dataset; the noise cluster (topic 1) is simply left out of the theme breakdowns
d = df.copy()

# ---------- product-type dimension (a lens distinct from the problem themes) ----------
SP_LABELS = ["Mobile / digital wallet", "Domestic transfer", "Virtual currency",
             "International transfer", "Other"]
sp_map = {"Mobile or digital wallet": 0, "Domestic (US) money transfer": 1,
          "Virtual currency": 2, "International money transfer": 3}
d["sp"] = d["Sub-product"].map(sp_map).fillna(4).astype(int)

DATA = {
    "topic": [int(x) for x in d["topic"]],
    "year": [int(x) for x in d["year"]],
    "fin": [round(float(x), 3) for x in d["finbert"]],
    "vad": [round(float(x), 3) for x in d["vader"]],
    "relief": [int(x) for x in d["relief"]],
    "sp": [int(x) for x in d["sp"]],
}
PAYLOAD = json.dumps({"data": DATA, "labels": {str(k): v for k, v in LABELS.items()},
                      "sp_labels": SP_LABELS,
                      "years": sorted(int(y) for y in d["year"].unique())})

# ---------- HTML ----------
STYLE = """
<style>
 :root{--ink:#0f172a;--mut:#64748b;--line:#e6eaf0;--bg:#f6f8fb;--card:#fff;--blue:#2563eb;
   --blue2:#1d4ed8;--green:#059669;--red:#dc2626;--chip:#eef2f7;}
 *{box-sizing:border-box} body{margin:0;background:var(--bg)}
 .app{max-width:1160px;margin:0 auto;padding:26px 18px 60px;
   font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--ink)}
 h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em} .sub{color:var(--mut);margin:0;font-size:14px}
 .bar{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-end;background:var(--card);
   border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:18px 0}
 .ctl{display:flex;flex-direction:column;gap:6px} .ctl label{font-size:11px;font-weight:600;
   text-transform:uppercase;letter-spacing:.04em;color:var(--mut);display:flex;align-items:center;gap:5px}
 .info{position:relative;display:inline-flex;align-items:center;justify-content:center;width:15px;
   height:15px;border-radius:50%;background:#c3ccd8;color:#fff;font-size:10px;font-weight:700;
   font-style:italic;font-family:Georgia,serif;cursor:help;text-transform:none;letter-spacing:0}
 .info:hover{background:var(--blue)}
 .info .tip{position:absolute;top:26px;left:50%;width:240px;background:#0f172a;color:#e5eaf1;
   padding:11px 13px;border-radius:9px;font-size:12px;font-weight:400;line-height:1.5;
   text-transform:none;letter-spacing:0;opacity:0;visibility:hidden;
   transform:translateX(-50%) translateY(-4px);transition:.14s;z-index:60;
   box-shadow:0 8px 24px rgba(15,23,42,.28)}
 .info:hover .tip{opacity:1;visibility:visible;transform:translateX(-50%) translateY(0)}
 .info .tip b{color:#93c5fd;font-weight:600} .info .tip .r{color:#86efac;font-weight:600}
 .info .tip .n{color:#fca5a5;font-weight:600}
 .info .tip::before{content:"";position:absolute;bottom:100%;left:50%;transform:translateX(-50%);
   border:6px solid transparent;border-bottom-color:#0f172a}
 .info.il .tip{left:0;transform:translateX(0) translateY(-4px)}
 .info.il:hover .tip{transform:translateX(0) translateY(0)}
 .info.il .tip::before{left:12px;transform:none}
 .seg{display:flex;background:var(--chip);border-radius:9px;padding:3px}
 .seg button{border:0;background:transparent;padding:6px 11px;border-radius:7px;font-size:13px;
   cursor:pointer;color:var(--mut);font-weight:600}
 .seg button.on{background:var(--card);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.08)}
 .slider{position:relative;width:220px;height:30px}
 .slider .track{position:absolute;top:13px;left:0;width:100%;height:4px;background:#dbe2ea;border-radius:2px}
 .slider .fill{position:absolute;top:13px;height:4px;background:var(--blue);border-radius:2px}
 .slider input[type=range]{position:absolute;top:0;left:0;width:100%;height:30px;margin:0;background:none;
   pointer-events:none;-webkit-appearance:none;appearance:none}
 .slider input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;pointer-events:auto;height:18px;
   width:18px;border-radius:50%;background:var(--blue);border:2px solid #fff;cursor:pointer;
   box-shadow:0 1px 3px rgba(0,0,0,.35)}
 .slider input[type=range]::-moz-range-thumb{pointer-events:auto;height:16px;width:16px;border-radius:50%;
   background:var(--blue);border:2px solid #fff;cursor:pointer}
 .yr{font-variant-numeric:tabular-nums;font-weight:700;font-size:13px;min-width:34px;text-align:center}
 .reset{margin-left:auto;border:1px solid var(--line);background:var(--card);border-radius:9px;
   padding:8px 14px;cursor:pointer;font-weight:600;font-size:13px;color:var(--ink)}
 .reset:hover{border-color:var(--blue);color:var(--blue)}
 .focusnote{font-size:13px;color:var(--blue);font-weight:600;min-height:18px;margin:2px 0 0}
 .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:6px 0 4px}
 .kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:15px 16px}
 .kpi .n{font-size:27px;font-weight:700;color:var(--blue);font-variant-numeric:tabular-nums}
 .kpi .l{font-size:12px;color:var(--mut);margin-top:3px;line-height:1.3}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
 .panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 8px 6px}
 .panel h3{margin:2px 10px 0;font-size:14px} .panel .hint{margin:1px 10px 6px;font-size:11px;color:var(--mut)}
 .panel .note{margin:10px 10px 3px;font-size:11.5px;color:var(--mut);line-height:1.55;
   border-top:1px solid var(--line);padding-top:9px}
 .panel .note b{color:var(--ink);font-weight:600}
 .full{grid-column:1 / -1}
 .foot{margin-top:26px;font-size:12px;color:var(--mut);border-top:1px solid var(--line);padding-top:14px}
 .js-plot{width:100%;height:330px}
 @media(max-width:820px){.grid{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}
</style>
"""

BODY = """
<div class="app">
  <h1>Digital-Payment Complaints: Interactive Analytics</h1>
  <p class="sub">Public financial complaints about digital payments. Filter by year and outcome, switch
  the sentiment model, and click any theme or product to cross-filter every chart.</p>

  <div class="bar">
    <div class="ctl"><label>Year range</label>
      <div style="display:flex;gap:10px;align-items:center">
        <span class="yr" id="yminL"></span>
        <div class="slider"><div class="track"></div><div class="fill" id="fill"></div>
          <input id="ymin" type="range"><input id="ymax" type="range"></div>
        <span class="yr" id="ymaxL"></span>
      </div>
    </div>
    <div class="ctl"><label>Sentiment model
      <span class="info">i<span class="tip">How the complaint text is scored for sentiment.<br><br>
      <b>FinBERT</b>: trained on financial language.<br>
      <b>VADER</b>: a general-purpose model built for social media.<br><br>
      Switching between them shows how the choice of model changes the results.</span></span></label>
      <div class="seg" id="model">
        <button data-v="fin" class="on">FinBERT</button>
        <button data-v="vad">VADER</button>
      </div>
    </div>
    <div class="ctl"><label>Outcome
      <span class="info il">i<span class="tip">How the company resolved each complaint.<br><br>
      <span class="r">Won relief</span>: the customer got money back or a fix (a reversed fee,
      restored access).<br>
      <span class="n">Explanation only</span>: the company replied but gave nothing concrete.<br>
      <b>All</b>: no filter.</span></span></label>
      <div class="seg" id="outcome">
        <button data-v="all" class="on">All</button>
        <button data-v="relief">Won relief</button>
        <button data-v="expl">Explanation only</button>
      </div>
    </div>
    <button class="reset" id="reset">Reset filters</button>
  </div>
  <p class="focusnote" id="focusnote"></p>

  <div class="kpis" id="kpis"></div>

  <div class="grid">
    <div class="panel"><h3>Complaint Priority Index</h3>
      <p class="hint">Click a bar to focus that theme across the whole dashboard.</p>
      <div id="cpi" class="js-plot"></div>
      <p class="note">A combined score ranking each theme by how many complaints it has, how negative they are, and how
      rarely they are resolved.</p></div>
    <div class="panel"><h3>Priority map</h3>
      <p class="hint">Each bubble is one theme.</p>
      <div id="bubble" class="js-plot"></div>
      <p class="note">Each theme placed by its number of complaints and average sentiment; bubble size
      is the priority score, and each theme keeps its own colour across the dashboard.</p></div>
    <div class="panel"><h3>Theme mix over time</h3>
      <p class="hint">Share of each year's complaints.</p>
      <div id="trend" class="js-plot"></div>
      <p class="note">The share of each year's complaints falling into each theme.</p></div>
    <div class="panel"><h3>Sentiment distribution</h3>
      <p class="hint">Switch the model above to compare the two.</p>
      <div id="dist" class="js-plot"></div>
      <p class="note">The spread of complaint sentiment, from negative on the left to positive on the right.</p></div>
    <div class="panel full"><h3>Complaints by product type</h3>
      <p class="hint">A different lens: what kind of payment product each complaint is about. Responds
      to every filter above.</p>
      <div id="sp" class="js-plot" style="height:300px"></div>
      <p class="note">The mix of product types behind the complaints, each bar coloured by how negative those complaints
      are.</p></div>
  </div>

  <p class="foot">Data: public US CFPB Consumer Complaint Database (money transfer / virtual currency /
  money service, complaints with narratives, 2017-2024). A small low-coherence cluster of complaints is
  left out of the theme breakdowns. No confidential or employer data; company names excluded from the
  analysis.</p>
</div>
"""

APP = """
<script>
const P = __PAYLOAD__;
const D = P.data, LAB = P.labels, SP = P.sp_labels;
const THEMES = Object.keys(LAB).map(Number);
const YMIN = P.years[0], YMAX = P.years[P.years.length-1];
const COL = {"0":"#dc2626","2":"#059669","3":"#d97706","4":"#7c3aed","5":"#2563eb","6":"#64748b"};
const N = D.topic.length;
let S = {ymin:YMIN, ymax:YMAX, model:"fin", outcome:"all", focus:null};
const PLOT = {displayModeBar:false, responsive:true};
const LY = {template:"plotly_white", margin:{t:8,l:48,r:12,b:34}, font:{size:11},
            paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)"};

function baseIdx(){ // year + outcome filter (not focus, not model)
  const out=[];
  for(let i=0;i<N;i++){
    if(D.year[i]<S.ymin||D.year[i]>S.ymax) continue;
    if(S.outcome==="relief"&&D.relief[i]!==1) continue;
    if(S.outcome==="expl"&&D.relief[i]!==0) continue;
    out.push(i);
  } return out;
}
const sentOf = i => S.model==="fin"?D.fin[i]:D.vad[i];
function mm(arr){const mn=Math.min(...arr),mx=Math.max(...arr);return arr.map(v=>mx>mn?(v-mn)/(mx-mn):0);}

function aggregate(idx){
  const a={}; THEMES.forEach(t=>a[t]={n:0,sent:0,rel:0});
  idx.forEach(i=>{const t=D.topic[i]; if(a[t]===undefined) return; a[t].n++;a[t].sent+=sentOf(i);a[t].rel+=D.relief[i];});
  const T=THEMES.filter(t=>a[t].n>0);
  T.forEach(t=>{a[t].mean=a[t].sent/a[t].n;a[t].reliefRate=a[t].rel/a[t].n;});
  const vol=mm(T.map(t=>a[t].n)), neg=mm(T.map(t=>-a[t].mean)), unr=mm(T.map(t=>1-a[t].reliefRate));
  T.forEach((t,k)=>{a[t].cpi=100*(0.4*vol[k]+0.4*neg[k]+0.2*unr[k]);});
  return {a,T};
}

function kpis(idx){
  const n=idx.length;
  const neg=idx.filter(i=>sentOf(i)<=-0.05).length/(n||1)*100;
  const rel=idx.filter(i=>D.relief[i]===1).length/(n||1)*100;
  const {a,T}=aggregate(idx);
  let top="-",best=-1; T.forEach(t=>{if(a[t].cpi>best){best=a[t].cpi;top=LAB[t];}});
  const K=[[n.toLocaleString(),"complaints in view"],
           [neg.toFixed(0)+"%","negative sentiment"],
           [rel.toFixed(1)+"%","won relief"],
           [top,"highest-priority theme"]];
  document.getElementById("kpis").innerHTML=K.map(k=>
    `<div class="kpi"><div class="n">${k[0]}</div><div class="l">${k[1]}</div></div>`).join("");
}

const isThemeFocus = t => S.focus && S.focus.k==="t" && S.focus.v===t;
const isProdFocus  = s => S.focus && S.focus.k==="p" && S.focus.v===s;

function drawCPI(agg){
  const T=[...agg.T].sort((x,y)=>agg.a[x].cpi-agg.a[y].cpi);
  Plotly.react("cpi",[{type:"bar",orientation:"h",x:T.map(t=>agg.a[t].cpi),y:T.map(t=>LAB[t]),
    marker:{color:T.map(t=>(S.focus&&S.focus.k==="t"&&!isThemeFocus(t))?"#cbd5e1":COL[t])},
    customdata:T.map(t=>t),
    hovertemplate:"%{y}<br>CPI %{x:.0f}<extra></extra>"}],
    {...LY,margin:{t:8,l:150,r:12,b:30},xaxis:{title:""}},PLOT);
}
function drawBubble(agg){
  const T=agg.T, big=Math.max(...T.map(t=>agg.a[t].cpi),1);
  Plotly.react("bubble",[{type:"scatter",mode:"markers+text",
    x:T.map(t=>agg.a[t].mean),y:T.map(t=>agg.a[t].n),text:T.map(t=>LAB[t].slice(0,16)),textposition:"top center",
    textfont:{size:9},
    marker:{size:T.map(t=>agg.a[t].cpi),sizemode:"area",sizeref:2*big/(46*46),sizemin:5,
      color:T.map(t=>COL[t]),
      line:{width:T.map(t=>isThemeFocus(t)?3:1),color:"#0f172a"}},
    customdata:T.map(t=>[agg.a[t].cpi,agg.a[t].reliefRate*100,t]),
    hovertemplate:"<b>%{text}</b><br>vol %{y}<br>sent %{x:.2f}<br>relief %{customdata[1]:.1f}%<br>CPI %{customdata[0]:.0f}<extra></extra>"}],
    {...LY,xaxis:{title:"sentiment (angrier <-)"},yaxis:{title:"complaints"}},PLOT);
}
function drawTrend(idx){
  const ts=(S.focus&&S.focus.k==="t")?[S.focus.v]:THEMES;
  const yrs=[]; for(let y=Math.max(S.ymin,2018);y<=Math.min(S.ymax,2023);y++) yrs.push(y);
  const per={}; yrs.forEach(y=>per[y]={tot:0}); idx.forEach(i=>{const y=D.year[i],t=D.topic[i];if(per[y]&&t!==1){per[y].tot++;per[y][t]=(per[y][t]||0)+1;}});
  const traces=ts.map(t=>({type:"scatter",mode:"lines+markers",name:LAB[t],
    x:yrs,y:yrs.map(y=>per[y].tot?100*(per[y][t]||0)/per[y].tot:0),
    line:{width:3,color:COL[t]}, customdata:yrs.map(()=>t)}));
  Plotly.react("trend",traces,{...LY,showlegend:ts.length>1,legend:{font:{size:9},orientation:"h",y:-0.2},
    yaxis:{title:"% of year",ticksuffix:"%"}},PLOT);
}
function drawDist(idx){
  Plotly.react("dist",[{type:"histogram",x:idx.map(sentOf),nbinsx:40,
    marker:{color:S.model==="fin"?"#2563eb":"#64748b"}}],
    {...LY,xaxis:{title:"negative to positive"},yaxis:{title:"complaints"},bargap:0.02},PLOT);
}
function drawSubproduct(idx){
  const a=SP.map(()=>({n:0,sent:0}));
  idx.forEach(i=>{const s=D.sp[i]; a[s].n++; a[s].sent+=sentOf(i);});
  const rows=[]; for(let s=0;s<SP.length;s++) if(a[s].n>0) rows.push({s,n:a[s].n,mean:a[s].sent/a[s].n});
  rows.sort((x,y)=>x.n-y.n);
  Plotly.react("sp",[{type:"bar",orientation:"h",
    x:rows.map(r=>r.n), y:rows.map(r=>SP[r.s]),
    marker:{color:rows.map(r=>r.mean),colorscale:[[0,"#b91c1c"],[0.5,"#f87171"],[1,"#fecaca"]],
      cmin:-0.42,cmax:-0.26,
      line:{width:rows.map(r=>isProdFocus(r.s)?2.5:0),color:"#0f172a"},
      colorbar:{title:{text:"avg sentiment<br>(redder = angrier)",side:"right"},thickness:10,len:0.85,x:1.02,
        tickfont:{size:9}}},
    customdata:rows.map(r=>r.s), text:rows.map(r=>r.mean.toFixed(2)), textposition:"none",
    hovertemplate:"%{y}<br>%{x} complaints<br>avg sentiment %{text}<extra></extra>"}],
    {...LY,margin:{t:8,l:150,r:50,b:30},xaxis:{title:"complaints"}},PLOT);
}

function render(){
  const bidx=baseIdx();
  const themeAxis=(S.focus&&S.focus.k==="p")?bidx.filter(i=>D.sp[i]===S.focus.v):bidx;
  const prodAxis =(S.focus&&S.focus.k==="t")?bidx.filter(i=>D.topic[i]===S.focus.v):bidx;
  const fullIdx  =(S.focus===null)?bidx:bidx.filter(i=>S.focus.k==="t"?D.topic[i]===S.focus.v:D.sp[i]===S.focus.v);
  kpis(fullIdx);
  drawCPI(aggregate(themeAxis)); drawBubble(aggregate(themeAxis)); drawTrend(themeAxis);
  drawDist(fullIdx); drawSubproduct(prodAxis);
  document.getElementById("focusnote").textContent =
    S.focus===null ? "" : "Focused on: "+(S.focus.k==="t"?LAB[S.focus.v]:SP[S.focus.v])+"  (click it again, or Reset, to clear)";
}
function toggleTheme(t){ S.focus=isThemeFocus(t)?null:{k:"t",v:t}; render(); }
function toggleProd(s){ S.focus=isProdFocus(s)?null:{k:"p",v:s}; render(); }
function attachClicks(){
  document.getElementById("cpi").on("plotly_click",e=>toggleTheme(e.points[0].customdata));
  document.getElementById("bubble").on("plotly_click",e=>toggleTheme(e.points[0].customdata[2]));
  document.getElementById("trend").on("plotly_click",e=>toggleTheme(e.points[0].customdata));
  document.getElementById("sp").on("plotly_click",e=>toggleProd(e.points[0].customdata));
}

// ---- controls ----
function initRange(id,val){const el=document.getElementById(id);el.min=YMIN;el.max=YMAX;el.value=val;return el;}
const emin=initRange("ymin",YMIN), emax=initRange("ymax",YMAX);
const fillEl=document.getElementById("fill"), span=(YMAX-YMIN)||1;
function syncLabels(){
  document.getElementById("yminL").textContent=S.ymin;
  document.getElementById("ymaxL").textContent=S.ymax;
  const l=(S.ymin-YMIN)/span*100, r=(S.ymax-YMIN)/span*100;
  fillEl.style.left=l+"%"; fillEl.style.width=(r-l)+"%";
  emin.style.zIndex = (S.ymin>=YMAX-1)?5:3; emax.style.zIndex=4;  // keep min thumb grabbable at the top end
}
emin.oninput=()=>{S.ymin=Math.min(+emin.value,S.ymax);emin.value=S.ymin;syncLabels();render();};
emax.oninput=()=>{S.ymax=Math.max(+emax.value,S.ymin);emax.value=S.ymax;syncLabels();render();};
function seg(id,key){document.querySelectorAll("#"+id+" button").forEach(b=>b.onclick=()=>{
  S[key]=b.dataset.v;document.querySelectorAll("#"+id+" button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on");render();});}
seg("model","model"); seg("outcome","outcome");
document.getElementById("reset").onclick=()=>{
  S={ymin:YMIN,ymax:YMAX,model:"fin",outcome:"all",focus:null};
  emin.value=YMIN;emax.value=YMAX;syncLabels();
  document.querySelectorAll(".seg button").forEach(b=>b.classList.toggle("on",b.dataset.v==="fin"||b.dataset.v==="all"));
  render();};
syncLabels(); render(); attachClicks();
</script>
"""

plotly_js = f"<script>{get_plotlyjs()}</script>"
app = APP.replace("__PAYLOAD__", PAYLOAD)

with open("dashboard/index.html", "w") as f:
    f.write("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Digital-Payment Complaints Interactive Dashboard</title>"
            + STYLE + plotly_js + "</head><body>" + BODY + app + "</body></html>")
with open("dashboard/_artifact_body.html", "w") as f:
    f.write(STYLE + plotly_js + BODY + app)

print("wrote interactive dashboard/index.html and dashboard/_artifact_body.html")
print("rows embedded:", len(DATA["topic"]), "| themes:", [LABELS[t] for t in LABELS])
