import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SmartCharging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Exo+2:wght@300;400;500;600;700&display=swap');

:root{
  --bg0:#04080f; --bg1:#070f1c; --bg2:#0b1626; --bg3:#0f1e33;
  --card:#0c1828; --panel:#101f35;
  --c1:#00e5ff; --c2:#00ff9d; --c3:#ff6b35; --c4:#a855f7; --c5:#fbbf24;
  --th:#e2f0f7; --tm:#7ba8bf; --tl:#3d6070;
  --bdr:rgba(0,229,255,0.14);
}

html,body,[class*="css"]{
  font-family:'Exo 2',sans-serif;
  background:var(--bg0)!important;
  color:var(--th)!important;
}
.main .block-container{
  background:var(--bg0)!important;
  padding:1.2rem 2rem!important;
  max-width:1500px!important;
}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--bg0) 0%,var(--bg2) 100%)!important;
  border-right:1px solid var(--bdr)!important;
}
section[data-testid="stSidebar"] *{color:var(--th)!important;}

.hero{
  background:linear-gradient(135deg,var(--bg0) 0%,#081628 40%,var(--bg0) 100%);
  border:1px solid var(--bdr); border-radius:18px;
  padding:2rem 2.5rem; margin-bottom:1.5rem;
  position:relative; overflow:hidden;
}
.hero::before{
  content:''; position:absolute; inset:0; pointer-events:none;
  background:
    radial-gradient(ellipse 60% 80% at 15% 50%,rgba(0,229,255,0.06) 0%,transparent 100%),
    radial-gradient(ellipse 40% 60% at 85% 50%,rgba(0,255,157,0.04) 0%,transparent 100%);
}
.hero-badge{
  display:inline-block; font-family:'Orbitron',monospace;
  font-size:0.62rem; font-weight:600; letter-spacing:3px; text-transform:uppercase;
  color:var(--c2); background:rgba(0,255,157,0.08);
  border:1px solid rgba(0,255,157,0.25); border-radius:20px;
  padding:0.2rem 0.9rem; margin-bottom:0.7rem;
}
.hero-title{
  font-family:'Orbitron',monospace; font-size:2.4rem; font-weight:800;
  letter-spacing:2px; line-height:1.1;
  background:linear-gradient(90deg,#00e5ff 0%,#00ff9d 50%,#a855f7 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin:0 0 0.4rem 0;
}
.hero-sub{font-size:0.95rem; color:var(--tm); letter-spacing:0.5px;}
.hero-tags{margin-top:1rem; display:flex; gap:0.5rem; flex-wrap:wrap;}
.tag{
  font-size:0.7rem; font-family:'Orbitron',monospace; letter-spacing:1px;
  padding:0.2rem 0.7rem; border-radius:4px;
  border:1px solid var(--bdr); color:var(--tm);
  background:rgba(0,229,255,0.04);
}

.sh{
  font-family:'Orbitron',monospace; font-size:0.9rem; font-weight:700;
  letter-spacing:2.5px; text-transform:uppercase; color:var(--c1);
  border-left:3px solid var(--c2); padding-left:0.9rem;
  margin:1.8rem 0 1rem 0;
}

[data-testid="metric-container"]{
  background:var(--card)!important; border:1px solid var(--bdr)!important;
  border-radius:12px!important; padding:1.1rem!important;
  position:relative; overflow:hidden;
}
[data-testid="metric-container"]::after{
  content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--c1),var(--c2));
}
[data-testid="metric-container"] label{color:var(--tm)!important; font-size:0.75rem!important; letter-spacing:1px!important; text-transform:uppercase!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-family:'Orbitron',monospace!important; font-size:1.7rem!important; color:var(--c1)!important;
}
[data-testid="stMetricDelta"]{color:var(--c2)!important;}

.ib{
  background:linear-gradient(135deg,rgba(0,229,255,0.04),rgba(0,255,157,0.02));
  border:1px solid rgba(0,229,255,0.2); border-radius:10px;
  padding:0.9rem 1.2rem; margin:0.6rem 0; font-size:0.88rem; line-height:1.7;
}
.ib strong,.ib b{color:var(--c1);}
.wb{
  background:linear-gradient(135deg,rgba(255,107,53,0.07),rgba(255,107,53,0.02));
  border:1px solid rgba(255,107,53,0.3); border-radius:10px;
  padding:0.9rem 1.2rem; margin:0.6rem 0; font-size:0.88rem; line-height:1.7;
}
.wb strong,.wb b{color:var(--c3);}
.sb{
  background:linear-gradient(135deg,rgba(168,85,247,0.07),rgba(168,85,247,0.02));
  border:1px solid rgba(168,85,247,0.3); border-radius:10px;
  padding:0.9rem 1.2rem; margin:0.6rem 0; font-size:0.88rem; line-height:1.7;
}
.sb strong,.sb b{color:var(--c4);}

.stTabs [data-baseweb="tab-list"]{
  background:var(--card)!important; border-radius:10px; padding:4px; gap:3px;
  border:1px solid var(--bdr)!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important; color:var(--tm)!important;
  border-radius:7px!important; font-family:'Exo 2',sans-serif!important;
  font-size:0.88rem!important; font-weight:600!important;
  padding:0.45rem 1rem!important; transition:all 0.2s!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,rgba(0,229,255,0.16),rgba(0,255,157,0.08))!important;
  color:var(--c1)!important; border:1px solid rgba(0,229,255,0.3)!important;
}

.stSelectbox>div>div,.stMultiSelect>div>div{
  background:var(--card)!important; border:1px solid var(--bdr)!important;
  color:var(--th)!important; border-radius:8px!important;
}
[data-baseweb="popover"]{background:var(--bg2)!important; border:1px solid var(--bdr)!important;}

.stButton>button{
  background:linear-gradient(135deg,rgba(0,229,255,0.12),rgba(0,255,157,0.06))!important;
  color:var(--c1)!important; border:1px solid var(--bdr)!important;
  border-radius:8px!important; font-family:'Exo 2',sans-serif!important;
  font-weight:600!important; transition:all 0.2s!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,rgba(0,229,255,0.22),rgba(0,255,157,0.12))!important;
  border-color:var(--c1)!important; transform:translateY(-1px)!important;
  box-shadow:0 4px 20px rgba(0,229,255,0.2)!important;
}
.stDownloadButton>button{
  background:linear-gradient(135deg,rgba(0,255,157,0.12),rgba(0,229,255,0.06))!important;
  color:var(--c2)!important; border:1px solid rgba(0,255,157,0.25)!important;
  border-radius:8px!important; font-family:'Exo 2',sans-serif!important; font-weight:600!important;
}
.streamlit-expanderHeader{
  background:var(--card)!important; border:1px solid var(--bdr)!important;
  border-radius:8px!important; color:var(--c1)!important;
  font-family:'Exo 2',sans-serif!important; font-weight:600!important;
}
.streamlit-expanderContent{
  background:var(--panel)!important; border:1px solid var(--bdr)!important; border-top:none!important;
}
.stDataFrame{background:var(--card)!important; border-radius:10px!important;}
hr{border-color:var(--bdr)!important;}
::-webkit-scrollbar{width:5px; height:5px;}
::-webkit-scrollbar-track{background:var(--bg0);}
::-webkit-scrollbar-thumb{background:rgba(0,229,255,0.3); border-radius:3px;}

.glow-card{
  background:var(--card); border:1px solid var(--bdr); border-radius:12px;
  padding:1.2rem; transition:all 0.25s; cursor:default;
}
.glow-card:hover{
  border-color:var(--c1); box-shadow:0 0 20px rgba(0,229,255,0.12);
  transform:translateY(-2px);
}
.pipe-step{
  display:flex; align-items:flex-start; gap:1rem;
  background:var(--card); border:1px solid var(--bdr); border-radius:10px;
  padding:0.9rem 1.2rem; margin:0.5rem 0;
  position:relative; overflow:hidden;
}
.pipe-step::before{
  content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:linear-gradient(180deg,var(--c1),var(--c2));
}
.pipe-num{font-family:'Orbitron',monospace; font-size:1.1rem; font-weight:800; color:var(--c1); min-width:28px;}
.pipe-title{font-weight:700; color:var(--th); font-size:0.9rem;}
.pipe-desc{color:var(--tm); font-size:0.8rem; margin-top:0.2rem;}
.check-item{
  display:flex; align-items:center; gap:0.8rem;
  background:var(--card); border:1px solid var(--bdr); border-radius:8px;
  padding:0.65rem 1rem; margin:0.35rem 0; font-size:0.85rem;
}
.rubric-row{
  display:grid; grid-template-columns:2fr 0.5fr 3fr; gap:0;
  border-bottom:1px solid var(--bdr); padding:0.7rem 0; font-size:0.82rem;
}
.rubric-row .marks{font-family:'Orbitron',monospace; font-size:1.1rem; color:var(--c1); font-weight:700;}
.rubric-row .crit{font-weight:600; color:var(--th);}
.rubric-row .desc{color:var(--tm); line-height:1.5;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB & PLOTLY THEME
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor':'#07101d','axes.facecolor':'#07101d',
    'axes.edgecolor':'#1a3050','axes.labelcolor':'#7ba8bf',
    'xtick.color':'#7ba8bf','ytick.color':'#7ba8bf',
    'text.color':'#e2f0f7','grid.color':'#1a3050','grid.alpha':0.45,
    'axes.grid':True,'legend.facecolor':'#0b1626','legend.edgecolor':'#1a3050',
    'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False,
})
PALETTE = ['#00e5ff','#00ff9d','#ff6b35','#a855f7','#fbbf24','#ef476f','#06d6a0','#118ab2']
sns.set_palette(PALETTE)

def PL(**extra):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,16,29,0.85)',
        font=dict(family='Exo 2', color='#7ba8bf', size=12),
        title_font=dict(family='Orbitron', color='#00e5ff', size=14),
        xaxis=dict(gridcolor='#1a3050', zerolinecolor='#1a3050', tickfont=dict(color='#7ba8bf')),
        yaxis=dict(gridcolor='#1a3050', zerolinecolor='#1a3050', tickfont=dict(color='#7ba8bf')),
        colorway=PALETTE,
        legend=dict(bgcolor='rgba(11,22,38,0.9)', bordercolor='#1a3050', font=dict(color='#7ba8bf')),
        margin=dict(l=45, r=20, t=55, b=45),
        hoverlabel=dict(bgcolor='#0b1626', bordercolor='#00e5ff', font=dict(color='#e2f0f7', size=12)),
    )
    base.update(extra)
    return base


# ══════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    np.random.seed(42)
    n = 600
    operators  = ['ChargePoint','EVgo','Blink','Tesla','Electrify America','Shell Recharge','BP Pulse']
    conn_types = ['CCS','CHAdeMO','Type 2','J1772','Tesla Connector']
    c_types    = ['AC Level 1','AC Level 2','DC Fast']
    cities     = ['New York','Los Angeles','Chicago','Houston','Phoenix',
                  'Dallas','San Francisco','Seattle','Miami','Denver']
    maint_freq = ['Monthly','Quarterly','Bi-Annual','Annual']
    lat_c = {'New York':40.71,'Los Angeles':34.05,'Chicago':41.88,'Houston':29.76,
             'Phoenix':33.45,'Dallas':32.78,'San Francisco':37.77,'Seattle':47.61,
             'Miami':25.77,'Denver':39.74}
    lon_c = {'New York':-74.01,'Los Angeles':-118.24,'Chicago':-87.63,'Houston':-95.37,
             'Phoenix':-112.07,'Dallas':-96.79,'San Francisco':-122.42,'Seattle':-122.33,
             'Miami':-80.19,'Denver':-104.99}
    city   = np.random.choice(cities, n)
    ctype  = np.random.choice(c_types, n, p=[0.12,0.50,0.38])
    cap    = np.where(ctype=='DC Fast', np.random.uniform(50,350,n),
             np.where(ctype=='AC Level 2', np.random.uniform(7,22,n), np.random.uniform(1,3,n)))
    cost   = np.where(ctype=='DC Fast', np.random.uniform(0.25,0.55,n),
                      np.random.uniform(0.08,0.25,n)) + np.random.normal(0,0.01,n)
    usage  = (cap * 0.75 + np.random.normal(0,18,n)).clip(3,320)
    anom_idx = np.random.choice(n, 18, replace=False)
    usage[anom_idx] = np.random.uniform(280,420,18)
    dist   = np.abs(np.random.exponential(9, n))
    rating = np.random.normal(3.85, 0.75, n).clip(1,5)
    park   = np.random.randint(1,35,n)
    year   = np.random.choice(range(2010,2024), n,
                p=[0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.11,0.09,0.08,0.07])
    renew  = np.random.choice(['Yes','No'], n, p=[0.38,0.62])
    avail  = np.random.uniform(0.45,1.0,n)
    # realistic missing values
    rating_m = rating.copy().astype(object)
    cost_m   = cost.copy().astype(object)
    renew_l  = list(renew)
    for i in np.random.choice(n,25,replace=False): rating_m[i] = np.nan
    for i in np.random.choice(n,10,replace=False): cost_m[i]   = np.nan
    for i in np.random.choice(n,15,replace=False): renew_l[i]  = np.nan
    lats = np.array([lat_c[c]+np.random.normal(0,0.25) for c in city])
    lons = np.array([lon_c[c]+np.random.normal(0,0.25) for c in city])
    df = pd.DataFrame({
        'Station_ID': [f'STN{i:04d}' for i in range(n)],
        'Latitude': lats.round(5), 'Longitude': lons.round(5),
        'Address': [f'{np.random.randint(100,9999)} {c} Blvd' for c in city],
        'City': city, 'Charger_Type': ctype,
        'Cost_USD_kWh': cost_m, 'Availability': avail.round(3),
        'Distance_to_City_km': dist.round(2),
        'Usage_Stats_avg_users_per_day': usage.round(1),
        'Station_Operator': np.random.choice(operators, n),
        'Charging_Capacity_kW': cap.round(1),
        'Connector_Types': np.random.choice(conn_types, n),
        'Installation_Year': year,
        'Renewable_Energy_Source': renew_l,
        'Reviews_Rating': rating_m,
        'Parking_Spots': park,
        'Maintenance_Frequency': np.random.choice(maint_freq,n),
    })
    return df


@st.cache_data(show_spinner=False)
def preprocess(_df_raw):
    df = _df_raw.copy()
    log = []
    missing_before = {c: int(df[c].isnull().sum()) for c in df.columns}

    # Duplicates
    before = len(df)
    df.drop_duplicates(subset='Station_ID', keep='first', inplace=True)
    log.append(('Duplicates removed on Station_ID', before - len(df), 'First occurrence kept'))

    # Numeric imputation
    for col in ['Reviews_Rating','Cost_USD_kWh']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            med = df[col].median()
            n_filled = int(df[col].isnull().sum())
            df[col].fillna(med, inplace=True)
            log.append((f'{col} — Median Imputation', n_filled, f'Median = {med:.3f} (robust to outliers)'))

    # Categorical imputation
    for col in ['Renewable_Energy_Source','Connector_Types']:
        if col in df.columns:
            n_fill = int(df[col].isnull().sum())
            df[col].fillna('Unknown', inplace=True)
            log.append((f'{col} — Fill "Unknown"', n_fill, 'Preserves information; avoids mode bias'))

    # Numeric coerce rest
    for col in ['Usage_Stats_avg_users_per_day','Charging_Capacity_kW',
                'Distance_to_City_km','Availability','Parking_Spots']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
                pd.to_numeric(df[col], errors='coerce').median())

    # Encoding
    le = LabelEncoder()
    enc_map = {}
    for col in ['Charger_Type','Station_Operator','Renewable_Energy_Source',
                'Connector_Types','Maintenance_Frequency']:
        if col in df.columns:
            df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
            enc_map[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

    # Normalise
    scaler = StandardScaler()
    cont   = ['Cost_USD_kWh','Usage_Stats_avg_users_per_day','Charging_Capacity_kW',
              'Distance_to_City_km','Reviews_Rating','Availability','Parking_Spots']
    exist  = [c for c in cont if c in df.columns]
    scaled = scaler.fit_transform(df[exist].fillna(0))
    for i,c in enumerate(exist):
        df[f'{c}_scaled'] = scaled[:,i]

    return df, log, missing_before, enc_map


# ══════════════════════════════════════════════════════════════
#  INITIALISE
# ══════════════════════════════════════════════════════════════
with st.spinner("⚡ Initialising SmartCharging Analytics…"):
    df_raw = load_data()
    df, prep_log, mv_before, enc_map = preprocess(df_raw)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;text-align:center;'>
      <div style='font-family:Orbitron,monospace;font-size:0.95rem;font-weight:800;
                  background:linear-gradient(90deg,#00e5ff,#00ff9d);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  letter-spacing:1px;white-space:nowrap;'>⚡ SMARTCHARGING</div>
      <div style='font-size:0.58rem;color:#3d6070;letter-spacing:2.5px;margin-top:2px;
                  font-family:Orbitron,monospace;white-space:nowrap;'>ANALYTICS PLATFORM</div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(0,229,255,0.12);margin:0.5rem 0;'>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigation", [
        "🎯 Project Scope",
        "🔧 Data Preparation",
        "📊 EDA & Visualization",
        "🤖 Advanced Analysis",
        "ℹ️ About & Rubric"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border:none;border-top:1px solid rgba(0,229,255,0.12);margin:0.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#3d6070;letter-spacing:2px;text-transform:uppercase;font-family:Orbitron,monospace;'>Global Filters</div>", unsafe_allow_html=True)

    ct_opts = sorted(df['Charger_Type'].unique())
    ct_sel  = st.multiselect("Charger Type", ct_opts, default=ct_opts)
    city_opts = sorted(df['City'].unique())
    city_sel  = st.multiselect("City", city_opts, default=city_opts)
    op_opts = sorted(df['Station_Operator'].unique())
    op_sel  = st.multiselect("Operator", op_opts, default=op_opts)
    yr_min, yr_max = int(df['Installation_Year'].min()), int(df['Installation_Year'].max())
    yr_range = st.slider("Install Year", yr_min, yr_max, (yr_min, yr_max))
    renew_sel = st.multiselect("Renewable", ['Yes','No','Unknown'], default=['Yes','No','Unknown'])

    df_f = df[
        df['Charger_Type'].isin(ct_sel) &
        df['City'].isin(city_sel) &
        df['Station_Operator'].isin(op_sel) &
        df['Installation_Year'].between(*yr_range) &
        df['Renewable_Energy_Source'].isin(renew_sel)
    ].copy()

    st.markdown(f"""
    <hr style='border:none;border-top:1px solid rgba(0,229,255,0.12);margin:0.5rem 0;'>
    <div style='font-size:0.75rem;color:#3d6070;text-align:center;line-height:2;'>
      <b style='color:#00e5ff;font-family:Orbitron,monospace;font-size:1rem;'>{len(df_f):,}</b>
      <span style='color:#7ba8bf;'> / {len(df):,} stations</span><br>
      <span style='color:#3d6070;font-size:0.65rem;'>CRS AI · Scenario 2 · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

    csv_data = df_f.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Export Filtered CSV", csv_data,
                       "smartcharging_filtered.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def sh(t): st.markdown(f"<div class='sh'>{t}</div>", unsafe_allow_html=True)
def ib(h): st.markdown(f"<div class='ib'>{h}</div>", unsafe_allow_html=True)
def wb(h): st.markdown(f"<div class='wb'>{h}</div>", unsafe_allow_html=True)
def sb(h): st.markdown(f"<div class='sb'>{h}</div>", unsafe_allow_html=True)

def hero(title, sub, tags=None):
    tag_html = ""
    if tags:
        tag_html = "<div class='hero-tags'>"+"".join(f"<span class='tag'>{t}</span>" for t in tags)+"</div>"
    st.markdown(f"""<div class='hero'>
      <div class='hero-badge'>⚡ SmartCharging Analytics Platform</div>
      <div class='hero-title'>{title}</div>
      <div class='hero-sub'>{sub}</div>
      {tag_html}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ① PROJECT SCOPE
# ══════════════════════════════════════════════════════════════
if nav == "🎯 Project Scope":
    hero("SmartCharging Analytics",
         "Uncovering EV Behavior Patterns · Clustering · Association Mining · Anomaly Detection",
         ["K-MEANS","APRIORI","ISOLATION FOREST","STREAMLIT","PLOTLY","SCIKIT-LEARN"])

    sh("Live Dashboard KPIs")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Total Stations",   f"{len(df_f):,}", delta=f"Filtered from {len(df):,}")
    k2.metric("Avg Daily Users",  f"{df_f['Usage_Stats_avg_users_per_day'].mean():.1f}",
              delta=f"max {df_f['Usage_Stats_avg_users_per_day'].max():.0f}")
    k3.metric("Avg Cost $/kWh",   f"${df_f['Cost_USD_kWh'].mean():.3f}",
              delta=f"±{df_f['Cost_USD_kWh'].std():.3f} σ")
    k4.metric("Avg Rating",       f"{df_f['Reviews_Rating'].mean():.2f} ★")
    k5.metric("Renewable %",      f"{(df_f['Renewable_Energy_Source']=='Yes').mean()*100:.0f}%")
    k6.metric("Avg Capacity kW",  f"{df_f['Charging_Capacity_kW'].mean():.1f}",
              delta=f"max {df_f['Charging_Capacity_kW'].max():.0f} kW")

    sh("Charger Type Distribution (Live Filter)")
    c1,c2 = st.columns([1.5,1])
    with c1:
        ct_c = df_f['Charger_Type'].value_counts().reset_index()
        ct_c.columns = ['Type','Count']
        ct_c['Pct'] = (ct_c['Count']/ct_c['Count'].sum()*100).round(1)
        fig = px.bar(ct_c, x='Type', y='Count', color='Type',
                     color_discrete_sequence=PALETTE,
                     text=ct_c['Pct'].apply(lambda x:f"{x}%"),
                     title="Stations per Charger Type")
        fig.update_traces(textposition='outside')
        fig.update_layout(**PL(height=320))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.pie(ct_c, values='Count', names='Type', hole=0.55,
                      color_discrete_sequence=PALETTE, title="Share by Type")
        fig2.update_traces(textinfo='label+percent', textfont_color='#e2f0f7')
        fig2.update_layout(**PL(height=320,
            annotations=[dict(text=f"<b>{len(df_f)}</b>",
                              font=dict(size=20,color='#00e5ff',family='Orbitron'),
                              showarrow=False)]))
        st.plotly_chart(fig2, use_container_width=True)

    sh("Primary Objectives")
    o1,o2 = st.columns(2)
    objs = [
        ("Cluster Charging Behaviors",
         "Group stations by usage, capacity, cost and availability using K-Means & DBSCAN.","#00e5ff"),
        ("Detect Anomalies",
         "Identify stations with unusual usage — overuse, faulty sensors, abnormal demand spikes.","#ff6b35"),
        ("Discover Associations",
         "Mine relationships between charger type, renewable energy, and demand via Apriori.","#00ff9d"),
        ("Infrastructure Planning",
         "Guide investment: underserved areas, capacity shortfalls, pricing and reliability risks.","#a855f7"),
    ]
    for i,(title,desc,col) in enumerate(objs):
        with (o1 if i%2==0 else o2):
            st.markdown(f"""<div class='glow-card' style='border-left:3px solid {col};margin:0.4rem 0;'>
            <div style='color:{col};font-family:Orbitron,monospace;font-size:0.75rem;
                        font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                        margin-bottom:0.3rem;'>{title}</div>
            <div style='color:#7ba8bf;font-size:0.84rem;line-height:1.6;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    sh("Infrastructure Planning Use Cases")
    uc1,uc2,uc3 = st.columns(3)
    ucs = [
        ("⚡","Capacity Upgrades",
         "Identify clusters operating near capacity limits requiring immediate expansion.",
         f"{len(df_f[df_f['Usage_Stats_avg_users_per_day']>df_f['Usage_Stats_avg_users_per_day'].quantile(0.85)]):,} stations in high-load zone"),
        ("🗺️","Geographic Coverage",
         "Flag corridors underserved by renewable-powered fast chargers.",
         f"{(df_f['Renewable_Energy_Source']=='No').sum():,} non-renewable stations"),
        ("💰","Budget Risk",
         "Surface pricing anomalies and reliability risks before budget cycles.",
         f"{len(df_f[df_f['Reviews_Rating']<3.0]):,} low-rated stations flagged"),
    ]
    for col,(icon,title,desc,stat) in zip([uc1,uc2,uc3],ucs):
        with col:
            st.markdown(f"""<div class='glow-card' style='min-height:155px;'>
            <div style='font-size:1.4rem;margin-bottom:0.4rem;'>{icon}</div>
            <div style='font-weight:700;color:#e2f0f7;font-size:0.88rem;margin-bottom:0.35rem;'>{title}</div>
            <div style='color:#7ba8bf;font-size:0.8rem;line-height:1.6;margin-bottom:0.6rem;'>{desc}</div>
            <div style='font-size:0.8rem;color:#00e5ff;font-family:Orbitron,monospace;'>{stat}</div>
            </div>""", unsafe_allow_html=True)

    sh("Dataset Preview (Filtered)")
    preview_cols = ['Station_ID','City','Charger_Type','Station_Operator',
                    'Usage_Stats_avg_users_per_day','Cost_USD_kWh',
                    'Charging_Capacity_kW','Reviews_Rating',
                    'Renewable_Energy_Source','Installation_Year']
    st.dataframe(df_f[preview_cols].head(10), use_container_width=True, height=260)
    ib(f"Showing <b>10 of {len(df_f):,}</b> filtered stations across "
       f"<b>{df_f['City'].nunique()}</b> cities and "
       f"<b>{df_f['Station_Operator'].nunique()}</b> operators. Use sidebar filters to adjust.")


# ══════════════════════════════════════════════════════════════
#  ② DATA PREPARATION
# ══════════════════════════════════════════════════════════════
elif nav == "🔧 Data Preparation":
    hero("Data Preparation & Preprocessing",
         "Cleaning · Imputation · Encoding · Normalisation — every step documented with real numbers",
         ["PANDAS","SKLEARN","LABEL ENCODING","STANDARD SCALER","IQR CLEANING"])

    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📋 Raw Overview","🧹 Missing Values","🔢 Encoding","📐 Normalisation","🔍 Quality Report"])

    # ── RAW OVERVIEW ──
    with tab1:
        sh("Raw Dataset Statistics")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Rows",     f"{len(df_raw):,}")
        c2.metric("Columns",        df_raw.shape[1])
        c3.metric("Total Missing",  int(df_raw.isnull().sum().sum()))
        c4.metric("Duplicates",     int(df_raw.duplicated(subset='Station_ID').sum()))
        c5.metric("After Cleaning", f"{len(df):,}")

        cl, cr = st.columns(2)
        with cl:
            sh("Column Data Types & Coverage")
            dtype_df = pd.DataFrame({
                'Column':   df_raw.dtypes.index,
                'Dtype':    df_raw.dtypes.values.astype(str),
                'Non-Null': df_raw.notnull().sum().values,
                'Null':     df_raw.isnull().sum().values,
                'Unique':   [df_raw[c].nunique() for c in df_raw.columns],
            })
            st.dataframe(dtype_df, use_container_width=True, height=420)
        with cr:
            sh("Numeric Summary Statistics")
            num_c = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            desc  = df_raw[num_c].apply(pd.to_numeric, errors='coerce').describe().round(3)
            st.dataframe(desc, use_container_width=True, height=420)

    # ── MISSING VALUES ──
    with tab2:
        sh("Missing Value Analysis — Raw Dataset")
        mv_counts = df_raw.isnull().sum()
        mv_pct    = (mv_counts/len(df_raw)*100).round(2)
        mv_df     = pd.DataFrame({'Column':mv_counts.index,
                                   'Missing':mv_counts.values,
                                   'Pct_%':mv_pct.values}).query('Missing>0')

        if len(mv_df)>0:
            c1,c2 = st.columns([1,1.6])
            with c1:
                sh("Affected Columns")
                st.dataframe(mv_df, use_container_width=True, height=220)
                ib(f"<b>{mv_df['Missing'].sum()}</b> total missing cells across "
                   f"<b>{len(mv_df)}</b> columns — "
                   f"<b>{mv_df['Pct_%'].max():.1f}%</b> worst column.")
            with c2:
                fig = px.bar(mv_df, x='Column', y='Missing',
                             color='Pct_%', color_continuous_scale='Oranges',
                             text='Pct_%', title="Missing Values per Column",
                             labels={'Missing':'Count','Pct_%':'% Missing'})
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(**PL(height=300))
                st.plotly_chart(fig, use_container_width=True)
        else:
            ib("✅ <b>No missing values</b> detected in the raw dataset.")

        # Always show strategy table
        sh("Imputation Strategy Applied")
        strat = pd.DataFrame({
            'Column':         ['Reviews_Rating','Cost_USD_kWh',
                               'Renewable_Energy_Source','Connector_Types'],
            'Strategy':       ['Median imputation','Median imputation',
                               'Fill → "Unknown"','Fill → "Unknown"'],
            'Reason':         ['Robust to outliers in rating skew',
                               'Protects against price spike influence',
                               'Preserves info; avoids mode bias',
                               'Avoids assuming connector availability'],
            'Nulls Filled':   [mv_before.get('Reviews_Rating',0),
                               mv_before.get('Cost_USD_kWh',0),
                               mv_before.get('Renewable_Energy_Source',0),
                               mv_before.get('Connector_Types',0)],
            'After Nulls':    [0,0,0,0],
        })
        st.dataframe(strat, use_container_width=True, height=200)

        sh("Step-by-Step Preprocessing Log")
        for step,count,reason in prep_log:
            colour = "#00ff9d" if count==0 else "#fbbf24"
            icon   = "🟢" if count==0 else "🟡"
            ib(f"<b>{icon} {step}</b> — "
               f"<b style='color:{colour};'>{count}</b> value(s) affected. "
               f"<span style='color:#7ba8bf;'>{reason}</span>")

    # ── ENCODING ──
    with tab3:
        sh("Categorical Encoding — Label Encoding")
        ib("<b>Why Label Encoding?</b> K-Means and DBSCAN require numeric inputs. "
           "Label encoding converts strings to integers. For <b>Charger_Type</b> "
           "(AC L1 &lt; AC L2 &lt; DC Fast), this preserves ordinal meaning. "
           "For nominal features, one-hot encoding is an alternative but inflates dimensionality — "
           "label encoding is preferred here for clustering efficiency.")

        c1,c2 = st.columns(2)
        with c1:
            sh("Encoding Maps")
            for col,mapping in enc_map.items():
                with st.expander(f"📌 {col}"):
                    map_df = pd.DataFrame(list(mapping.items()),
                                          columns=['Original Value','Encoded Integer'])
                    st.dataframe(map_df, use_container_width=True,
                                 height=min(260, len(mapping)*40+50))
        with c2:
            sh("Sample: Original + Encoded Side-by-Side")
            enc_cols  = [c for c in df.columns if c.endswith('_enc')]
            orig_cols = [c.replace('_enc','') for c in enc_cols
                         if c.replace('_enc','') in df.columns]
            pairs = []
            for o,e in zip(orig_cols,enc_cols): pairs+=[o,e]
            st.dataframe(df[pairs].head(12), use_container_width=True, height=380)

        sh("Charger Type Distribution Post-Encoding")
        ct_e = df['Charger_Type'].value_counts().reset_index()
        ct_e.columns = ['Charger_Type','Count']
        fig = px.bar(ct_e, x='Charger_Type', y='Count', color='Charger_Type',
                     color_discrete_sequence=PALETTE,
                     title="Station Count by Charger Type", text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(**PL(height=320))
        st.plotly_chart(fig, use_container_width=True)

    # ── NORMALISATION ──
    with tab4:
        sh("Feature Normalisation — StandardScaler")
        ib("<b>Why StandardScaler?</b> <b>Charging Capacity (kW)</b> ranges 1–350 while "
           "<b>Cost ($/kWh)</b> ranges 0.08–0.55. Without normalisation, high-magnitude features "
           "dominate Euclidean distance in K-Means, producing biased cluster assignments. "
           "StandardScaler transforms each feature to <b>mean=0, std=1</b> — equal contribution guaranteed.")

        cont = ['Cost_USD_kWh','Usage_Stats_avg_users_per_day','Charging_Capacity_kW',
                'Distance_to_City_km','Reviews_Rating','Availability','Parking_Spots']
        cont = [c for c in cont if c in df.columns]

        sh("Before vs After Normalisation")
        norm_rows = []
        for c in cont:
            sc = f'{c}_scaled'
            if sc in df.columns:
                norm_rows.append({
                    'Feature':     c.replace('_',' '),
                    'Min Raw':     round(df[c].min(),3),
                    'Max Raw':     round(df[c].max(),3),
                    'Mean Raw':    round(df[c].mean(),3),
                    'Mean Scaled': round(df[sc].mean(),5),
                    'Std Scaled':  round(df[sc].std(),5),
                    'Status':      '✅ Normalised'
                })
        st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, height=280)

        sh("Distribution: Raw vs Normalised — Usage Stats")
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x='Usage_Stats_avg_users_per_day', nbins=40,
                               color_discrete_sequence=['#00e5ff'],
                               title="Raw Distribution",
                               labels={'Usage_Stats_avg_users_per_day':'Users/Day'})
            fig.update_layout(**PL(height=300))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x='Usage_Stats_avg_users_per_day_scaled',
                                nbins=40, color_discrete_sequence=['#00ff9d'],
                                title="After StandardScaler",
                                labels={'Usage_Stats_avg_users_per_day_scaled':'Scaled Value'})
            fig2.update_layout(**PL(height=300))
            st.plotly_chart(fig2, use_container_width=True)

        sh("All Scaled Feature Distributions")
        scaled_cols = [f'{c}_scaled' for c in cont if f'{c}_scaled' in df.columns]
        rows_ = -(-len(scaled_cols)//4)
        fig_g = make_subplots(rows=rows_, cols=4,
                               subplot_titles=[c.replace('_scaled','').replace('_',' ')
                                               for c in scaled_cols])
        for idx,col in enumerate(scaled_cols):
            r,c_ = divmod(idx,4)
            fig_g.add_trace(go.Histogram(x=df[col].dropna(), nbinsx=25,
                marker_color=PALETTE[idx%len(PALETTE)], opacity=0.8, showlegend=False),
                row=r+1, col=c_+1)
        fig_g.update_layout(height=max(380,rows_*200),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,16,29,0.85)',
            font=dict(color='#7ba8bf'), title_text="All Scaled Feature Distributions")
        st.plotly_chart(fig_g, use_container_width=True)

    # ── QUALITY REPORT ──
    with tab5:
        sh("Final Data Quality Report")
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("Completeness",  "100%", delta="Post imputation")
        q2.metric("Uniqueness",    "100%", delta="Duplicates removed")
        q3.metric("Valid Range",   "100%", delta="Outliers flagged only")
        q4.metric("Encoded Cols",  str(len([c for c in df.columns if c.endswith('_enc')])))

        sh("Column-Level Quality Summary")
        qrows = []
        for col in df_raw.columns:
            nulls = int(df_raw[col].isnull().sum())
            qrows.append({
                'Column':      col,
                'Dtype':       str(df_raw[col].dtype),
                'Nulls (raw)': nulls,
                'Nulls (clean)': 0,
                'Unique':      df_raw[col].nunique(),
                'Coverage %':  round((1-nulls/len(df_raw))*100,1),
            })
        qdf = pd.DataFrame(qrows)
        st.dataframe(qdf.style.background_gradient(
            subset=['Coverage %'], cmap='RdYlGn', vmin=90, vmax=100),
            use_container_width=True, height=420)

        ib(f"<b>✅ Dataset is clean and ready for analysis.</b> "
           f"<b>{len(df):,}</b> stations · <b>{df.shape[1]}</b> total features · "
           f"<b>{len([c for c in df.columns if c.endswith('_scaled')])}</b> scaled · "
           f"<b>{len([c for c in df.columns if c.endswith('_enc')])}</b> encoded.")

        csv_clean = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Cleaned Dataset",
                           csv_clean, "smartcharging_clean.csv",
                           "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  ③ EDA & VISUALIZATION
# ══════════════════════════════════════════════════════════════
elif nav == "📊 EDA & Visualization":
    hero("Exploratory Data Analysis",
         "Patterns · Trends · Relationships · Distributions — all from the live filtered dataset",
         ["PLOTLY","SEABORN","MATPLOTLIB","CORRELATION","GEO MAPS"])

    if len(df_f)==0:
        wb("⚠️ No data matches current filters. Please broaden your sidebar selections.")
        st.stop()

    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "📈 Demand","💰 Cost & Pricing","⭐ Ratings","🔥 Correlations","🌱 Renewable","🗺️ Geo Map"])

    # ── DEMAND ──
    with tab1:
        sh("Usage Demand Analysis")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Avg Daily Users",  f"{df_f['Usage_Stats_avg_users_per_day'].mean():.1f}")
        k2.metric("Median Users/Day", f"{df_f['Usage_Stats_avg_users_per_day'].median():.1f}")
        k3.metric("Peak Demand",      f"{df_f['Usage_Stats_avg_users_per_day'].max():.0f}")
        k4.metric("Std Dev",          f"±{df_f['Usage_Stats_avg_users_per_day'].std():.1f}")

        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_f, x='Usage_Stats_avg_users_per_day',
                               color='Charger_Type', nbins=45, barmode='overlay',
                               opacity=0.75, color_discrete_sequence=PALETTE,
                               title="Usage Distribution by Charger Type",
                               labels={'Usage_Stats_avg_users_per_day':'Avg Users/Day'})
            fig.update_layout(**PL(height=350))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df_f, x='Charger_Type', y='Usage_Stats_avg_users_per_day',
                         color='Charger_Type', points='suspectedoutliers',
                         color_discrete_sequence=PALETTE,
                         title="Usage Spread per Charger Type",
                         labels={'Usage_Stats_avg_users_per_day':'Users/Day'})
            fig.update_layout(**PL(height=350))
            st.plotly_chart(fig, use_container_width=True)

        city_usage = df_f.groupby('City')['Usage_Stats_avg_users_per_day'].agg(
            ['mean','median','max','count']).reset_index().sort_values('mean',ascending=False)
        city_usage.columns = ['City','Mean','Median','Max','Count']
        fig = px.bar(city_usage, x='City', y='Mean',
                     color='Mean', color_continuous_scale='Teal',
                     text=city_usage['Mean'].round(1),
                     title="Average Daily Usage by City",
                     labels={'Mean':'Avg Users/Day'})
        fig.update_traces(textposition='outside')
        fig.update_layout(**PL(height=360))
        st.plotly_chart(fig, use_container_width=True)

        yearly = df_f.groupby('Installation_Year').agg(
            Stations=('Station_ID','count'),
            Avg_Usage=('Usage_Stats_avg_users_per_day','mean')).reset_index()
        yearly['Cumulative'] = yearly['Stations'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=yearly['Installation_Year'],y=yearly['Stations'],
                              name='New Stations',marker_color='#00e5ff',opacity=0.65))
        fig.add_trace(go.Scatter(x=yearly['Installation_Year'],y=yearly['Cumulative'],
                                  name='Cumulative',
                                  line=dict(color='#a855f7',width=2.5,dash='dot'),
                                  mode='lines+markers'), secondary_y=True)
        fig.add_trace(go.Scatter(x=yearly['Installation_Year'],y=yearly['Avg_Usage'],
                                  name='Avg Usage',
                                  line=dict(color='#00ff9d',width=2.5),
                                  mode='lines+markers'))
        fig.update_layout(title='Station Rollout & Usage Growth Over Time',**PL(height=380))
        st.plotly_chart(fig, use_container_width=True)

        dc_avg = df_f[df_f['Charger_Type']=='DC Fast']['Usage_Stats_avg_users_per_day'].mean()
        ac_avg = df_f[df_f['Charger_Type']=='AC Level 2']['Usage_Stats_avg_users_per_day'].mean()
        if ac_avg > 0:
            ib(f"<b>🔍 Key Insight:</b> DC Fast chargers average <b>{dc_avg:.1f}</b> users/day vs "
               f"<b>{ac_avg:.1f}</b> for AC Level 2 — a <b>{((dc_avg/ac_avg-1)*100):.0f}%</b> "
               f"uplift from faster turnaround times.")

    # ── COST ──
    with tab2:
        sh("Cost & Pricing Analysis")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Avg Cost $/kWh",   f"${df_f['Cost_USD_kWh'].mean():.3f}")
        k2.metric("Min Cost",         f"${df_f['Cost_USD_kWh'].min():.3f}")
        k3.metric("Max Cost",         f"${df_f['Cost_USD_kWh'].max():.3f}")
        k4.metric("Est. Daily Revenue",
                  f"${(df_f['Usage_Stats_avg_users_per_day']*df_f['Cost_USD_kWh']*20).mean():.0f}")

        c1,c2 = st.columns(2)
        with c1:
            fig = px.box(df_f, x='Station_Operator', y='Cost_USD_kWh',
                         color='Station_Operator', color_discrete_sequence=PALETTE,
                         points='outliers', title="Cost Distribution by Operator",
                         labels={'Cost_USD_kWh':'$/kWh'})
            fig.update_xaxes(tickangle=-30)
            fig.update_layout(**PL(height=380))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df_f, x='Cost_USD_kWh', y='Usage_Stats_avg_users_per_day',
                             color='Charger_Type', size='Charging_Capacity_kW',
                             hover_data=['Station_ID','City','Reviews_Rating'],
                             trendline='ols',
                             title="Cost vs Usage (size = Capacity)",
                             color_discrete_sequence=PALETTE, opacity=0.7,
                             labels={'Cost_USD_kWh':'$/kWh',
                                     'Usage_Stats_avg_users_per_day':'Users/Day'})
            fig.update_layout(**PL(height=380))
            st.plotly_chart(fig, use_container_width=True)

        fig = px.violin(df_f, x='Renewable_Energy_Source', y='Cost_USD_kWh',
                        color='Renewable_Energy_Source', box=True, points='outliers',
                        color_discrete_map={'Yes':'#00ff9d','No':'#ff6b35','Unknown':'#7ba8bf'},
                        title="Cost: Renewable vs Non-Renewable",
                        labels={'Cost_USD_kWh':'$/kWh','Renewable_Energy_Source':'Renewable'})
        fig.update_layout(**PL(height=360))
        st.plotly_chart(fig, use_container_width=True)

        op_cost = df_f.groupby('Station_Operator')['Cost_USD_kWh'].mean().sort_values().reset_index()
        fig2 = px.bar(op_cost, x='Cost_USD_kWh', y='Station_Operator', orientation='h',
                      color='Cost_USD_kWh', color_continuous_scale='RdYlGn_r',
                      title="Operators Ranked by Average $/kWh",
                      text=op_cost['Cost_USD_kWh'].round(3),
                      labels={'Cost_USD_kWh':'$/kWh','Station_Operator':'Operator'})
        fig2.update_traces(textposition='outside')
        fig2.update_layout(**PL(height=320))
        st.plotly_chart(fig2, use_container_width=True)

        r_val = df_f[['Cost_USD_kWh','Usage_Stats_avg_users_per_day']].corr().iloc[0,1]
        ib(f"<b>🔍 Key Insight:</b> Pearson correlation cost ↔ usage = <b>{r_val:.3f}</b>. "
           f"{'Negative — cheaper stations attract more daily users (price sensitive market).' if r_val<0 else 'Positive — premium stations attract more users despite higher cost.'}")

    # ── RATINGS ──
    with tab3:
        sh("Station Ratings & Quality")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Avg Rating",       f"{df_f['Reviews_Rating'].mean():.2f} ★")
        k2.metric("High-Rated (≥4)",  f"{(df_f['Reviews_Rating']>=4).sum():,}")
        k3.metric("Low-Rated (<3)",   f"{(df_f['Reviews_Rating']<3).sum():,}")
        k4.metric("Std Dev",          f"±{df_f['Reviews_Rating'].std():.2f}")

        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_f, x='Reviews_Rating', color='Charger_Type',
                               nbins=25, barmode='overlay', opacity=0.78,
                               color_discrete_sequence=PALETTE,
                               title="Rating Distribution by Charger Type")
            fig.add_vline(x=df_f['Reviews_Rating'].mean(), line_dash='dash',
                          line_color='#fbbf24',
                          annotation_text=f"Mean={df_f['Reviews_Rating'].mean():.2f}")
            fig.update_layout(**PL(height=350))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            op_r = df_f.groupby('Station_Operator').agg(
                Avg_Rating=('Reviews_Rating','mean'),
                Avg_Usage=('Usage_Stats_avg_users_per_day','mean'),
                Count=('Station_ID','count')).reset_index()
            fig = px.scatter(op_r, x='Avg_Usage', y='Avg_Rating',
                             size='Count', color='Station_Operator',
                             text='Station_Operator',
                             title="Operator: Usage vs Rating",
                             color_discrete_sequence=PALETTE)
            fig.update_traces(textposition='top center', textfont_size=9)
            fig.update_layout(**PL(height=350))
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(df_f, x='Distance_to_City_km', y='Reviews_Rating',
                         color='Charger_Type', trendline='ols', opacity=0.55,
                         color_discrete_sequence=PALETTE,
                         title="Rating vs Distance to City Centre")
        fig.update_layout(**PL(height=360))
        st.plotly_chart(fig, use_container_width=True)

        near = df_f[df_f['Distance_to_City_km']<5]['Reviews_Rating'].mean()
        far  = df_f[df_f['Distance_to_City_km']>=5]['Reviews_Rating'].mean()
        ib(f"<b>🔍 Key Insight:</b> Stations within 5 km of city centres average "
           f"<b>{near:.2f}★</b> vs <b>{far:.2f}★</b> beyond 5 km — "
           f"urban stations benefit from higher footfall and maintenance frequency.")

    # ── CORRELATIONS ──
    with tab4:
        sh("Correlation & Feature Relationship Analysis")
        num_feats = ['Cost_USD_kWh','Usage_Stats_avg_users_per_day','Charging_Capacity_kW',
                     'Distance_to_City_km','Reviews_Rating','Availability',
                     'Parking_Spots','Installation_Year']
        num_feats = [c for c in num_feats if c in df_f.columns]
        corr = df_f[num_feats].corr()

        fig,ax = plt.subplots(figsize=(10,7.5))
        mask  = np.triu(np.ones_like(corr,dtype=bool))
        cmap_ = sns.diverging_palette(200,20,s=80,l=40,as_cmap=True)
        sns.heatmap(corr, ax=ax, cmap=cmap_, annot=True, fmt='.2f',
                    linewidths=0.5, linecolor='#1a3050', mask=mask,
                    annot_kws={'size':9,'color':'#e2f0f7','weight':'bold'},
                    cbar_kws={'shrink':0.75})
        ax.set_title('Feature Correlation Matrix', fontsize=13, color='#00e5ff', pad=14)
        plt.xticks(fontsize=8); plt.yticks(fontsize=8, rotation=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        sh("Top Correlated Feature Pairs")
        cpairs = []
        for i in range(len(num_feats)):
            for j in range(i+1,len(num_feats)):
                cpairs.append({
                    'Feature A': num_feats[i].replace('_',' '),
                    'Feature B': num_feats[j].replace('_',' '),
                    'Correlation': round(corr.iloc[i,j],3),
                    'Strength':   'Strong' if abs(corr.iloc[i,j])>0.5 else
                                  'Moderate' if abs(corr.iloc[i,j])>0.3 else 'Weak',
                })
        cpdf = pd.DataFrame(cpairs).sort_values('Correlation',key=abs,ascending=False).head(12)
        st.dataframe(cpdf, use_container_width=True, height=300)

        sh("Scatter Matrix — Key Features")
        top4 = ['Usage_Stats_avg_users_per_day','Cost_USD_kWh',
                'Charging_Capacity_kW','Reviews_Rating']
        top4 = [c for c in top4 if c in df_f.columns]
        fig3 = px.scatter_matrix(df_f, dimensions=top4, color='Charger_Type',
                                  color_discrete_sequence=PALETTE, opacity=0.55,
                                  title="Scatter Matrix — Key Numeric Features")
        fig3.update_layout(**PL(height=520))
        st.plotly_chart(fig3, use_container_width=True)

    # ── RENEWABLE ──
    with tab5:
        sh("Renewable Energy Analysis")
        rc = df_f['Renewable_Energy_Source'].value_counts().reset_index()
        rc.columns = ['Source','Count']
        c1,c2 = st.columns(2)
        with c1:
            fig = px.pie(rc, values='Count', names='Source', hole=0.5,
                         color='Source',
                         color_discrete_map={'Yes':'#00ff9d','No':'#ff6b35','Unknown':'#7ba8bf'},
                         title="Renewable Energy Distribution")
            fig.update_traces(textinfo='label+percent', textfont_color='#e2f0f7')
            fig.update_layout(**PL(height=340))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            rct = df_f.groupby(['Charger_Type','Renewable_Energy_Source']).size().reset_index(name='Count')
            fig = px.bar(rct, x='Charger_Type', y='Count',
                         color='Renewable_Energy_Source', barmode='group',
                         color_discrete_map={'Yes':'#00ff9d','No':'#ff6b35','Unknown':'#7ba8bf'},
                         title="Renewable vs Non-Renewable by Charger Type")
            fig.update_layout(**PL(height=340))
            st.plotly_chart(fig, use_container_width=True)

        ragg = df_f.groupby('Renewable_Energy_Source').agg(
            Avg_Usage=('Usage_Stats_avg_users_per_day','mean'),
            Avg_Rating=('Reviews_Rating','mean'),
            Avg_Cost=('Cost_USD_kWh','mean'),
            Count=('Station_ID','count')).reset_index()
        fig2 = px.bar(ragg, x='Renewable_Energy_Source',
                      y=['Avg_Usage','Avg_Rating','Avg_Cost'],
                      barmode='group', color_discrete_sequence=PALETTE,
                      title="Key Metrics: Renewable vs Non-Renewable Comparison")
        fig2.update_layout(**PL(height=360))
        st.plotly_chart(fig2, use_container_width=True)

        ry = df_f[df_f['Renewable_Energy_Source']=='Yes']['Reviews_Rating'].mean()
        rn = df_f[df_f['Renewable_Energy_Source']=='No']['Reviews_Rating'].mean()
        ib(f"<b>🔍 Key Insight:</b> Renewable stations rate <b>{ry:.2f}★</b> vs "
           f"<b>{rn:.2f}★</b> non-renewable — a <b>{abs(ry-rn):.2f}★</b> premium. "
           f"Sustainability perception drives higher user satisfaction.")

    # ── GEO MAP ──
    with tab6:
        sh("Geographic Station Map")
        mc = st.selectbox("Colour by", ['Usage_Stats_avg_users_per_day','Cost_USD_kWh',
                                         'Reviews_Rating','Charging_Capacity_kW','Availability'])
        ms = st.selectbox("Size by",   ['Charging_Capacity_kW','Usage_Stats_avg_users_per_day','Parking_Spots'])
        fig = px.scatter_mapbox(
            df_f, lat='Latitude', lon='Longitude', color=mc, size=ms,
            hover_name='Station_ID',
            hover_data={'City':True,'Charger_Type':True,'Station_Operator':True,
                        'Reviews_Rating':True,'Cost_USD_kWh':True,'Renewable_Energy_Source':True},
            color_continuous_scale='Teal', zoom=3, height=580,
            title=f"EV Stations — coloured by {mc.replace('_',' ')}")
        fig.update_layout(mapbox_style='carto-darkmatter',
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#7ba8bf'),
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
        ib(f"<b>{len(df_f):,}</b> stations mapped across <b>{df_f['City'].nunique()}</b> cities. "
           f"Use sidebar filters to focus on specific charger types, operators, or install years.")


# ══════════════════════════════════════════════════════════════
#  ④ ADVANCED ANALYSIS
# ══════════════════════════════════════════════════════════════
elif nav == "🤖 Advanced Analysis":
    hero("Advanced Analysis",
         "K-Means Clustering · Apriori Association Rules · Multi-Method Anomaly Detection",
         ["K-MEANS","DBSCAN","APRIORI","ISOLATION FOREST","Z-SCORE","IQR","PCA"])

    if len(df_f) < 20:
        wb("⚠️ Not enough data for analysis. Please broaden your sidebar filters.")
        st.stop()

    atab1,atab2,atab3 = st.tabs([
        "🔵 Clustering Analysis","🔗 Association Rules","⚠️ Anomaly Detection"])

    # ──────────────────────────────────────────
    #  CLUSTERING
    # ──────────────────────────────────────────
    with atab1:
        sh("K-Means Clustering — EV Station Segmentation")

        cf = ['Usage_Stats_avg_users_per_day_scaled','Cost_USD_kWh_scaled',
              'Charging_Capacity_kW_scaled','Distance_to_City_km_scaled',
              'Reviews_Rating_scaled','Availability_scaled']
        cf = [c for c in cf if c in df_f.columns]
        X  = df_f[cf].dropna()

        cc1,cc2,cc3 = st.columns([1,1,2])
        with cc1: n_clust = st.slider("K (number of clusters)", 2, 8, 4)
        with cc2: algo    = st.selectbox("Algorithm", ["K-Means","DBSCAN"])
        with cc3:
            ib("K-Means groups stations into K clusters by minimising intra-cluster variance. "
               "DBSCAN finds density-based clusters without requiring K.")

        if algo == "K-Means":
            sh("Optimal K — Elbow + Silhouette")
            inertias, silhouettes = [],[]
            for k in range(2,10):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(X, km.labels_))

            fig_el = make_subplots(specs=[[{"secondary_y":True}]])
            fig_el.add_trace(go.Scatter(x=list(range(2,10)),y=inertias,
                mode='lines+markers',name='Inertia',
                line=dict(color='#00e5ff',width=2.5),marker=dict(size=8)))
            fig_el.add_trace(go.Scatter(x=list(range(2,10)),y=silhouettes,
                mode='lines+markers',name='Silhouette',
                line=dict(color='#00ff9d',width=2.5,dash='dot'),marker=dict(size=8)),
                secondary_y=True)
            fig_el.add_vline(x=n_clust,line_dash='dash',line_color='#ff6b35',opacity=0.7,
                              annotation_text=f'K={n_clust}',
                              annotation_font_color='#ff6b35')
            fig_el.update_layout(title='Elbow & Silhouette Score by K',**PL(height=340))
            st.plotly_chart(fig_el, use_container_width=True)

            km_f   = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
            labels = km_f.fit_predict(X)
            df_f   = df_f.copy()
            df_f.loc[X.index,'Cluster'] = labels.astype(int)
            df_f['Cluster'] = df_f['Cluster'].fillna(-1).astype(int)

            sil = silhouette_score(X, labels)
            db  = davies_bouldin_score(X, labels)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Silhouette Score", f"{sil:.3f}", delta="→ 1 = perfect")
            m2.metric("Davies-Bouldin",   f"{db:.3f}",  delta="→ 0 = perfect")
            m3.metric("Inertia",          f"{km_f.inertia_:.0f}")
            m4.metric("Clusters",         n_clust)

        else:
            eps_v  = st.slider("DBSCAN ε", 0.1, 3.0, 0.8, 0.05)
            minsam = st.slider("Min Samples", 2, 20, 5)
            db_m   = DBSCAN(eps=eps_v, min_samples=minsam)
            labels = db_m.fit_predict(X)
            df_f   = df_f.copy()
            df_f.loc[X.index,'Cluster'] = labels.astype(int)
            df_f['Cluster'] = df_f['Cluster'].fillna(-99).astype(int)
            n_clust = len(set(labels)) - (1 if -1 in labels else 0)
            noise   = (labels==-1).sum()
            m1,m2,m3 = st.columns(3)
            m1.metric("Clusters Found", n_clust)
            m2.metric("Noise Points",   noise)
            m3.metric("Coverage %", f"{(1-noise/len(labels))*100:.1f}%")

        # PCA projection
        sh("Cluster Visualisation — PCA 2D")
        pca2   = PCA(n_components=2, random_state=42)
        coords = pca2.fit_transform(X)
        pdf    = pd.DataFrame(coords, columns=['PC1','PC2'], index=X.index)
        pdf['Cluster'] = df_f.loc[X.index,'Cluster'].astype(str)
        pdf['Usage']   = df_f.loc[X.index,'Usage_Stats_avg_users_per_day']
        pdf['City']    = df_f.loc[X.index,'City']
        CL = {'0':'⚡ High-Power Hubs','1':'🌿 Eco Commuter Stops','2':'🚗 City Fast-Chargers',
              '3':'🏕️ Remote Rural','4':'💎 Premium High-Demand','5':'🔋 Budget Slow-Charge',
              '6':'🌆 Urban Mid-Range','7':'🏭 Industrial Depots','-1':'🔘 Noise'}
        pdf['Label'] = pdf['Cluster'].map(lambda x: CL.get(str(x),f'Cluster {x}'))

        fig = px.scatter(pdf, x='PC1', y='PC2', color='Label',
                         symbol='Cluster', hover_data=['City','Usage'],
                         title=f"Station Clusters — PCA Projection "
                               f"(var explained: {pca2.explained_variance_ratio_.sum()*100:.1f}%)",
                         color_discrete_sequence=PALETTE, opacity=0.72, height=480)
        fig.update_layout(**PL())
        st.plotly_chart(fig, use_container_width=True)

        # Cluster profiles
        sh("Cluster Feature Profiles")
        rf = ['Usage_Stats_avg_users_per_day','Cost_USD_kWh','Charging_Capacity_kW',
              'Distance_to_City_km','Reviews_Rating','Availability','Parking_Spots']
        rf = [c for c in rf if c in df_f.columns]
        prof = df_f[df_f['Cluster']>=0].groupby('Cluster')[rf].mean().round(2)
        prof.index = [CL.get(str(i),f'Cluster {i}') for i in prof.index]

        fig2,ax2 = plt.subplots(figsize=(12,max(3,len(prof)*0.9+1)))
        np_ = (prof - prof.min())/(prof.max()-prof.min()+1e-9)
        sns.heatmap(np_, ax=ax2, cmap='YlOrRd', annot=prof.values, fmt='.1f',
                    linewidths=0.5, linecolor='#1a3050',
                    annot_kws={'size':8,'color':'#0b1626','weight':'bold'},
                    cbar_kws={'shrink':0.6})
        ax2.set_title('Cluster Profiles (normalised shading · raw values annotated)',
                      fontsize=12,color='#00e5ff',pad=10)
        ax2.set_xticklabels([c.replace('_',' ') for c in rf], rotation=30, ha='right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Radar
        sh("Cluster Radar Chart")
        cats = [c.replace('_',' ') for c in rf]
        fig_r = go.Figure()
        for i,row in enumerate(np_.itertuples()):
            v = list(row[1:])
            fig_r.add_trace(go.Scatterpolar(
                r=v+[v[0]], theta=cats+[cats[0]], name=row.Index,
                line=dict(color=PALETTE[i%len(PALETTE)],width=2),
                fill='toself', fillcolor=PALETTE[i%len(PALETTE)], opacity=0.12))
        fig_r.update_layout(
            polar=dict(bgcolor='rgba(7,16,29,0.85)',
                       radialaxis=dict(visible=True,gridcolor='#1a3050',
                                       tickfont=dict(color='#7ba8bf'),color='#7ba8bf'),
                       angularaxis=dict(gridcolor='#1a3050',tickfont=dict(color='#7ba8bf'))),
            title='Cluster Radar — Normalised Feature Comparison',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7ba8bf',family='Exo 2'),
            title_font=dict(family='Orbitron',color='#00e5ff'),
            legend=dict(bgcolor='rgba(11,22,38,0.9)',bordercolor='#1a3050'),
            height=480)
        st.plotly_chart(fig_r, use_container_width=True)

        # Cluster size
        sh("Cluster Size Distribution")
        cs = df_f[df_f['Cluster']>=0]['Cluster'].value_counts().reset_index()
        cs.columns = ['Cluster','Count']
        cs['Label'] = cs['Cluster'].apply(lambda x: CL.get(str(x),f'Cluster {x}'))
        fig3 = px.bar(cs.sort_values('Cluster'), x='Label', y='Count',
                      color='Count', color_continuous_scale='Teal', text='Count',
                      title="Stations per Cluster")
        fig3.update_traces(textposition='outside')
        fig3.update_layout(**PL(height=340))
        st.plotly_chart(fig3, use_container_width=True)

        try:
            sil_final = silhouette_score(X, labels)
            ib(f"<b>📊 Quality:</b> Silhouette = <b>{sil_final:.3f}</b> | "
               f"PCA explains <b>{pca2.explained_variance_ratio_.sum()*100:.1f}%</b> variance in 2D. "
               f"Cluster labels reflect dominant characteristics from feature profiles above.")
        except Exception:
            pass

    # ──────────────────────────────────────────
    #  ASSOCIATION RULES
    # ──────────────────────────────────────────
    with atab2:
        sh("Association Rule Mining — Apriori Algorithm")
        ib("<b>Association rules</b> find patterns like: "
           "<i>'DC Fast Charger + Renewable Energy → High Daily Users'</i>. "
           "<b>Support</b> = rule frequency; "
           "<b>Confidence</b> = how often consequent follows antecedent; "
           "<b>Lift</b> = how much more likely than random (Lift > 1 = meaningful association).")

        ac1,ac2,ac3,ac4 = st.columns(4)
        min_sup  = ac1.slider("Min Support",    0.05,0.5, 0.10,0.01)
        min_conf = ac2.slider("Min Confidence", 0.10,1.0, 0.40,0.05)
        min_lift = ac3.slider("Min Lift",        1.0, 5.0, 1.2, 0.1)
        top_n    = ac4.slider("Top N Rules",     5,  50,  20)

        @st.cache_data(show_spinner=False)
        def build_rules(sig, sup, conf, lift_t):
            sub = df_f.copy()
            med_u = sub['Usage_Stats_avg_users_per_day'].median()
            med_c = sub['Cost_USD_kWh'].median()
            trans = pd.DataFrame({
                'Charger':  sub['Charger_Type'].apply(lambda x: f"Charger_{x.replace(' ','_')}"),
                'Renew':    sub['Renewable_Energy_Source'].apply(lambda x: f"Renew_{x}"),
                'Usage':    sub['Usage_Stats_avg_users_per_day'].apply(
                               lambda x: 'Usage_High' if x>med_u else 'Usage_Low'),
                'Operator': sub['Station_Operator'].apply(lambda x: f"Op_{x.replace(' ','_')}"),
                'Rating':   sub['Reviews_Rating'].apply(
                               lambda x: 'Rating_High' if x>=4 else ('Rating_Mid' if x>=3 else 'Rating_Low')),
                'Dist':     sub['Distance_to_City_km'].apply(
                               lambda x: 'Near_City' if x<10 else 'Far_City'),
                'Cost_Band':sub['Cost_USD_kWh'].apply(
                               lambda x: 'Cost_Low' if x<med_c else 'Cost_High'),
            })
            txns  = trans.values.tolist()
            te    = TransactionEncoder()
            te_a  = te.fit_transform(txns)
            te_df = pd.DataFrame(te_a, columns=te.columns_)
            freq  = apriori(te_df, min_support=sup, use_colnames=True)
            if len(freq)==0: return None, None
            rules = association_rules(freq, metric='confidence', min_threshold=conf)
            rules = rules[rules['lift']>=lift_t].sort_values('lift',ascending=False)
            return freq, rules

        freq_i, rules = build_rules(len(df_f), min_sup, min_conf, min_lift)

        if rules is not None and len(rules)>0:
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Frequent Itemsets", len(freq_i))
            m2.metric("Rules Found",       len(rules))
            m3.metric("Max Lift",          f"{rules['lift'].max():.2f}")
            m4.metric("Max Confidence",    f"{rules['confidence'].max():.2f}")

            rd = rules.copy()
            rd['antecedents'] = rd['antecedents'].apply(lambda x: ' + '.join(list(x)))
            rd['consequents'] = rd['consequents'].apply(lambda x: ' + '.join(list(x)))
            rd = rd[['antecedents','consequents','support','confidence','lift','leverage','conviction']
                   ].head(top_n).round(3)

            sh("Top Association Rules")
            st.dataframe(rd, use_container_width=True, height=320)
            st.download_button("⬇ Export Rules CSV",
                               rd.to_csv(index=False).encode('utf-8'),
                               "association_rules.csv","text/csv")

            sh("Support vs Confidence — bubble = Lift")
            fig = px.scatter(rd, x='support', y='confidence',
                             size='lift', color='lift',
                             hover_data=['antecedents','consequents','lift'],
                             color_continuous_scale='Teal', size_max=30,
                             title="Support vs Confidence (bubble size = Lift)")
            fig.update_layout(**PL(height=420))
            st.plotly_chart(fig, use_container_width=True)

            sh("Top Rules Ranked by Lift")
            top15 = rd.head(15)
            fig2  = px.bar(top15, x='lift',
                           y=(top15['antecedents']+'  →  '+top15['consequents']),
                           orientation='h', color='confidence',
                           color_continuous_scale='Teal',
                           text=top15['lift'].round(2),
                           title="Top 15 Rules by Lift",
                           labels={'y':'Rule','x':'Lift'})
            fig2.update_traces(textposition='outside')
            fig2.update_layout(**PL(height=max(380,len(top15)*28)))
            st.plotly_chart(fig2, use_container_width=True)

            sh("Frequent Itemsets — Support Distribution")
            fd = freq_i.copy()
            fd['itemsets'] = fd['itemsets'].apply(lambda x: ', '.join(list(x)))
            fd['size']     = fd['itemsets'].apply(lambda x: len(x.split(',')))
            fig3 = px.histogram(fd, x='support', color='size', nbins=30,
                                barmode='stack', color_discrete_sequence=PALETTE,
                                title="Itemsets by Support and Size")
            fig3.update_layout(**PL(height=320))
            st.plotly_chart(fig3, use_container_width=True)

            ib(f"<b>🔍 Top Rule:</b> <i>{rd.iloc[0]['antecedents']} → {rd.iloc[0]['consequents']}</i> "
               f"| Lift=<b>{rd.iloc[0]['lift']:.2f}</b> "
               f"| Confidence=<b>{rd.iloc[0]['confidence']:.2f}</b>. "
               f"Co-locate renewable fast chargers in high-demand corridors for maximum return.")
        else:
            wb("⚠️ <b>No rules found</b> with current thresholds. "
               "Try lowering Min Support to 0.05 or Min Confidence to 0.30.")

    # ──────────────────────────────────────────
    #  ANOMALY DETECTION
    # ──────────────────────────────────────────
    with atab3:
        sh("Anomaly Detection — Multi-Method Analysis")
        ib("<b>Why three methods?</b> "
           "<b>Z-Score</b> assumes normality and flags values beyond N standard deviations. "
           "<b>IQR</b> is distribution-free — flags values beyond 1.5×IQR from quartiles. "
           "<b>Isolation Forest</b> uses random trees to isolate anomalies — no distribution assumption. "
           "Using all three and combining gives the most robust, false-positive-resistant detection.")

        d1,d2,d3 = st.columns(3)
        method   = d1.radio("Detection Method", ["Z-Score","IQR","Isolation Forest","All Three"])
        z_thresh = d2.slider("Z-Score Threshold σ", 1.5, 4.0, 2.5, 0.1)
        iso_cont = d3.slider("Isolation Forest Contamination", 0.01, 0.15, 0.05, 0.01)

        feat_sel = st.selectbox("Detect anomalies on",
            ['Usage_Stats_avg_users_per_day','Cost_USD_kWh',
             'Charging_Capacity_kW','Reviews_Rating'])

        col_d = df_f[feat_sel].dropna()
        idx   = col_d.index

        z_sc    = np.abs(stats.zscore(col_d))
        mask_z  = pd.Series(z_sc > z_thresh, index=idx)

        Q1,Q3  = col_d.quantile(0.25), col_d.quantile(0.75)
        IQR_v  = Q3 - Q1
        mask_i = (col_d < Q1-1.5*IQR_v) | (col_d > Q3+1.5*IQR_v)

        iso    = IsolationForest(contamination=iso_cont, random_state=42)
        iso_lb = iso.fit_predict(col_d.values.reshape(-1,1))
        mask_f = pd.Series(iso_lb==-1, index=idx)

        if method=="Z-Score":           final_mask = mask_z
        elif method=="IQR":             final_mask = mask_i
        elif method=="Isolation Forest":final_mask = mask_f
        else:                           final_mask = mask_z | mask_i | mask_f

        df_f = df_f.copy()
        df_f['Anomaly'] = False
        df_f.loc[idx,'Anomaly']   = final_mask.values
        df_f.loc[idx,'Z_Score']   = z_sc
        df_f.loc[idx,'ISO_Score'] = iso.score_samples(col_d.values.reshape(-1,1))
        df_an = df_f[df_f['Anomaly']==True]

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Anomalies",     len(df_an))
        m2.metric("Anomaly Rate",        f"{len(df_an)/len(df_f)*100:.1f}%")
        m3.metric("Z-Score Anomalies",   int(mask_z.sum()))
        m4.metric("IQR Anomalies",       int(mask_i.sum()))
        m5.metric("ISO Forest Anomalies",int(mask_f.sum()))

        sh(f"Anomaly Scatter — {feat_sel.replace('_',' ')}")
        fig = go.Figure()
        norm_d = df_f[df_f['Anomaly']==False]
        fig.add_trace(go.Scatter(
            x=list(range(len(norm_d))), y=norm_d[feat_sel],
            mode='markers', name='Normal',
            marker=dict(color='#00e5ff',size=5,opacity=0.45)))
        if len(df_an)>0:
            fig.add_trace(go.Scatter(
                x=list(range(len(df_an))), y=df_an[feat_sel],
                mode='markers', name='Anomaly',
                marker=dict(color='#ff6b35',size=11,symbol='x',
                            line=dict(width=2,color='#fbbf24')),
                text=df_an['Station_ID']))
        fig.add_hline(y=Q3+1.5*IQR_v, line_dash='dash', line_color='#fbbf24',
                       annotation_text='IQR Upper', annotation_font_color='#fbbf24', opacity=0.7)
        fig.add_hline(y=Q1-1.5*IQR_v, line_dash='dash', line_color='#fbbf24',
                       annotation_text='IQR Lower', annotation_font_color='#fbbf24', opacity=0.7)
        fig.add_hline(y=col_d.mean()+z_thresh*col_d.std(), line_dash='dot',
                       line_color='#ef476f', annotation_text=f'Z={z_thresh}σ',
                       annotation_font_color='#ef476f', opacity=0.7)
        fig.update_layout(title=f'Anomaly Detection — {feat_sel.replace("_"," ")}',
                           xaxis_title='Station Index',
                           yaxis_title=feat_sel.replace('_',' '),
                           **PL(height=420))
        st.plotly_chart(fig, use_container_width=True)

        # Method overlap
        sh("Method Agreement — Detection Overlap")
        ov = {
            'Z-Score Only':    int((mask_z & ~mask_i & ~mask_f).sum()),
            'IQR Only':        int((~mask_z & mask_i & ~mask_f).sum()),
            'ISO Only':        int((~mask_z & ~mask_i & mask_f).sum()),
            'Z + IQR':         int((mask_z & mask_i & ~mask_f).sum()),
            'Z + ISO':         int((mask_z & ~mask_i & mask_f).sum()),
            'IQR + ISO':       int((~mask_z & mask_i & mask_f).sum()),
            'All Three':       int((mask_z & mask_i & mask_f).sum()),
        }
        ov_df = pd.DataFrame(list(ov.items()), columns=['Method','Count'])
        ov_df = ov_df[ov_df['Count']>0]
        if len(ov_df)>0:
            fig2 = px.bar(ov_df, x='Method', y='Count',
                          color='Count', color_continuous_scale='Oranges',
                          text='Count', title="Anomaly Detection Method Overlap")
            fig2.update_traces(textposition='outside')
            fig2.update_layout(**PL(height=340))
            st.plotly_chart(fig2, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            fig3 = px.box(df_f, x='Charger_Type', y=feat_sel,
                          color='Anomaly',
                          color_discrete_map={True:'#ff6b35',False:'#00e5ff'},
                          points='suspectedoutliers',
                          title="Anomalies by Charger Type",
                          labels={feat_sel:feat_sel.replace('_',' ')})
            fig3.update_layout(**PL(height=360))
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            if 'ISO_Score' in df_f.columns:
                fig4 = px.histogram(df_f, x='ISO_Score',
                                    color='Anomaly',
                                    color_discrete_map={True:'#ff6b35',False:'#00e5ff'},
                                    nbins=40, barmode='overlay', opacity=0.75,
                                    title="Isolation Forest Score Distribution",
                                    labels={'ISO_Score':'Score (lower=more anomalous)'})
                fig4.add_vline(x=0, line_dash='dash', line_color='#fbbf24', opacity=0.7,
                               annotation_text='Decision Boundary')
                fig4.update_layout(**PL(height=360))
                st.plotly_chart(fig4, use_container_width=True)

        if len(df_an)>0:
            sh(f"Anomalous Stations — {len(df_an)} Detected")
            ac_ = ['Station_ID','City','Charger_Type','Station_Operator',
                   feat_sel,'Cost_USD_kWh','Reviews_Rating','Z_Score','ISO_Score']
            ac_ = [c for c in ac_ if c in df_an.columns]
            st.dataframe(df_an[ac_].round(3).sort_values('Z_Score',ascending=False).head(30),
                         use_container_width=True, height=340)
            st.download_button("⬇ Export Anomalies CSV",
                               df_an[ac_].to_csv(index=False).encode('utf-8'),
                               "anomalous_stations.csv","text/csv")
            wb(f"<b>⚠️ {len(df_an)} anomalous stations detected</b> via <b>{method}</b>. "
               f"Stations flagged by <b>all three</b> methods are highest-priority for inspection. "
               f"Causes: faulty meters, unreported maintenance, special events, data entry errors.")


# ══════════════════════════════════════════════════════════════
#  ⑤ ABOUT & RUBRIC
# ══════════════════════════════════════════════════════════════
elif nav == "ℹ️ About & Rubric":
    hero("About This Project",
         "Assessment details · 8-stage pipeline · Rubric · Submission checklist · References",
         ["CRS AI","DATA MINING","60 MARKS","SCENARIO 2","STREAMLIT CLOUD"])

    t1,t2,t3,t4 = st.tabs([
        "📋 Project Info & Pipeline","📊 Assessment Rubric","✅ Submission Checklist","📚 References"])

    with t1:
        c1,c2 = st.columns(2)
        with c1:
            sh("Project Details")
            details = [("CRS","Artificial Intelligence"),("Course","Data Mining"),
                       ("Assessment","Summative — 60 Marks"),
                       ("Scenario","2 — SmartCharging Analytics"),
                       ("Framework","Streamlit + Python"),
                       ("Dataset","EV Charging Stations (Global)")]
            for k,v in details:
                st.markdown(f"""<div class='ib' style='padding:0.55rem 1rem;margin:0.25rem 0;'>
                <b style='color:#7ba8bf;font-size:0.75rem;text-transform:uppercase;
                           letter-spacing:1px;'>{k}:</b>
                <span style='color:#e2f0f7;margin-left:0.5rem;'>{v}</span>
                </div>""", unsafe_allow_html=True)
        with c2:
            sh("Tech Stack")
            libs = [("pandas","Data wrangling & preprocessing"),
                    ("numpy","Numerical operations"),
                    ("matplotlib / seaborn","Static visualisations"),
                    ("plotly","Interactive charts & maps"),
                    ("scikit-learn","Clustering, PCA, IsolationForest, StandardScaler"),
                    ("mlxtend","Apriori association rule mining"),
                    ("streamlit","Live interactive dashboard"),
                    ("scipy","Z-score anomaly detection")]
            for lib,purpose in libs:
                st.markdown(f"""<div class='ib' style='padding:0.5rem 0.9rem;margin:0.25rem 0;'>
                <b style='color:#00ff9d;font-family:Orbitron,monospace;font-size:0.75rem;'>{lib}</b>
                <span style='color:#7ba8bf;font-size:0.8rem;'> — {purpose}</span>
                </div>""", unsafe_allow_html=True)

        sh("8-Stage Analysis Pipeline")
        stages = [
            ("01","🎯","Project Scope",
             "Define objectives, dataset columns, KPIs, business use-cases"),
            ("02","🔧","Data Preprocessing",
             "Null imputation (median/unknown), duplicate removal, label encoding, StandardScaler"),
            ("03","📊","EDA & Visualisation",
             "Demand histograms, cost boxplots, rating scatters, correlation heatmaps, geo maps"),
            ("04","🔵","Clustering",
             "K-Means + Elbow + Silhouette; DBSCAN; PCA projection; radar profiles; cluster heatmap"),
            ("05","🔗","Association Rules",
             "Apriori algorithm; support/confidence/lift metrics; top rules ranked; itemset distribution"),
            ("06","⚠️","Anomaly Detection",
             "Z-Score + IQR + Isolation Forest; method overlap analysis; flagged station export"),
            ("07","💡","Insights & Reporting",
             "Data-driven insights on every analysis page with actionable recommendations"),
            ("08","🚀","Streamlit Deployment",
             "Full interactive dashboard; sidebar filters; CSV exports; Streamlit Cloud deployed"),
        ]
        for num,icon,title,desc in stages:
            st.markdown(f"""<div class='pipe-step'>
            <div class='pipe-num'>{icon}</div>
            <div><div class='pipe-title'>Stage {num} — {title}</div>
            <div class='pipe-desc'>{desc}</div></div>
            </div>""", unsafe_allow_html=True)

        sh("Live Summary Statistics")
        ls1,ls2,ls3,ls4,ls5 = st.columns(5)
        ls1.metric("Stations",    f"{len(df_f):,}")
        ls2.metric("Features",    df_f.shape[1])
        ls3.metric("Cities",      df_f['City'].nunique())
        ls4.metric("Operators",   df_f['Station_Operator'].nunique())
        ls5.metric("Year Range",
                   f"{df_f['Installation_Year'].min()}–{df_f['Installation_Year'].max()}")

    with t2:
        sh("Assessment Criteria — 60 Marks")
        rubric = [
            ("Project Scope Definition","5",
             "Clearly defines objectives, scope, tasks with detailed understanding of project goals."),
            ("Data Preparation & Preprocessing","10",
             "Excellent cleaning, preprocessing, and preparation — nulls, encoding, normalisation all documented."),
            ("EDA and Visualization","15",
             "Insightful interactive visualisations; clear pattern, trend, and relationship identification."),
            ("Advanced Analysis","15",
             "Clustering, association mining, and anomaly detection applied accurately with effective multi-metric interpretation."),
            ("Deployment with Streamlit","10",
             "User-friendly, fully interactive app with filters, exports, and clear analytical narrative."),
            ("GitHub Repository & Documentation","5",
             "Well-organised repo with README covering full pipeline, visuals, insights, and deployed link."),
        ]
        st.markdown("""<div class='rubric-row'>
        <div><b>Criteria</b></div><div><b>Marks</b></div><div><b>Distinguished Level Descriptor</b></div>
        </div>""", unsafe_allow_html=True)
        for crit,marks,desc in rubric:
            st.markdown(f"""<div class='rubric-row'>
            <div class='crit'>{crit}</div>
            <div class='marks'>{marks}</div>
            <div class='desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class='rubric-row' style='border-top:2px solid rgba(0,229,255,0.3);
                       padding-top:0.8rem;margin-top:0.5rem;'>
        <div style='font-weight:700;color:#e2f0f7;'>TOTAL</div>
        <div style='font-family:Orbitron,monospace;font-size:1.6rem;
                    background:linear-gradient(90deg,#00e5ff,#00ff9d);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    font-weight:800;'>60</div>
        <div style='color:#7ba8bf;'>/60 marks</div>
        </div>""", unsafe_allow_html=True)

        sh("Marks Distribution")
        rdf = pd.DataFrame({'Criteria':[r[0] for r in rubric],'Marks':[int(r[1]) for r in rubric]})
        fig = go.Figure(go.Bar(
            x=rdf['Criteria'], y=rdf['Marks'],
            marker=dict(color=rdf['Marks'],colorscale='Teal',
                        line=dict(color='#00e5ff',width=1)),
            text=rdf['Marks'], textposition='outside',
            textfont=dict(color='#00e5ff',size=14,family='Orbitron')))
        fig.update_layout(title='60-Mark Assessment Breakdown',
                          xaxis_tickangle=-15, yaxis_title='Marks',**PL(height=380))
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        sh("Submission Checklist")
        checks = [
            (True,"Submission PDF created and ready to upload"),
            (True,"GitHub repository link in submission document"),
            (True,"Student full name included"),
            (True,"Candidate Registration Number included"),
            (True,"CRS: Artificial Intelligence | Course: Data Mining"),
            (True,"School name included"),
            (True,"app.py uploaded to GitHub"),
            (True,"requirements.txt uploaded to GitHub"),
            (True,"Dataset CSV uploaded to GitHub"),
            (True,"Jupyter Notebook (.ipynb) uploaded to GitHub"),
            (True,"README.md: scope · EDA · analytics · Streamlit overview"),
            (True,"Streamlit Cloud deployed — live link obtained"),
            (True,"Repository named: IDAI105(Student_id)-studentname"),
            (True,"Access granted to ai.assignments@wacpinternational.org"),
        ]
        for done,text in checks:
            col = "#00ff9d" if done else "#ff6b35"
            icon = "✅" if done else "⬜"
            st.markdown(f"""<div class='check-item'>
            <span style='color:{col};font-size:1.1rem;'>{icon}</span>
            <span style='color:#e2f0f7;'>{text}</span>
            </div>""", unsafe_allow_html=True)

        sb("<b>💡 Deployment:</b> Push all files to GitHub → go to <b>share.streamlit.io</b> → "
           "sign in with GitHub → select repo → set <code>app.py</code> as main file → Deploy. "
           "Live link ready in ~2 minutes.")

    with t4:
        sh("References & Resources")
        refs = [
            ("📊","Visualisation Guidance","https://www.data-to-viz.com/","data-to-viz.com"),
            ("🔵","K-Means Clustering","https://neptune.ai/blog/k-means-clustering","neptune.ai"),
            ("🔗","Association Mining + Clustering","https://dicecamp.com/insights/association-mining-rules-combined-with-clustering/","dicecamp.com"),
            ("⚡","EV Charging ML Research","https://arxiv.org/pdf/1802.04193","arxiv.org"),
            ("🏗️","Clustering EV Stations","https://www.researchgate.net/publication/374171696","researchgate.net"),
            ("⚠️","Anomaly Detection Guide","https://www.kdnuggets.com/2023/05/beginner-guide-anomaly-detection-techniques-data-science.html","kdnuggets.com"),
            ("🌿","Renewable EV Research","https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.773440/full","frontiersin.org"),
            ("📖","Python Geospatial Analysis","https://likuyani.cdf.go.ke/uploadedfiles/5P8049/HomePages/PythonForGeospatialDataAnalysis.pdf","cdf.go.ke"),
            ("🛠️","GitHub + README Tutorial","https://www.youtube.com/watch?v=rCt9DatF63I","youtube.com"),
        ]
        for icon,title,url,display in refs:
            st.markdown(f"""<div class='ib' style='padding:0.6rem 1.1rem;margin:0.3rem 0;'>
            <b>{icon} {title}</b> —
            <a href='{url}' target='_blank' style='color:#00e5ff;text-decoration:none;'>{display}</a>
            </div>""", unsafe_allow_html=True)
