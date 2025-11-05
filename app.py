# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
from datetime import datetime, timedelta
import io

# ============================================================
# 📊 Auto-create dataset if not found
# ============================================================
default_csv = "TV_Scheduling_Genetic_Algorithm.csv"

if not os.path.exists(default_csv):
    # Automatically create dataset if CSV doesn't exist
    data = {
        "Program": [
            "morning_news", "tv_series_a", "tv_series_b", "reality_show",
            "music_program", "documentary", "movie_a", "movie_b",
            "live_soccer", "boxing_show"
        ],
        "Rating": [8.0, 7.2, 7.8, 6.9, 6.5, 7.3, 8.5, 8.8, 9.4, 8.1],
        "Duration": [30, 60, 60, 45, 30, 60, 120, 120, 90, 60]
    }
    df_auto = pd.DataFrame(data)
    df_auto.to_csv(default_csv, index=False)
    print(f"✅ Auto-created dataset: {default_csv}")

# ============================================================
# 📥 Load CSV File (Automatic or Uploaded)
# ============================================================
def load_dataframe(uploaded_file):
    if uploaded_file is not None:
        # Read uploaded file
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Successfully loaded file: {uploaded_file.name}")
            return df
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return None
    else:
        # Automatically detect CSV in current folder
        try:
            for f in os.listdir():
                if f.lower().endswith('.csv'):
                    df = pd.read_csv(f)
                    st.sidebar.info(f"Detected CSV file: {f}")
                    return df
        except Exception as e:
            st.sidebar.error(f"Failed to detect CSV file: {e}")
        return None

# ============================================================
# 🧩 Detect important columns (Program, Rating, Duration)
# ============================================================
def infer_columns(df):
    cols = list(df.columns)
    prog_col = None
    for c in cols:
        if c.lower() in ['program', 'title', 'name']:
            prog_col = c; break
    if prog_col is None:
        prog_col = cols[0]

    rating_col = None
    for c in cols:
        if c.lower() in ['rating', 'score', 'preference']:
            rating_col = c; break
    if rating_col is None:
        df['Rating'] = np.random.uniform(6.0, 9.5, len(df))
        rating_col = 'Rating'

    duration_col = None
    for c in cols:
        if c.lower() in ['duration', 'time', 'mins', 'minutes', 'hour', 'hours']:
            duration_col = c; break
    if duration_col is None:
        df['Duration'] = np.random.choice([30, 45, 60, 90, 120], len(df))
        duration_col = 'Duration'

    return df, prog_col, rating_col, duration_col

# ============================================================
# 🧠 Convert DataFrame into program dictionaries
# ============================================================
def programs_from_df(df, prog_col, rating_col, duration_col):
    programs = []
    for i, row in df.iterrows():
        dur = row[duration_col]
        dur_min = 30.0
        try:
            if isinstance(dur, (int, float)):
                dur_min = float(dur)
            else:
                s = str(dur).lower()
                if ':' in s:
                    h, m = s.split(':')
                    dur_min = int(h)*60 + int(m)
                elif 'h' in s:
                    dur_min = float(s.replace('h', '').strip()) * 60
                else:
                    dur_min = float(s)
        except:
            dur_min = 30.0
        programs.append({
            'id': int(i),
            'program': str(row[prog_col]),
            'rating': float(row[rating_col]),
            'duration': dur_min
        })
    return programs

# ============================================================
# ⚙️ Genetic Algorithm Components
# ============================================================
def create_initial_population(ids, pop_size):
    return [random.sample(ids, len(ids)) for _ in range(pop_size)]

def decode_chromosome(chrom, programs, start="08:00", total_minutes=480):
    prog_map = {p['id']: p for p in programs}
    start_dt = datetime.strptime(start, "%H:%M")
    used = 0
    schedule = []
    for pid in chrom:
        p = prog_map[pid]
        if used + p['duration'] <= total_minutes:
            st_time = start_dt + timedelta(minutes=used)
            end_time = st_time + timedelta(minutes=p['duration'])
            schedule.append({
                'Program': p['program'],
                'Rating': p['rating'],
                'Duration_min': p['duration'],
                'Start': st_time.strftime("%H:%M"),
                'End': end_time.strftime("%H:%M")
            })
            used += p['duration']
        else:
            continue
    return schedule, used

def fitness(chrom, programs, total_minutes=480):
    schedule, used = decode_chromosome(chrom, programs, total_minutes=total_minutes)
    total_rating = sum(item['Rating'] for item in schedule)
    utilization = used / total_minutes
    return 0.7 * total_rating + 0.3 * (utilization * 100)

def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(range(len(pop)), k)
    best = max(selected, key=lambda i: fitnesses[i])
    return pop[best]

def ordered_crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = [-1]*len(p1)
    child[a:b+1] = p1[a:b+1]
    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = fill[idx]; idx += 1
    return child

def swap_mutation(chrom, rate):
    ch = chrom.copy()
    for i in range(len(ch)):
        if random.random() < rate:
            j = random.randint(0, len(ch)-1)
            ch[i], ch[j] = ch[j], ch[i]
    return ch

def run_ga(programs, pop_size=100, generations=200, co_r=0.8, mut_r=0.02, elitism=2, total_minutes=480, seed=None):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    ids = [p['id'] for p in programs]
    pop = create_initial_population(ids, pop_size)
    history = []

    for _ in range(generations):
        fits = [fitness(c, programs, total_minutes) for c in pop]
        new_pop = []
        elites = np.argsort(fits)[-elitism:]
        for e in elites:
            new_pop.append(pop[e])
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            if random.random() < co_r:
                child = ordered_crossover(p1, p2)
            else:
                child = p1.copy()
            child = swap_mutation(child, mut_r)
            new_pop.append(child)
        pop = new_pop
        history.append(max(fits))

    final_fits = [fitness(c, programs, total_minutes) for c in pop]
    best_idx = int(np.argmax(final_fits))
    best_chrom = pop[best_idx]
    sched, used = decode_chromosome(best_chrom, programs, total_minutes=total_minutes)
    total_rating = sum(s['Rating'] for s in sched)
    return {
        'fitness': max(final_fits),
        'schedule': sched,
        'used': used,
        'total_rating': total_rating,
        'history': history
    }

# ============================================================
# 🖥️ Streamlit Interface
# ============================================================
st.set_page_config(page_title="TV Scheduling - Genetic Algorithm", layout="wide")
st.title("📺 TV Scheduling - Genetic Algorithm (JIE42903)")

st.sidebar.header("Load Dataset / Settings")
uploaded_file = st.sidebar.file_uploader("Upload TV Program CSV", type=['csv'])
df = load_dataframe(uploaded_file)

if df is None:
    st.warning("⚠️ Please upload a CSV file or let the auto-generated dataset load.")
    st.stop()

df, prog_col, rating_col, duration_col = infer_columns(df)
st.sidebar.markdown(f"**Detected Columns:** Program=`{prog_col}`, Rating=`{rating_col}`, Duration=`{duration_col}`")

st.subheader("1️⃣ Check and edit data (you may modify ratings if needed)")
edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")

csv_data = edited_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Modified CSV", data=csv_data, file_name="Modified_TV_Scheduling.csv")

programs = programs_from_df(edited_df, prog_col, rating_col, duration_col)

st.subheader("2️⃣ Set Parameters for 3 Trials")
cols = st.columns(3)
trials_params = []
defaults = [(0.8,0.02), (0.6,0.03), (0.9,0.01)]
for i, c in enumerate(cols, start=1):
    with c:
        st.markdown(f"### Trial {i}")
        co = st.slider(f"Crossover Rate (Trial {i})", 0.0, 0.95, defaults[i-1][0], step=0.01)
        mu = st.slider(f"Mutation Rate (Trial {i})", 0.01, 0.05, defaults[i-1][1], step=0.001)
        trials_params.append((co, mu))

if st.button("🚀 Run All Trials"):
    st.info("Running Genetic Algorithm... please wait.")
    results = []
    for i, (co, mu) in enumerate(trials_params, start=1):
        random_seed = int((time.time() * 1000) % (2**32 - 1)) + random.randint(0, 999)
        res = run_ga(programs, pop_size=100, generations=200, co_r=co, mut_r=mu, elitism=2, total_minutes=480, seed=random_seed)
        results.append({'trial': i, 'CO_R': co, 'MUT_R': mu, **res})
    st.success("✅ All trials completed successfully!")

    # Center align table text
    st.markdown(
        """
        <style>
        table {
            width: 100%;
        }
        th, td {
            text-align: center !important;
            vertical-align: middle !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    for r in results:
        st.markdown(f"### 🧪 Trial {r['trial']} — CO_R={r['CO_R']}, MUT_R={r['MUT_R']}")
        st.write(f"**Fitness:** {r['fitness']:.3f} | **Total Rating:** {r['total_rating']:.2f} | **Used Time:** {int(r['used'])}/480 min ({r['used']/480*100:.1f}%)")
        df_sched = pd.DataFrame(r['schedule'])
        st.table(df_sched)
        st.line_chart(r['history'])

    st.markdown("## 📊 Summary Comparison")
    summary = pd.DataFrame([{
        'Trial': r['trial'],
        'CO_R': r['CO_R'],
        'MUT_R': r['MUT_R'],
        'Fitness': round(r['fitness'],3),
        'TotalRating': round(r['total_rating'],2),
        'UsedMinutes': int(r['used']),
        'Util(%)': round(r['used']/480*100,1)
    } for r in results])

    # Center align summary
    st.markdown(
    """
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        text-align: center !important;
        vertical-align: middle !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.dataframe(summary, use_container_width=True)

    csv_out = io.BytesIO()
    summary.to_csv(csv_out, index=False)
    st.download_button("📥 Download Summary (CSV)", data=csv_out.getvalue(), file_name="TV_Scheduling_Summary.csv")

# ============================================================
# 🚀 Safe one-time Streamlit browser launch (no repeated tabs)
# ============================================================
import threading
import webbrowser

def open_browser_once():
    time.sleep(1)
    try:
        webbrowser.open_new("http://localhost:8501")
    except:
        pass

if os.environ.get("STREAMLIT_RUN_ONCE") != "true":
    os.environ["STREAMLIT_RUN_ONCE"] = "true"
    threading.Thread(target=open_browser_once).start()
