
import csv
import random
import pandas as pd
import streamlit as st
import os
import threading
import time
import webbrowser

# ============================================================
# 1️⃣ READ DATASET
# ============================================================

# Function to read the CSV file and convert it into a dictionary
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # skip header row
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings

# Load the dataset (make sure CSV file is in same directory)
file_path = "TV Scheduling - Genetic Algorithm.csv"
ratings = read_csv_to_dict(file_path)

all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))  # 6 AM - 11 PM

# ============================================================
# 2️⃣ DEFINE GENETIC ALGORITHM PARAMETERS
# ============================================================
GEN = 100
POP = 50
EL_S = 2

# Fitness function: total ratings of all programs in a schedule
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot % len(ratings[program])]
    return total_rating

# Crossover operator
def crossover(schedule1, schedule2):
    point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2

# Mutation operator
def mutate(schedule):
    idx = random.randint(0, len(schedule) - 1)
    new_prog = random.choice(all_programs)
    schedule[idx] = new_prog
    return schedule

# ============================================================
# 3️⃣ STREAMLIT UI
# ============================================================

st.set_page_config(page_title="TV Scheduling - Genetic Algorithm", layout="centered")

st.title("📺 TV Scheduling Optimization using Genetic Algorithm")
st.markdown("""
This app optimizes TV program schedules to maximize total audience ratings using a **Genetic Algorithm (GA)**.  
Adjust the parameters below and click **▶️ Run Trials** to compare results across different settings.
""")

# Parameter sliders
co_r = st.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01)
mut_r = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01)

# ============================================================
# 4️⃣ RUN MULTIPLE TRIALS (With Randomness + Reset Population)
# ============================================================
if st.button("▶️ Run Trials"):
    results = []
    trials = [
        (co_r, mut_r),
        (min(0.95, co_r + 0.05), min(0.05, mut_r + 0.01)),
        (max(0.0, co_r - 0.05), max(0.01, mut_r - 0.01))
    ]

    st.markdown("### 🧪 Running Genetic Algorithm Trials...")
    progress = st.progress(0)

    for i, (co, mu) in enumerate(trials, 1):
        random.seed(time.time() + i)  # ensure different results each trial

        def genetic_algorithm_trial(co_rate, mu_rate):
            # Initialize population
            population = []
            for _ in range(POP):
                schedule = all_programs.copy()
                random.shuffle(schedule)
                population.append(schedule)

            # Evolve population
            for _ in range(GEN):
                population.sort(key=lambda s: fitness_function(s), reverse=True)
                new_pop = population[:EL_S]

                while len(new_pop) < POP:
                    p1, p2 = random.sample(population, 2)
                    if random.random() < co_rate:
                        c1, c2 = crossover(p1, p2)
                    else:
                        c1, c2 = p1.copy(), p2.copy()
                    if random.random() < mu_rate:
                        c1 = mutate(c1)
                    if random.random() < mu_rate:
                        c2 = mutate(c2)
                    new_pop.extend([c1, c2])
                population = new_pop

            best = max(population, key=lambda s: fitness_function(s))
            return best, fitness_function(best)

        # Run GA
        best_schedule, total_rating = genetic_algorithm_trial(co, mu)

        # Save results
        results.append({
            "Trial": i,
            "Crossover Rate": co,
            "Mutation Rate": mu,
            "Total Rating": total_rating
        })

        # Display table for this trial
        st.subheader(f"Trial {i} Results")
        df_trial = pd.DataFrame({
            "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
            "Program": best_schedule
        })
        st.table(df_trial)
        st.success(f"✅ Trial {i} Completed — Total Rating: {total_rating:.2f}")
        progress.progress(i / len(trials))

    # Summary
    df_summary = pd.DataFrame(results)
    st.markdown("### 📊 Summary of All Trials")
    st.table(df_summary)

    # Download CSV
    csv = df_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results as CSV",
        csv,
        "GA_TV_Scheduling_Results.csv",
        "text/csv",
        key="download-csv"
    )

# ============================================================
# 5️⃣ SAFE AUTO-OPEN (One time only)
# ============================================================
def open_browser_once():
    time.sleep(2)
    try:
        webbrowser.open_new("http://localhost:8501")
    except:
        pass

if os.environ.get("STREAMLIT_RUN_ONCE") != "true":
    os.environ["STREAMLIT_RUN_ONCE"] = "true"
    threading.Thread(target=open_browser_once).start()

