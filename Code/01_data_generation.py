
import clr
import sys
import os
import pandas as pd
import numpy as np
import time
import traceback
from System import Array, Double


# PATHS


DWSIM_PATH     = r"C:\DWSIM"
FLOWSHEET_PATH = r"C:\DWSIM Simulation Example\DWSIM_Column.dwsim.dwxmz"
OUTPUT_CSV     = r"C:\DWSIM Simulation Example\distillation_dataset.csv"

N_SAMPLES   = 3000
RANDOM_SEED = 42
P_FEED      = 101325.0


# PARAMETER BOUNDS


BOUNDS = {
    "T_feed_K"          : (340,  385),
    "xF_benzene"        : (0.2,  0.8),
    "N_stages"          : (10,   20),
    "feed_stage_frac"   : (0.25, 0.65),
    "reflux_ratio"      : (1.5,  4.0),
    "bottoms_flow_mols" : (30,   70),
}


# STEP 1 - LATIN HYPERCUBE SAMPLING


def latin_hypercube_sample(n_samples, bounds, seed=42):
    np.random.seed(seed)
    keys     = list(bounds.keys())
    n_params = len(keys)
    lhs      = np.zeros((n_samples, n_params))
    for j in range(n_params):
        perm      = np.random.permutation(n_samples)
        lhs[:, j] = (perm + np.random.uniform(size=n_samples)) / n_samples
    samples = []
    for i in range(n_samples):
        point = {}
        for j, key in enumerate(keys):
            lo, hi     = bounds[key]
            point[key] = lo + lhs[i, j] * (hi - lo)
        samples.append(point)
    return samples

def process_sample(s):
    N  = int(round(s["N_stages"]))
    N  = max(10, min(20, N))
    fs = int(round(s["feed_stage_frac"] * N))
    fs = max(2, min(N - 1, fs))
    return {
        "T_feed_K"          : round(s["T_feed_K"], 1),
        "P_feed_Pa"         : P_FEED,
        "xF_benzene"        : round(s["xF_benzene"], 4),
        "N_stages"          : N,
        "feed_stage"        : fs,
        "reflux_ratio"      : round(s["reflux_ratio"], 3),
        "bottoms_flow_mols" : round(s["bottoms_flow_mols"], 1),
    }


# STEP 2 - LOAD DWSIM


def load_dwsim():
    sys.path.append(DWSIM_PATH)
    clr.AddReference("DWSIM.Automation")
    clr.AddReference("DWSIM.Interfaces")
    clr.AddReference("DWSIM.GlobalSettings")
    clr.AddReference("DWSIM.SharedClasses")
    clr.AddReference("DWSIM.Thermodynamics")
    clr.AddReference("DWSIM.UnitOperations")
    from DWSIM.Automation import Automation3
    auto = Automation3()
    print("DWSIM loaded.")
    return auto


# STEP 3 - LOAD FLOWSHEET


def load_flowsheet(auto, path):
    fs = auto.LoadFlowsheet(path)
    print(f"Flowsheet loaded: {path}")
    return fs


# STEP 4 - SET ALL PARAMETERS 


def set_all_params(flowsheet, T, xF, N, fs, RR, B):
    feed    = flowsheet.GetFlowsheetSimulationObject("Feed")
    col     = flowsheet.GetFlowsheetSimulationObject("Distillation Column")

    # FEED 
    feed_obj = feed.GetAsObject()
    comp     = Array[Double]([xF, 1.0 - xF])
    feed_obj.SetOverallComposition(comp)
    feed_obj.SetMolarFlow(100.0)
    feed_obj.SetTemperature(T)
    feed_obj.SetPressure(P_FEED)

    # COLUMN 
    col_obj = col.GetAsObject()

    # Number of stages
    col_obj.SetNumberOfStages(N)

    # Feed stage
    try:
        col_obj.SetStreamFeedStage("Feed", fs)
    except:
        pass

    # Reflux Ratio — confirmed key 'C', attribute SpecValue
    specs = col_obj.Specs
    specs["C"].SpecValue = RR

    # Bottoms Molar Flow — confirmed key 'R', attribute SpecValue
    specs["R"].SpecValue = B


# STEP 5 - READ RESULTS


def read_results(flowsheet):
    col_obj  = flowsheet.GetFlowsheetSimulationObject(
                   "Distillation Column").GetAsObject()
    dist_obj = flowsheet.GetFlowsheetSimulationObject(
                   "Distillate").GetAsObject()
    bot_obj  = flowsheet.GetFlowsheetSimulationObject(
                   "Bottoms").GetAsObject()

    dist_comp = dist_obj.GetOverallComposition()
    bot_comp  = bot_obj.GetOverallComposition()

    xD = float(dist_comp[0])
    xB = float(bot_comp[1])
    QC = abs(float(col_obj.CondenserDuty) / 1000.0)
    QR = abs(float(col_obj.ReboilerDuty)  / 1000.0)

    return xD, xB, QC, QR


# STEP 6 - VERIFY BEFORE FULL RUN


def verify(auto, flowsheet):
    print("\n" + "=" * 65)
    print("  VERIFICATION - SAME xF/T, DIFFERENT RR/B/N")
    print("=" * 65)

    cases = [
        dict(T=365, xF=0.5, N=10, fs=4,  RR=1.5, B=30),
        dict(T=365, xF=0.5, N=15, fs=7,  RR=2.5, B=50),
        dict(T=365, xF=0.5, N=20, fs=10, RR=4.0, B=70),
    ]

    prev_QR = None
    all_ok  = True

    for i, p in enumerate(cases):
        set_all_params(flowsheet,
                       p["T"], p["xF"], p["N"],
                       p["fs"], p["RR"], p["B"])
        auto.CalculateFlowsheet2(flowsheet)
        xD, xB, QC, QR = read_results(flowsheet)

        print(f"\n  Case {i+1}: N={p['N']} RR={p['RR']} B={p['B']}")
        print(f"    xD={xD:.6f}  xB={xB:.6f}  "
              f"QC={QC:.4f}  QR={QR:.4f}")

        if prev_QR is not None:
            if abs(QR - prev_QR) < 0.01:
                print(f"    FAIL: QR not changing!")
                all_ok = False
            else:
                print(f"    OK: QR changed ({prev_QR:.4f} -> {QR:.4f})")

        prev_QR = QR

    print("\n" + "=" * 65)
    if all_ok:
        print("  VERIFICATION PASSED - all params writing correctly")
    else:
        print("  VERIFICATION FAILED")
    print("=" * 65 + "\n")
    return all_ok


# STEP 7 - RUN ONE SIMULATION


def run_simulation(auto, flowsheet, params):
    T  = params["T_feed_K"]
    xF = params["xF_benzene"]
    N  = params["N_stages"]
    fs = params["feed_stage"]
    RR = params["reflux_ratio"]
    B  = params["bottoms_flow_mols"]

    try:
        set_all_params(flowsheet, T, xF, N, fs, RR, B)
        auto.CalculateFlowsheet2(flowsheet)
        xD, xB, QC, QR = read_results(flowsheet)

        if not (0.01 < xD < 0.9999 and 0.01 < xB < 0.9999):
            return None
        if QC <= 0 or QR <= 0:
            return None

        return {
            "T_feed_K"          : T,
            "P_feed_Pa"         : P_FEED,
            "xF_benzene"        : xF,
            "N_stages"          : N,
            "feed_stage"        : fs,
            "reflux_ratio"      : RR,
            "bottoms_flow_mols" : B,
            "xD_benzene"        : round(xD, 6),
            "xB_toluene"        : round(xB, 6),
            "QC_kW"             : round(QC, 4),
            "QR_kW"             : round(QR, 4),
        }

    except Exception:
        return None

# STEP 8 - MAIN


def main():
    print("=" * 65)
    print("  DWSIM Data Generation - FINAL CORRECT VERSION")
    print("  System : Benzene - Toluene")
    print("  Model  : Peng-Robinson (PR)")
    print(f"  Samples: {N_SAMPLES}")
    print("=" * 65)

    print("\n[1] Loading DWSIM...")
    auto = load_dwsim()

    print(f"\n[2] Loading flowsheet...")
    flowsheet = load_flowsheet(auto, FLOWSHEET_PATH)

    print("\n[3] Verifying all parameters write correctly...")
    ok = verify(auto, flowsheet)
    if not ok:
        print("FAILED. Do not proceed.")
        return

    print("[4] Generating LHS samples...")
    raw     = latin_hypercube_sample(N_SAMPLES, BOUNDS, RANDOM_SEED)
    params  = [process_sample(s) for s in raw]
    print(f"Generated {len(params)} parameter sets.")

    print("\n[5] Running simulations...\n")

    results = []
    failed  = 0
    start   = time.time()

    for i, p in enumerate(params):
        print(f"  [{i+1}/{N_SAMPLES}] "
              f"T={p['T_feed_K']} "
              f"xF={p['xF_benzene']} "
              f"N={p['N_stages']} "
              f"fs={p['feed_stage']} "
              f"RR={p['reflux_ratio']:.2f} "
              f"B={p['bottoms_flow_mols']}",
              end=" -> ")

        result = run_simulation(auto, flowsheet, p)

        if result:
            results.append(result)
            print(f"OK  "
                  f"xD={result['xD_benzene']:.4f} "
                  f"xB={result['xB_toluene']:.4f} "
                  f"QC={result['QC_kW']:.2f} "
                  f"QR={result['QR_kW']:.2f}")
        else:
            failed += 1
            print("FAILED")

        if (i + 1) % 100 == 0:
            elapsed   = time.time() - start
            rate      = (i + 1) / elapsed
            remaining = (N_SAMPLES - i - 1) / rate / 60
            pd.DataFrame(results).to_csv(
                OUTPUT_CSV + ".checkpoint.csv", index=False)
            print(f"\n  Checkpoint {i+1}/{N_SAMPLES}: "
                  f"{len(results)} OK | {failed} failed | "
                  f"{elapsed/60:.1f} min | "
                  f"~{remaining:.1f} min left\n")

    elapsed = time.time() - start
    print("\n" + "=" * 65)
    print(f"  COMPLETED in {elapsed/60:.1f} minutes")
    print(f"  Successful : {len(results)}")
    print(f"  Failed     : {failed}")
    print("=" * 65)

    if results:
        df = pd.DataFrame(results)

        # Final correlation check
        print("\nFinal correlation check:")
        for col in ["xF_benzene", "T_feed_K", "reflux_ratio",
                    "bottoms_flow_mols", "N_stages"]:
            c1 = df[col].corr(df["xD_benzene"])
            c2 = df[col].corr(df["QR_kW"])
            print(f"  {col:<25} "
                  f"corr(xD)={c1:+.3f}  "
                  f"corr(QR)={c2:+.3f}")

        rr_corr = abs(df["reflux_ratio"].corr(df["xD_benzene"]))
        b_corr  = abs(df["bottoms_flow_mols"].corr(df["QR_kW"]))

        if rr_corr < 0.05 and b_corr < 0.05:
            print("\nWARNING: RR and B still not affecting outputs!")
            print("Dataset NOT saved.")
        else:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\nDataset saved: {OUTPUT_CSV}")
            print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
            

if __name__ == "__main__":
    main()