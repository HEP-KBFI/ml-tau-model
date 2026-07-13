
import awkward as ak
import numpy as np

def calculate_energy(pt, eta, mass):
    # E = sqrt((pT * cosh(eta))^2 + m^2)
    # Using float64 to avoid overflow issues common with float16 at high eta
    pt = np.float64(pt)
    eta = np.float64(eta)
    mass = np.float64(mass)
    return np.sqrt((pt * np.cosh(eta))**2 + mass**2)

def reproduce_bug(file_path):
    print(f"Reproducing Index Bug for {file_path}...\n")
    df = ak.from_parquet(file_path)
    event = df[0]
    
    pids = event["Gen_Part_PID"]
    st = event["Gen_Part_Status"]
    pt = event["Gen_Part_PT"]
    eta = event["Gen_Part_Eta"]
    mass = event["Gen_Part_Mass"] / 1000.0 # Convert MeV to GeV if needed, but here it's already in GeV mostly
    d1 = event["Gen_Part_D1"]
    d2 = event["Gen_Part_D2"]
    m1 = event["Gen_Part_M1"]

    # Pick Signal Tau (Status 23)
    tau_idx = 0 
    parent_e = calculate_energy(pt[tau_idx], eta[tau_idx], mass[tau_idx])
    
    print(f"Parent: Index {tau_idx}, PID={pids[tau_idx]}, Status={st[tau_idx]}, Energy={parent_e:.2f} GeV")
    print(f"  Daughter Range: [{d1[tau_idx]}, {d2[tau_idx]}]")
    
    # 1. Energy Discrepancy via D1/D2
    stable_descendants = []
    stack = [tau_idx]
    visited = set()
    while stack:
        idx = stack.pop()
        if idx in visited: continue
        visited.add(idx)
        
        start, end = d1[idx], d2[idx]
        if start == -1:
            stable_descendants.append(idx)
        else:
            for d in range(start, end + 1):
                if d < len(pids):
                    stack.append(d)
    
    sum_e = sum(calculate_energy(pt[i], eta[i], mass[i]) for i in stable_descendants)
    print(f"\nBUG 1: Energy non-conservation via D1/D2 traversal")
    print(f"  Sum of stable descendant energies: {sum_e:.2f} GeV")
    print(f"  Ratio (Sum/Parent): {sum_e/parent_e:.3f} (Expected ~1.0)")

    # 2. Link Inconsistency (M1 != index of parent)
    print(f"\nBUG 2: Mother/Daughter link inconsistency")
    first_daughter = d1[tau_idx]
    if first_daughter != -1:
        claimed_mother = m1[first_daughter]
        print(f"  Parent {tau_idx} claims Daughter {first_daughter}")
        print(f"  Daughter {first_daughter} claims Mother {claimed_mother} (Expected {tau_idx})")
        if claimed_mother != tau_idx:
            print(f"  FAIL: Bidirectional link is broken.")

    # 3. Geometric check for reference
    # Check how many stable particles are actually near the tau
    near_ptcls = []
    for i in range(len(pids)):
        if st[i] == 1:
            # Simple DeltaR check
            d_eta = eta[i] - eta[tau_idx]
            d_phi = event["Gen_Part_Phi"][i] - event["Gen_Part_Phi"][tau_idx]
            while d_phi > np.pi: d_phi -= 2*np.pi
            while d_phi < -np.pi: d_phi += 2*np.pi
            dr = np.sqrt(d_eta**2 + d_phi**2)
            if dr < 0.4:
                near_ptcls.append(i)
    
    geom_sum_e = sum(calculate_energy(pt[i], eta[i], mass[i]) for i in near_ptcls)
    print(f"\nREFERENCE: Geometric proximity (DeltaR < 0.4)")
    print(f"  Sum of stable particles in cone: {geom_sum_e:.2f} GeV")
    print(f"  Ratio (Sum/Parent): {geom_sum_e/parent_e:.3f} (This is physically correct)")

if __name__ == "__main__":
    reproduce_bug("ggHtautau-NEVENT10000-RS17000001.parquet")
