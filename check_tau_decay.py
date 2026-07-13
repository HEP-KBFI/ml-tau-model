
import awkward as ak
import numpy as np
import vector

vector.register_awkward()

def print_decay_tree(input_path, event_idx=0):
    print(f"Loading {input_path}...")
    df = ak.from_parquet(input_path)
    event = df[event_idx]
    
    pids = event["Gen_Part_PID"]
    st = event["Gen_Part_Status"]
    pt = event["Gen_Part_PT"]
    eta = event["Gen_Part_Eta"]
    phi = event["Gen_Part_Phi"]
    mass = event["Gen_Part_Mass"]
    d1 = event["Gen_Part_D1"]
    d2 = event["Gen_Part_D2"]
    m1 = event["Gen_Part_M1"]
    m2 = event["Gen_Part_M2"]
    
    tau_indices = np.where((np.abs(pids) == 15) & (st == 23))[0]
    if len(tau_indices) == 0:
        print("No Status 23 Taus found in event.")
        return

    stable_indices = np.where(st == 1)[0]

    for tau_idx in tau_indices:
        print(f"\n{'='*80}")
        # Use float64 for robust energy calculation
        t_pt = float(pt[tau_idx])
        t_eta = float(eta[tau_idx])
        t_phi = float(phi[tau_idx])
        t_mass = float(mass[tau_idx]) / 1000.0
        tau_e = np.sqrt((t_pt * np.cosh(t_eta))**2 + t_mass**2)
        
        print(f"Signal Tau {tau_idx}: PID={pids[tau_idx]}, Status={st[tau_idx]}, E={tau_e:.2f} GeV")
        print(f"{'='*80}")
        
        # Pure index-based traversal to find stable descendants
        stable_descendants = []
        stack = [tau_idx]
        visited = set()
        
        while stack:
            idx = stack.pop()
            if idx in visited: continue
            visited.add(idx)
            
            start = d1[idx]
            end = d2[idx]
            
            if start == -1:
                stable_descendants.append(idx)
            else:
                for d in range(start, end + 1):
                    if d < len(pids):
                        stack.append(d)
        
        vis_e = 0
        total_e = 0
        
        for i in stable_descendants:
            p_pt = float(pt[i])
            p_eta = float(eta[i])
            p_mass = float(mass[i]) / 1000.0
            p_e = np.sqrt((p_pt * np.cosh(p_eta))**2 + p_mass**2)
            
            total_e += p_e
            if abs(pids[i]) not in [12, 14, 16]:
                vis_e += p_e
        
        print(f"Found {len(stable_descendants)} stable descendants via D1/D2 indexing")
        print(f"  Sum All Stable Energy: {total_e:.2f} GeV (Ratio to Parent: {total_e/tau_e:.3f})")
        print(f"  Sum Visible Energy:    {vis_e:.2f} GeV")
        
        print("\nDecay structure (Following D1/D2 links):")
        def walk(idx, indent=0):
            start = d1[idx]
            end = d2[idx]
            mother1 = m1[idx]
            mother2 = m2[idx]
            
            p_pt = float(pt[idx])
            p_eta = float(eta[idx])
            p_mass = float(mass[idx]) / 1000.0
            p_e = np.sqrt((p_pt * np.cosh(p_eta))**2 + p_mass**2)
            
            info = f"{'  ' * indent}Index {idx}: PID={pids[idx]:>5}, Status={st[idx]:>2}, E={p_e:>7.2f}, Mothers=[{mother1}, {mother2}], Daughters=[{start}, {end}]"
            if start == -1:
                info += " (Stable)"
            print(info)
            
            if start != -1:
                for d_idx in range(start, end + 1):
                    if d_idx == idx: continue
                    if d_idx >= len(pids): continue
                    walk(d_idx, indent + 1)

        walk(tau_idx)

if __name__ == "__main__":
    # You can change the event_idx here
    print_decay_tree("ggHtautau-NEVENT10000-RS17000001.parquet", event_idx=0)
