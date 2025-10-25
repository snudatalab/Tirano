import sys
import csv
import pickle
from collections import defaultdict
import statistics  

def load_tkg_data(filepath):
    relation2timestamps = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')  
        for line in reader:
            if not line:
                continue
            s_str, r_str, o_str, t_str = line
            s, r, o, t = int(s_str), int(r_str), int(o_str), int(t_str)
            

            relation2timestamps[r].append(t)
    
    return relation2timestamps

def compute_alpha_from_mean(relation2timestamps):
    """
    per relation:
      1) time list align
      2) consecutive sequence of Delta T
      3) mu_r computation
      4) alpha_r = 1/mu_r
    """
    relation2alpha = {}
    
    for r, t_list in relation2timestamps.items():
        # 시점 정렬
        t_list.sort()
        
        # consecutive sequence of Delta T
        deltas = []
        for i in range(len(t_list) - 1):
            dt = t_list[i+1] - t_list[i]
            if dt > 0:  
                deltas.append(dt)
        

        if len(deltas) == 0:
            # fallback: alpha=0.0 
            alpha_r = 0.0
        else:
            #mu_r = sum(deltas) / len(deltas)  
            mu_r = statistics.median(deltas)
            if mu_r <= 1e-9:
                alpha_r = 999999999.0  
            else:
                alpha_r = 1.0 / mu_r
        
        relation2alpha[r] = alpha_r
    
    return relation2alpha

def main():
    input_file = './src/dataset/icews14/train.txt'
    
    # 1) data load
    relation2timestamps = load_tkg_data(input_file)
    
    # 2) alpha_r 
    relation2alpha = compute_alpha_from_mean(relation2timestamps)
    
    print("=== Relation -> alpha ===")
    for r in sorted(relation2alpha.keys()):
        print(f"Relation {r}: alpha={relation2alpha[r]:.4f}")

    # 3) relation2alpha (pickle)
    alpha_file = "relation2alpha.pkl"
    with open(alpha_file, 'wb') as fw:
        pickle.dump(relation2alpha, fw)
    print(f"\nrelation2alpha saved to {alpha_file}")

if __name__ == "__main__":
    main()
