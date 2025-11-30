#!/usr/bin/env python3
"""
Hash Code style optimization using gurobipy.

Usage:
    python solve_hashcode_mip.py input_file.in output_solution.txt

This script:
 - reads Hash Code input
 - aggregates requests by (video, endpoint)
 - builds a MIP:
     x[v,c] in {0,1} : video v stored in cache c
     y[v,e,c] in {0,1} : requests for (v,e) served by cache c  (only for caches connected to e)
 - capacity constraints per cache
 - link constraints: y[v,e,c] <= x[v,c]
 - per (v,e): sum_c y[v,e,c] <= 1
 - objective: maximize total latency saving = sum_{v,e,c} n_{v,e} * (L_d_e - L_c_e) * y[v,e,c]
 - sets MIPGap <= 5e-3
 - prints simple traces while reading/building
"""
import sys
import time
from collections import defaultdict

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise ImportError("gurobipy is required to run this script. Install Gurobi and gurobipy.") from e

def parse_hashcode_input(path):
    """Parse Hash Code input file. Return a dict with videos, endpoints, caches, requests aggregated."""
    t0 = time.time()
    print(f"[parse] Opening input file: {path}")
    with open(path, 'r') as f:
        line = f.readline().strip()
        while line == "":
            line = f.readline().strip()
        V, E, R, C, X = map(int, line.split())
        print(f"[parse] V={V}, E={E}, R={R}, C={C}, cache_capacity={X}")

        sizes = list(map(int, f.readline().strip().split()))
        assert len(sizes) == V, "Mismatch in number of video sizes"

        # endpoints info
        endpoints = []  # each: dict with 'L_d' and 'caches': {c: latency}
        for e in range(E):
            line = f.readline().strip()
            while line == "":
                line = f.readline().strip()
            L_d, K = map(int, line.split())
            cache_map = {}
            for _ in range(K):
                c_line = f.readline().strip()
                while c_line == "":
                    c_line = f.readline().strip()
                c_id, L_c = map(int, c_line.split())
                cache_map[c_id] = L_c
            endpoints.append({'L_d': L_d, 'caches': cache_map})

        # requests: aggregate by (video, endpoint)
        req_agg = defaultdict(int)  # key (v,e) -> total requests
        for i in range(R):
            line = f.readline().strip()
            while line == "":
                line = f.readline().strip()
            v, e, n = map(int, line.split())
            req_agg[(v, e)] += n
            # optionally print occasional progress for very large R
            if (i+1) % 200000 == 0:
                print(f"[parse] read {i+1} requests...")

    t1 = time.time()
    print(f"[parse] Done parsing. Aggregated request pairs: {len(req_agg)} (read time {t1-t0:.2f}s)")
    return {
        'V': V, 'E': E, 'R': R, 'C': C, 'X': X,
        'sizes': sizes,
        'endpoints': endpoints,
        'req_agg': dict(req_agg)
    }

def build_and_solve(data, mipgap=5e-3, timelimit=None, verbose=True):
    """
    Build the gurobi model and solve it.
    - mipgap: maximum MIPGap allowed
    - timelimit: optional time limit in seconds
    """
    V = data['V']; E = data['E']; C = data['C']; X = data['X']
    sizes = data['sizes']
    endpoints = data['endpoints']
    req_agg = data['req_agg']

    # Precompute candidate y variables: only for (v,e,c) where c connected to e
    cand_y = []  # list of tuples (v,e,c, n_requests, saving_per_request)
    for (v,e), n in req_agg.items():
        ep = endpoints[e]
        Ld = ep['L_d']
        for c, Lc in ep['caches'].items():
            saving = Ld - Lc
            if saving > 0:
                cand_y.append((v, e, c, n, saving))
    if verbose:
        print(f"[model] Candidate y variables (nonzero potential saving): {len(cand_y)}")
    # Make sets to index variables efficiently
    videos_involved = set(v for v,_,_,_,_ in cand_y)
    caches_involved = set(c for _,_,c,_,_ in cand_y)

    # Build model
    t0 = time.time()
    model = gp.Model("hashcode_mip")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('MIPGap', mipgap)
    if timelimit is not None:
        model.setParam('TimeLimit', timelimit)
    # create x[v,c] only for caches that appear (we can also create for all caches if preferred)
    x = {}
    for v in range(V):
        for c in range(C):
            # optionally skip caches that never appear in any endpoint to save vars
            # but simpler: create only for caches_involved
            if c in caches_involved:
                x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
    if verbose:
        print(f"[model] x variables created: {len(x)}")

    # y variables only for candidate tuples
    y = {}
    for v, e, c, n, saving in cand_y:
        y[v, e, c] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}_{e}_{c}")

    if verbose:
        print(f"[model] y variables created: {len(y)}")

    model.update()

    # Capacity constraints: for each cache c: sum_v sizes[v]*x[v,c] <= X
    if verbose:
        print("[model] Adding capacity constraints...")
    for c in caches_involved:
        expr = gp.quicksum(sizes[v] * x[v, c] for v in range(V) if (v, c) in x)
        model.addConstr(expr <= X, name=f"cap_{c}")

    # Link constraints: y[v,e,c] <= x[v,c]
    if verbose:
        print("[model] Adding link constraints (y <= x)...")
    for (v,e,c), var in y.items():
        # If x[v,c] might not exist (video never used elsewhere), ensure its presence:
        if (v,c) not in x:
            # create x[v,c] (rare)
            x[v,c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
        model.addConstr(var <= x[v, c], name=f"link_{v}_{e}_{c}")

    # For each (v,e), at most one cache serves it: sum_c y[v,e,c] <= 1
    if verbose:
        print("[model] Adding per-(v,e) constraints (at most one cache)...")
    # group candidate y by (v,e)
    y_groups = defaultdict(list)
    for (v,e,c) in y.keys():
        y_groups[(v,e)].append(y[v,e,c])
    for (v,e), vars_list in y_groups.items():
        model.addConstr(gp.quicksum(vars_list) <= 1, name=f"serveOnce_{v}_{e}")

    # Objective: maximize total saving = sum_{v,e,c} n_{v,e} * saving * y[v,e,c]
    # Objective: maximize total saving = sum_{v,e,c} n_{v,e} * saving * y[v,e,c]
    if verbose:
        print("[model] Setting objective...")

    obj = gp.quicksum(n * saving * y[v,e,c] for (v,e,c,n,saving) in cand_y)
    model.setObjective(obj, GRB.MAXIMIZE)


    model.update()
    t1 = time.time()
    if verbose:
        print(f"[model] Build finished in {t1 - t0:.2f}s - vars: {model.NumVars}, constrs: {model.NumConstrs}")

    # Solve
    if verbose:
        print("[solve] Starting optimization (MIPGap <= {:.1e})...".format(mipgap))
        solve_start = time.time()
    model.optimize()
    solve_end = time.time()
    if verbose:
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.OPTIMAL_TOL:
            print(f"[solve] Solve finished in {solve_end - solve_start:.2f}s - obj {model.ObjVal:.2f}")
            if model.Status == GRB.OPTIMAL:
                print("[solve] Status: OPTIMAL")
            elif model.Status == GRB.TIME_LIMIT:
                print("[solve] Status: TIME LIMIT")
            else:
                print(f"[solve] Status code: {model.Status}")
        else:
            print(f"[solve] Solve ended with status {model.Status}")

    return model, x, y

def cand_y_for_obj_iter(cand_y):
    """Helper to iterate cand_y (kept for clarity)"""
    for t in cand_y:
        yield t

def write_solution(model, x_vars, output_path):
    """
    Write solution in Hash Code submission format:
    N
    c v0 v1 v2 ...
    Only caches with at least one video are written.
    """
    # find caches used
    cache_to_videos = defaultdict(list)
    for (v,c), var in x_vars.items():
        try:
            val = var.X
        except AttributeError:
            # var may be a gurobi Var; access .X
            val = var.X
        if val > 0.5:
            cache_to_videos[c].append(v)
    # prepare output
    lines = []
    lines.append(str(len(cache_to_videos)))
    for c, vids in cache_to_videos.items():
        lines.append(str(c) + " " + " ".join(map(str, vids)))
    with open(output_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[output] Solution written to {output_path}. Caches used: {len(cache_to_videos)}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python solve_hashcode_mip.py input_file.in output_solution.txt")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    data = parse_hashcode_input(input_file)
    model, x, y = build_and_solve(data, mipgap=5e-3, timelimit=None, verbose=True)
    write_solution(model, x, output_file)

if __name__ == "__main__":
    main()
