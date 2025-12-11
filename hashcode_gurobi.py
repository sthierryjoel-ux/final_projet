import sys
import os
from collections import defaultdict, namedtuple
import time
import math

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    print("Erreur: gurobipy non trouvé. Assure-toi que gurobipy est installé et que la licence est disponible.")
    raise

# --- Data containers ---
Request = namedtuple('Request', ['video', 'endpoint', 'n'])

def parse_input(path):
    """
    Parse Hash Code 2017 input format.
    Returns a dict with:
      V, E, R, C, X (cache capacity)
      sizes: list of video sizes (len V)
      endpoints: list of dicts { 'L_d': int, 'cache_latencies': {c: latency, ...} }
      requests: list of Request(video, endpoint, n)
    """
    with open(path, 'r', encoding='utf-8') as f:
        tokens = iter(f.read().strip().split())
    def nextint():
        return int(next(tokens))

    V = nextint(); E = nextint(); R = nextint(); C = nextint(); X = nextint()
    sizes = [nextint() for _ in range(V)]

    endpoints = []
    for ei in range(E):
        Ld = nextint()
        K = nextint()
        cache_lat = {}
        for _ in range(K):
            c = nextint()
            lc = nextint()
            cache_lat[c] = lc
        endpoints.append({'L_d': Ld, 'cache_latencies': cache_lat})

    requests = []
    for ri in range(R):
        rv = nextint(); re = nextint(); rn = nextint()
        requests.append(Request(video=rv, endpoint=re, n=rn))

    return {'V':V,'E':E,'R':R,'C':C,'X':X,'sizes':sizes,'endpoints':endpoints,'requests':requests}

def build_model(data, time_limit=None, mipgap=5e-3, presolve=1):
    """
    Build and return a gurobi Model for the problem.
    Model structure as explained in the script docstring.
    """
    V = data['V']; C = data['C']; sizes = data['sizes']
    requests = data['requests']; endpoints = data['endpoints']

    # Precompute: for each request r, the set of caches connected to its endpoint
    req_cache_list = []
    latency_saving = []  # for each request and cache, compute saving = L_d - L_cache
    for r in requests:
        e = r.endpoint
        Ld = endpoints[e]['L_d']
        caches = sorted(endpoints[e]['cache_latencies'].keys())
        req_cache_list.append(caches)
        latency_saving.append({c: (Ld - endpoints[e]['cache_latencies'][c]) for c in caches})

    print(f"[trace] Building model: V={V}, C={C}, #requests={len(requests)}, cache_cap={data['X']}")
    t0 = time.time()

    m = gp.Model("videos")
    m.setParam('OutputFlag', 0)  # suppress verbose gurobi; we will print minimal traces
    if time_limit is not None:
        m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', mipgap)
    m.setParam('Presolve', presolve)

    # Variables x[v,c] binary if video v stored on cache c
    # We'll create only x for caches that appear in some endpoint (saves variables)
    # Collect set of caches that actually appear
    caches_used = set()
    for e in endpoints:
        caches_used.update(e['cache_latencies'].keys())
    caches_used = sorted(caches_used)
    cache_index = {c:i for i,c in enumerate(caches_used)}
    # Map global cache id -> local index
    # x variables: create for v in 0..V-1, for c in caches_used
    x = {}
    for v in range(V):
        for c in caches_used:
            x[v,c] = m.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")

    # y variables: for each request r index and each cache in its endpoint
    y = {}
    for ri, r in enumerate(requests):
        v = r.video
        for c in req_cache_list[ri]:
            # It makes sense only if that x[v,c] exists; but we created x for all caches_used
            y[ri, c] = m.addVar(vtype=GRB.BINARY, name=f"y_{ri}_{c}")

    m.update()

    # Capacity constraints: for each cache c
    for c in caches_used:
        m.addConstr(gp.quicksum(sizes[v] * x[v,c] for v in range(V)) <= data['X'], name=f"cap_{c}")

    # For each request r: sum_c y[r,c] <= 1
    for ri, r in enumerate(requests):
        if req_cache_list[ri]:
            m.addConstr(gp.quicksum(y[ri,c] for c in req_cache_list[ri]) <= 1, name=f"req_once_{ri}")
        else:
            # no caches -> nothing to add
            pass

    # Linking constraints y[r,c] <= x[v_r, c]
    for ri, r in enumerate(requests):
        v = r.video
        for c in req_cache_list[ri]:
            m.addConstr(y[ri,c] <= x[v,c], name=f"link_{ri}_{c}")

    # Objective: maximize total saved latency * number of requests
    obj_terms = []
    for ri, r in enumerate(requests):
        nreq = r.n
        for c in req_cache_list[ri]:
            saving = latency_saving[ri][c]
            if saving > 0:
                obj_terms.append(nreq * saving * y[ri,c])
            # if saving <= 0, there's no benefit to cache; linking still enforced but term is zero

    m.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)

    t_build = time.time() - t0
    print(f"[trace] Model built in {t_build:.2f}s. #vars={m.NumVars}, #constr={m.NumConstrs}")

    return m, x, y, caches_used

def write_solution_and_output(model, x_vars, caches_used, V, out_path="videos.out"):
    """
    Read x_vars and write videos.out in the format required:
    First line: N = number of caches used (where some x[v,c]=1)
    Then, for each used cache: "c v_id v_id ..."
    """
    # Extract solution
    used_caches = []
    cache_videos = {}
    for c in caches_used:
        vids = [v for v in range(V) if x_vars.get((v,c)) is not None and x_vars[(v,c)].X > 0.5]
        if vids:
            used_caches.append(c)
            cache_videos[c] = vids

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"{len(used_caches)}\n")
        for c in used_caches:
            vids = cache_videos[c]
            f.write(str(c))
            for v in vids:
                f.write(" " + str(v))
            f.write("\n")
    print(f"[trace] Wrote solution to {out_path} (caches used = {len(used_caches)})")

def main():
    if len(sys.argv) != 2:
        print("Usage: python videos.py path/to/dataset.in")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"Erreur: dataset {path} introuvable.")
        sys.exit(1)

    data = parse_input(path)
    # quick sanity traces
    print("[trace] Parsed dataset:")
    print(f"  Videos: {data['V']}  Endpoints: {data['E']}  Requests: {data['R']}  Caches: {data['C']}  CacheCap: {data['X']}")

    # Build model. Keep TimeLimit=None (let Gurobi run until gap), but ensure MIPGap <= 5e-3
    # If you want a time limit for testing, pass time_limit=300 (seconds)
    model, x, y, caches_used = build_model(data, time_limit=None, mipgap=5e-3, presolve=1)

    # Write model to MPS
    mps_name = "videos.mps"
    model.write(mps_name)
    print(f"[trace] Wrote model to {mps_name}")

    # Optimize
    print("[trace] Starting optimization (MIPGap set to 5e-3). This may take time depending on dataset.")
    t0 = time.time()
    model.setParam('OutputFlag', 1)  # brief solver output visible now
    model.optimize()
    elapsed = time.time() - t0
    print(f"[trace] Optimization done in {elapsed:.1f}s. Status={model.Status}")

    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.INTERRUPTED:
        # write videos.out
        write_solution_and_output(model, x, caches_used, data['V'], out_path="videos.out")
    else:
        print(f"[trace] Solver did not return an acceptable solution. Status={model.Status}")

if __name__ == "__main__":
    main()
