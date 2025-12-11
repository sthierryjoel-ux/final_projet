import sys
import time
from collections import defaultdict

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise ImportError("gurobipy est requis pour exécuter ce script. Installez Gurobi et gurobipy.") from e


def parse_hashcode_input(path):
    """Analyse le fichier d'entrée Hash Code et retourne un dictionnaire contenant vidéos, endpoints, caches et requêtes agrégées."""
    t0 = time.time()
    print(f"[parse] Ouverture du fichier d'entrée : {path}")
    with open(path, 'r') as f:
        line = f.readline().strip()
        while line == "":
            line = f.readline().strip()
        V, E, R, C, X = map(int, line.split())
        print(f"[parse] V={V}, E={E}, R={R}, C={C}, capacite_cache={X}")

        sizes = list(map(int, f.readline().strip().split()))
        assert len(sizes) == V, "Incohérence dans le nombre de tailles de vidéos"

        # Lecture des endpoints
        endpoints = []  # chaque élément : {'L_d': latence_datacenter, 'caches': {cache_id: latence_cache}}
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

        # Aggregation des requêtes par (vidéo, endpoint)
        req_agg = defaultdict(int)
        for i in range(R):
            line = f.readline().strip()
            while line == "":
                line = f.readline().strip()
            v, e, n = map(int, line.split())
            req_agg[(v, e)] += n
            if (i+1) % 200000 == 0:
                print(f"[parse] {i+1} requêtes lues...")

    t1 = time.time()
    print(f"[parse] Analyse terminée. Paires de requêtes agrégées : {len(req_agg)} (temps {t1-t0:.2f}s)")
    return {
        'V': V, 'E': E, 'R': R, 'C': C, 'X': X,
        'sizes': sizes,
        'endpoints': endpoints,
        'req_agg': dict(req_agg)
    }


def build_and_solve(data, mipgap=5e-3, timelimit=None, verbose=True):
    """
    Construit le modèle Gurobi et le résout.
    - mipgap : écart MIP maximum autorisé
    - timelimit : limite de temps optionnelle (en secondes)
    """
    V = data['V']; E = data['E']; C = data['C']; X = data['X']
    sizes = data['sizes']
    endpoints = data['endpoints']
    req_agg = data['req_agg']

    # Pré-calcul des variables y possibles (uniquement si gain de latence positif)
    cand_y = []
    for (v,e), n in req_agg.items():
        ep = endpoints[e]
        Ld = ep['L_d']
        for c, Lc in ep['caches'].items():
            saving = Ld - Lc
            if saving > 0:
                cand_y.append((v, e, c, n, saving))
    if verbose:
        print(f"[model] Variables y candidates (gain potentiel non nul) : {len(cand_y)}")

    videos_involved = set(v for v,_,_,_,_ in cand_y)
    caches_involved = set(c for _,_,c,_,_ in cand_y)

    # Construction du modèle
    t0 = time.time()
    model = gp.Model("hashcode_mip")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('MIPGap', mipgap)
    if timelimit is not None:
        model.setParam('TimeLimit', timelimit)

    # Création des variables x[v,c]
    x = {}
    for v in range(V):
        for c in range(C):
            if c in caches_involved:
                x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
    if verbose:
        print(f"[model] Variables x créées : {len(x)}")

    # Variables y uniquement pour les candidats
    y = {}
    for v, e, c, n, saving in cand_y:
        y[v, e, c] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}_{e}_{c}")
    if verbose:
        print(f"[model] Variables y créées : {len(y)}")

    model.update()

    # Contraintes de capacité des caches
    if verbose:
        print("[model] Ajout des contraintes de capacité...")
    for c in caches_involved:
        expr = gp.quicksum(sizes[v] * x[v, c] for v in range(V) if (v, c) in x)
        model.addConstr(expr <= X, name=f"cap_{c}")

    # Contraintes de liaison : une requête ne peut être servie par un cache que si la vidéo y est stockée
    if verbose:
        print("[model] Ajout des contraintes de liaison (y <= x)...")
    for (v,e,c), var in y.items():
        if (v,c) not in x:
            x[v,c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
        model.addConstr(var <= x[v, c], name=f"link_{v}_{e}_{c}")

    # Pour chaque (v,e), au plus un cache peut servir la requête
    if verbose:
        print("[model] Ajout des contraintes par (v,e) (au plus un cache servant chaque demande)...")
    y_groups = defaultdict(list)
    for (v,e,c) in y.keys():
        y_groups[(v,e)].append(y[v,e,c])
    for (v,e), vars_list in y_groups.items():
        model.addConstr(gp.quicksum(vars_list) <= 1, name=f"serveOnce_{v}_{e}")

    # Objectif : maximiser le gain de latence total
    if verbose:
        print("[model] Définition de l'objectif...")
    obj = gp.quicksum(n * saving * y[v,e,c] for (v,e,c,n,saving) in cand_y)
    model.setObjective(obj, GRB.MAXIMIZE)

    model.update()
    t1 = time.time()
    if verbose:
        print(f"[model] Construction terminée en {t1 - t0:.2f}s - variables : {model.NumVars}, contraintes : {model.NumConstrs}")

    # Lancement de la résolution
    if verbose:
        print(f"[solve] Début de l'optimisation (MIPGap <= {mipgap:.1e})...")
        solve_start = time.time()
    model.optimize()
    solve_end = time.time()

    if verbose:
       if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:

            print(f"[solve] Résolution terminée en {solve_end - solve_start:.2f}s - obj {model.ObjVal:.2f}")
            if model.Status == GRB.OPTIMAL:
                print("[solve] Statut : OPTIMAL")
            elif model.Status == GRB.TIME_LIMIT:
                print("[solve] Statut : LIMITE DE TEMPS")
            else:
                print(f"[solve] Code statut : {model.Status}")
    else:
        print(f"[solve] Résolution terminée avec le statut {model.Status}")

    return model, x, y


def write_solution(model, x_vars, output_path):
    """
    Écrit la solution au format Hash Code :
    N
    c v0 v1 v2 ...
    (N = nombre de caches contenant au moins une vidéo)
    """
    cache_to_videos = defaultdict(list)
    for (v,c), var in x_vars.items():
        val = var.X
        if val > 0.5:
            cache_to_videos[c].append(v)

    lines = []
    lines.append(str(len(cache_to_videos)))
    for c, vids in cache_to_videos.items():
        lines.append(str(c) + " " + " ".join(map(str, vids)))
    with open(output_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[output] Solution écrite dans {output_path}. Caches utilisés : {len(cache_to_videos)}")


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
