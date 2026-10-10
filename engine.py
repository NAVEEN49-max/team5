"""Adaptive Intelligent Traffic Network - simulation + control engine (pure Python, no dependencies).

Two identical cities run in lockstep on the SAME demand stream:
  * adaptive (AI): max-pressure signals + congestion-aware live routing + emergency green corridor
  * baseline     : fixed-timing signals + static shortest-path routing
Complexity: signal control O(deg) per junction/tick; routing = Dijkstra O(E log V) per re-plan.
"""
import heapq, random
from collections import deque

N, LEN, VF, CAP, SAT = 4, 150.0, 10.0, 22, 0.5      # grid size, edge metres, m/s, storage, veh/s per green lane
FREE = LEN / VF                                       # free-flow seconds per edge
MING, MAXG, FIXED, YEL = 8, 45, 20, 2                 # min/max green, fixed-time green, yellow (s)
NODES = N * N
BOUNDARY = [r * N + c for r in range(N) for c in range(N) if r in (0, N - 1) or c in (0, N - 1)]
ARTERIAL = [n for n in BOUNDARY if n % N in (0, N - 1)]
INF = float("inf")


def xy(n): return divmod(n, N)


class Edge:
    __slots__ = ("u", "v", "axis", "mv", "q", "closed", "credit", "f", "s")

    def __init__(s, u, v):
        s.u, s.v = u, v
        s.axis = 0 if xy(u)[1] == xy(v)[1] else 1      # 0 = north-south phase, 1 = east-west phase
        s.mv, s.q, s.closed, s.credit, s.f, s.s = [], deque(), False, 0.0, 0.0, 0.0

    @property
    def load(s): return len(s.mv) + len(s.q)


class Veh:
    __slots__ = ("id", "path", "i", "born", "amb", "ff")

    def __init__(s, vid, path, t, amb):
        s.id, s.path, s.i, s.born, s.amb, s.ff = vid, path, 1, t, amb, FREE * (len(path) - 1)


class Sim:
    def __init__(s, adaptive):
        s.adaptive = adaptive
        s.E, s.inc, s.out = {}, {n: [] for n in range(NODES)}, {n: [] for n in range(NODES)}
        for u in range(NODES):
            r, c = xy(u)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    e = Edge(u, rr * N + cc)
                    s.E[(u, e.v)] = e; s.out[u].append(e); s.inc[e.v].append(e)
        s.straight = {}
        for (a, u), e in s.E.items():
            (r1, c1), (r2, c2) = xy(a), xy(u)
            rr, cc = 2 * r2 - r1, 2 * c2 - c1
            s.straight[e] = s.E.get((u, rr * N + cc)) if 0 <= rr < N and 0 <= cc < N else None
        s.phase, s.since, s.yel, s.tgt = [0] * NODES, [0] * NODES, [0] * NODES, [0] * NODES
        s.src = {n: deque() for n in BOUNDARY}
        s.t, s.wait, s.thru, s.done, s.ambs, s.amb_times, s.hist = 0, 0, 0, deque(maxlen=200), [], [], deque(maxlen=120)

    # ---------- congestion prediction + routing ----------
    def pred(s, e):                                     # predicted queue ~30 s ahead (fast EMA + trend)
        return max(0.0, e.f + max(0.0, e.f - e.s) * 2.0)

    def route(s, a, b, live):
        if a == b: return [a], 0.0
        dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b: break
            if d > dist[u]: continue
            for e in s.out[u]:
                if e.closed: continue
                nd = d + FREE + (3.0 * s.pred(e) if live else 0.0)
                if nd < dist.get(e.v, INF):
                    dist[e.v], prev[e.v] = nd, u
                    heapq.heappush(pq, (nd, e.v))
        if b not in prev: return None, INF
        p = [b]
        while p[-1] != a: p.append(prev[p[-1]])
        return p[::-1], dist[b]

    # ---------- vehicles ----------
    def spawn(s, vid, o, d, amb=False):
        p, _ = s.route(o, d, s.adaptive)
        if not p or len(p) < 2: return
        v = Veh(vid, p, s.t, amb)
        if amb: s.ambs.append(v)
        (s.src[o].appendleft if amb else s.src[o].append)(v)

    def _finish(s, v):
        s.thru += 1
        s.done.append((s.t - v.born) - v.ff)
        if v.amb:
            s.amb_times.append(s.t - v.born); s.ambs.remove(v)

    # ---------- signals ----------
    def press(s, u, ph):
        p = 0.0
        for e in s.inc[u]:
            if e.axis != ph or e.closed: continue
            dn = s.straight[e]
            p += max(0.0, len(e.q) + 0.3 * len(e.mv) - 0.4 * (len(dn.q) if dn else 0))
        return p

    def force(s):                                       # emergency green corridor (adaptive only)
        f = {}
        for v in s.ambs:
            for j in (v.i, v.i + 1, v.i + 2):
                if 1 <= j < len(v.path) - 1:
                    e = s.E[(v.path[j - 1], v.path[j])]; f[e.v] = e.axis
        return f

    def control(s, u, f):
        if s.yel[u] > 0:
            s.yel[u] -= 1
            if s.yel[u] == 0: s.phase[u], s.since[u] = s.tgt[u], 0
            return
        s.since[u] += 1; cur = s.phase[u]
        if s.adaptive:
            if u in f: want = f[u] != cur
            else:
                pc, po = s.press(u, cur), s.press(u, 1 - cur)
                want = s.since[u] >= MING and (po > pc * 1.3 + 0.5 or (s.since[u] >= MAXG and po > 0))
        else:
            want = s.since[u] >= FIXED
        if want: s.tgt[u], s.yel[u] = 1 - cur, YEL

    # ---------- one second of simulated time ----------
    def step(s):
        s.t += 1
        for n, qq in s.src.items():                     # release vehicles from entry queues
            while qq:
                v = qq[0]; e = s.E[(v.path[0], v.path[1])]
                if e.closed or e.load >= CAP: break
                qq.popleft(); e.mv.append([v, FREE])
            s.wait += len(qq)
        for e in s.E.values():                          # move along edges
            if not e.mv: continue
            keep = []
            for m in e.mv:
                m[1] -= 1
                if m[1] > 0: keep.append(m); continue
                v = m[0]
                if v.path[v.i] == v.path[-1]: s._finish(v); continue
                if s.adaptive:                          # live re-plan at every junction
                    p, _ = s.route(v.path[v.i], v.path[-1], True)
                    if p and len(p) > 1: v.path = v.path[:v.i] + p
                (e.q.appendleft if (v.amb and s.adaptive) else e.q.append)(v)
            e.mv = keep
        f = s.force() if s.adaptive else {}
        for u in range(NODES): s.control(u, f)
        for u in range(NODES):                          # discharge through green approaches
            for e in s.inc[u]:
                if s.yel[u] == 0 and e.axis == s.phase[u]: e.credit = min(e.credit + SAT, 2.0)
                else: e.credit = min(e.credit, 1.0)
                while e.credit >= 1 and e.q:
                    v = e.q[0]; ne = s.E[(u, v.path[v.i + 1])]
                    if ne.closed:                       # next road closed -> re-route from here
                        p, _ = s.route(u, v.path[-1], s.adaptive)
                        if not p: break
                        v.path = v.path[:v.i] + p; ne = s.E[(u, v.path[v.i + 1])]
                        if ne.closed: break
                    if ne.load >= CAP: break            # spill-back protection
                    e.q.popleft(); v.i += 1; ne.mv.append([v, FREE]); e.credit -= 1
        for e in s.E.values():
            qn = len(e.q); e.f += 0.15 * (qn - e.f); e.s += 0.02 * (qn - e.s)
        s.wait += sum(len(e.q) for e in s.E.values())
        if s.t % 5 == 0: s.hist.append(sum(len(e.q) for e in s.E.values()) + sum(len(q) for q in s.src.values()))

    def stats(s):
        d = list(s.done)
        return dict(done=s.thru, avg_delay=round(sum(d) / len(d), 1) if d else 0.0,
                    wait_vh=round(s.wait / 3600, 2), in_net=sum(e.load for e in s.E.values()),
                    queued=sum(len(e.q) for e in s.E.values()) + sum(len(q) for q in s.src.values()),
                    amb=s.amb_times[-1] if s.amb_times else None)

    def view(s):
        es = []
        for e in s.E.values():
            amb = any(x.amb for x in e.q) or any(m[0].amb for m in e.mv)
            es.append(dict(u=e.u, v=e.v, q=len(e.q), occ=round(e.load / CAP, 2), pred=round(min(1.0, s.pred(e) / CAP * 2), 2),
                           closed=e.closed, mv=[round(1 - m[1] / FREE, 2) for m in e.mv], amb=amb))
        return dict(edges=es, sig=[[s.phase[u], 1 if s.yel[u] else 0, s.tgt[u]] for u in range(NODES)],
                    ambpath=[v.path[v.i - 1:] for v in s.ambs], stats=s.stats(), hist=list(s.hist))


class World:
    """Shared demand + scenario engine driving both cities identically."""
    CLOSURES = [(5, 6), (9, 10)]                        # critical central east-west links

    def __init__(s, seed=7, rate=0.2):
        s.seed, s.base = seed, rate
        s.rng = random.Random(seed)
        s.mult = {n: 1.0 for n in BOUNDARY}
        s.a, s.b = Sim(True), Sim(False)
        s.t, s.vid, s.scenario, s.log, s.later = 0, 0, "normal", ["t=0  baseline traffic"], []

    def note(s, m): s.log = (s.log + [f"t={s.t}  {m}"])[-12:]

    def set_closed(s, u, v, flag):
        for sim in (s.a, s.b):
            for k in ((u, v), (v, u)):
                if k in sim.E: sim.E[k].closed = flag
        s.note(f"road {u}↔{v} {'CLOSED' if flag else 'reopened'}")

    def ambulance(s):
        s.vid += 1
        for sim in (s.a, s.b): sim.spawn(s.vid, 12, 3, True)
        s.note("EMERGENCY: ambulance dispatched SW→NE")

    def apply(s, name):
        for (u, v) in s.CLOSURES: s.set_closed(u, v, False) if s.scenario in ("closure", "multiple") else None
        s.mult = {n: 1.0 for n in BOUNDARY}; s.later = []; s.scenario = name
        if name in ("surge", "multiple"):
            for n in ARTERIAL: s.mult[n] = 2.0
            s.note("TRAFFIC SURGE: arterial demand x2")
        if name in ("closure", "multiple"):
            for (u, v) in s.CLOSURES: s.set_closed(u, v, True)
        if name == "emergency": s.ambulance()
        if name == "multiple": s.later.append(s.t + 20)
        if name == "normal": s.note("scenario: normal flow")

    def step(s):
        s.t += 1
        for n in BOUNDARY:
            if s.rng.random() < s.base * s.mult[n]:
                d = s.rng.choice([x for x in BOUNDARY if x != n]); s.vid += 1
                for sim in (s.a, s.b): sim.spawn(s.vid, n, d)
        if s.later and s.t >= s.later[0]: s.later.pop(0); s.ambulance()
        s.a.step(); s.b.step()

    def score(s):
        A, B = s.a.stats(), s.b.stats()
        r_wait = 1 - s.a.wait / max(s.b.wait, 1)
        r_del = 1 - A["avg_delay"] / B["avg_delay"] if B["avg_delay"] > 0 else 0.0
        r_thr = A["done"] / max(B["done"], 1) - 1
        imp = 0.5 * r_wait + 0.3 * r_del + 0.2 * r_thr
        return dict(score=round(50 + 50 * max(-1, min(1, imp))), wait=round(r_wait * 100), delay=round(r_del * 100),
                    thru=round(r_thr * 100))

    def view(s):
        return dict(t=s.t, scenario=s.scenario, log=s.log, score=s.score(), a=s.a.view(), b=s.b.view())
