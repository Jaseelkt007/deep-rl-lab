"""LunarLander-v3 (discrete) rebuilt in JAX on the Jax2D physics engine, so jit / vmap / scan can include it.

Copied from gymnasium/envs/box2d/lunar_lander.py: world size, terrain generation, the random start push, the main
and side engine impulses (with dispersion), gravity -10, 50 FPS, mass, the 8 observations, the shaped reward,
fuel costs, -100 for a crash or leaving the screen, +100 for coming to rest, and the 1,000-step limit.

Two deliberate differences, both forced by Jax2D (found with drop tests):
  1. Lander shape. gymnasium's lander is a 6-vertex polygon. Jax2D's collision is wrong for more than 4 vertices
     (a 6-vertex lander sinks ~0.5 m into the ground), so we use the 4-vertex trapezoid with the same top and bottom
     edges, at the same total mass.
  2. Legs. gymnasium's legs are separate bodies on spring-loaded revolute joints. In Jax2D a leg that light (68x
     lighter than the lander) on its soft joint cannot carry the lander: the legs sink into the ground. So each leg
     is a contact point at gymnasium's resting leg-tip position, solved with our own impulses. A leg pushes with at
     most LEG_FORCE (gymnasium's 40 N m leg torque over its ~0.95 m lever); a harder landing compresses it, and if it
     compresses more than the body's clearance the body touches the ground: a crash, as in gymnasium.
"Coming to rest" (+100): Box2D lets a body sleep after 0.5 s below 0.01 m/s and 2 deg/s; Jax2D has no sleeping, so we
count it ourselves, with looser thresholds (see SLEEP_V). Scores are therefore close to gymnasium's, not identical. The known-answer check is gymnasium's
own hand-written controller (heuristic below): it should land here about as well as it does in Box2D.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Jax2D calls jnp.cross on 2-D vectors (the scalar z of the 3-D cross product). JAX 0.12 removed 2-D support, so
# restore exactly that case for Jax2D; every other call goes to the original function.
_cross3 = jnp.cross


def _cross(a, b, *args, **kwargs):
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.shape[-1] == 2 and b.shape[-1] == 2 and not args and not kwargs:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return _cross3(a, b, *args, **kwargs)


jnp.cross = _cross

from jax2d.engine import PhysicsEngine, create_empty_sim  # noqa: E402
from jax2d.scene import add_polygon_to_scene  # noqa: E402
from jax2d.sim_state import SimParams, SimState, StaticSimParams  # noqa: E402

FPS, SCALE = 50, 30.0
DT = 1.0 / FPS
GRAVITY = -10.0
MAIN_ENGINE_POWER, SIDE_ENGINE_POWER = 13.0, 0.6
INITIAL_RANDOM = 1000.0
SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY, MAIN_ENGINE_Y, LEG_DOWN = 14, 12, 4, 18
W, H = 600 / SCALE, 400 / SCALE
CHUNKS = 11
HELIPAD_Y = H / 4
MAX_STEPS = 1000

# Lander: 4-vertex trapezoid (clockwise from top-left, as Jax2D wants), total mass = gymnasium's lander + 2 legs.
LANDER_POLY = np.array([(-14, 17), (14, 17), (17, -10), (-17, -10)], float) / SCALE
GYM_MASS = 5.0 * 867.0 / SCALE**2 + 2 * 1.0 * (4 * 16) / SCALE**2  # hexagon area x 5 + two 4x16 legs x 1
LANDER_DENSITY = GYM_MASS / ((28 + 34) / 2 * 27 / SCALE**2)

# Legs: gymnasium's leg tip at rest (joint at its 0.4 rad limit), relative to the lander's body origin.
LEG_TIP = np.array([[0.952, -0.539], [-0.952, -0.539]])  # [right leg = gymnasium legs[0], left leg = legs[1]]
LEG_FORCE = 40.0 / 0.952  # max push of one leg (N)
LEG_FRICTION = float(np.sqrt(0.2 * 0.1))  # Box2D mixes friction as sqrt(leg 0.2 * ground 0.1)
CONTACT_SLOP, BAUMGARTE = 0.005, 0.2
# Box2D sleeps a body after 0.5 s below 0.01 m/s and 2 deg/s. Contact jitter here keeps a landed lander at
# 0.02-0.04 m/s, so we use 0.05 m/s and 5 deg/s (otherwise landed episodes run into the time limit).
SLEEP_V, SLEEP_W, SLEEP_STEPS = 0.05, np.deg2rad(5.0), int(0.5 * FPS)

OBS_DIM, N_ACTIONS = 8, 4
N_TERRAIN = CHUNKS - 1  # polygons 0..9: ground segments (static); polygon 10: the lander
LANDER = N_TERRAIN

STATIC = StaticSimParams(num_polygons=N_TERRAIN + 1, num_circles=2, num_joints=1, num_thrusters=1,  # unused circles/
                         max_polygon_vertices=4, num_static_fixated_polys=N_TERRAIN)  # joint/thruster: Jax2D needs >= 1
PARAMS = SimParams(dt=DT, clip_position=100.0)
ENGINE = PhysicsEngine(STATIC)


def _template() -> tuple[SimState, np.ndarray]:
    """The scene without terrain heights: built once in Python, then reset() only fills in the random parts."""
    sim = create_empty_sim(STATIC, add_floor=False, add_walls_and_ceiling=False)
    sim = sim.replace(gravity=jnp.zeros(2))  # we add gravity ourselves, before the leg contacts are solved
    for _ in range(N_TERRAIN):  # placeholders; reset() overwrites their shape
        sim, _ = add_polygon_to_scene(sim, STATIC, jnp.zeros(2), jnp.array([[0, 1], [1, 1], [1, 0], [0, 0.0]]), 4,
                                      friction=0.1, fixated=True)
    x0, y0 = W / 2, H
    sim, _ = add_polygon_to_scene(sim, STATIC, jnp.array([x0, y0]), jnp.asarray(LANDER_POLY), 4,
                                  density=LANDER_DENSITY, friction=0.1)
    com_offset = np.asarray(sim.polygon.position[LANDER]) - np.array([x0, y0])  # Box2D reports the body origin
    return sim, com_offset


TEMPLATE, COM_OFFSET = _template()
NO_MOTORS = jnp.zeros(STATIC.num_joints + STATIC.num_thrusters)
_pairs = np.asarray(ENGINE.poly_poly_pairs)
PAIR_LANDER = jnp.asarray((_pairs[:, 0] < N_TERRAIN) & (_pairs[:, 1] == LANDER))
CHUNK_X = np.arange(CHUNKS) * W / (CHUNKS - 1)


class EnvState(NamedTuple):
    sim: SimState
    ground_y: jax.Array  # (CHUNKS,) terrain heights at CHUNK_X
    prev_shaping: jax.Array
    t: jax.Array
    still: jax.Array  # consecutive steps below the sleep thresholds


def _rot(a):
    return jnp.array([[jnp.cos(a), -jnp.sin(a)], [jnp.sin(a), jnp.cos(a)]])


def _cross2(r, j):
    return r[0] * j[1] - r[1] * j[0]


def _ground(ground_y, x):
    """Terrain height and upward unit normal at x (piecewise linear between the chunk points)."""
    i = jnp.clip(jnp.floor(x / (W / (CHUNKS - 1))).astype(jnp.int32), 0, CHUNKS - 2)
    xs = jnp.asarray(CHUNK_X)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ground_y[i], ground_y[i + 1]
    slope = (y1 - y0) / (x1 - x0)
    n = jnp.array([-slope, 1.0]) / jnp.sqrt(1.0 + slope**2)
    return y0 + slope * (x - x0), n


def _legs(com, angle, v, w, ground_y):
    """World tip positions, penetration depth along the normal, normals, and tip velocities of both legs."""
    origin = com - _rot(angle) @ COM_OFFSET
    tips = origin + LEG_TIP @ _rot(angle).T  # (2, 2)
    gy, n = jax.vmap(lambda x: _ground(ground_y, x))(tips[:, 0])
    pen = (gy - tips[:, 1]) * n[:, 1]  # > 0: the tip is below the ground
    r = tips - com
    v_tip = v + w * jnp.stack([-r[:, 1], r[:, 0]], -1)
    return tips, pen, n, v_tip


def _leg_impulses(sim: SimState, ground_y, iters: int = 4):
    """Sequential-impulse contact for the two leg tips (Box2D style), each push capped at LEG_FORCE * DT."""
    p = sim.polygon
    m_inv, i_inv = p.inverse_mass[LANDER], p.inverse_inertia[LANDER]
    com, angle = p.position[LANDER], p.rotation[LANDER]

    def one_pass(carry, _):
        v, w, acc_n, acc_t = carry
        for k in range(2):  # 2 legs: unrolled
            tips, pen, n, v_tip = _legs(com, angle, v, w, ground_y)
            r, nk = tips[k] - com, n[k]
            tk = jnp.array([nk[1], -nk[0]])
            vn, vt = v_tip[k] @ nk, v_tip[k] @ tk
            active = pen[k] - vn * DT > 0  # touching now, or will touch during this step
            target = BAUMGARTE * jnp.maximum(pen[k] - CONTACT_SLOP, 0.0) / DT
            m_n = 1.0 / (m_inv + i_inv * _cross2(r, nk) ** 2)
            new_n = jnp.clip(acc_n[k] + m_n * (target - vn), 0.0, LEG_FORCE * DT) * active
            dn = new_n - acc_n[k]
            m_t = 1.0 / (m_inv + i_inv * _cross2(r, tk) ** 2)
            lim = LEG_FRICTION * new_n
            new_t = jnp.clip(acc_t[k] - m_t * vt, -lim, lim)
            dt_ = new_t - acc_t[k]
            j = dn * nk + dt_ * tk
            v, w = v + m_inv * j, w + i_inv * _cross2(r, j)
            acc_n, acc_t = acc_n.at[k].set(new_n), acc_t.at[k].set(new_t)
        return (v, w, acc_n, acc_t), None

    (v, w, acc_n, _), _ = jax.lax.scan(one_pass, (p.velocity[LANDER], p.angular_velocity[LANDER],
                                                  jnp.zeros(2), jnp.zeros(2)), None, iters)
    return sim.replace(polygon=p.replace(velocity=p.velocity.at[LANDER].set(v),
                                         angular_velocity=p.angular_velocity.at[LANDER].set(w)))


def _impulse(sim: SimState, j: jax.Array, point: jax.Array) -> SimState:
    """Box2D ApplyLinearImpulse on the lander: dv = J/m, dw = (point - centre of mass) x J / I."""
    p = sim.polygon
    r = point - p.position[LANDER]
    return sim.replace(polygon=p.replace(
        velocity=p.velocity.at[LANDER].add(p.inverse_mass[LANDER] * j),
        angular_velocity=p.angular_velocity.at[LANDER].add(p.inverse_inertia[LANDER] * _cross2(r, j))))


def _physics(key, sim: SimState, ground_y, action):
    """Engines (gymnasium's formulas), gravity, leg contacts, one Jax2D step. Returns (sim, m_power, s_power, hit)."""
    lp = sim.polygon
    angle = lp.rotation[LANDER]
    origin = lp.position[LANDER] - _rot(angle) @ COM_OFFSET
    tip = jnp.array([jnp.sin(angle), jnp.cos(angle)])
    side = jnp.array([-tip[1], tip[0]])
    d = jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0) / SCALE

    m_power = (action == 2).astype(jnp.float32)
    ox = tip[0] * (MAIN_ENGINE_Y / SCALE + 2 * d[0]) + side[0] * d[1]
    oy = -tip[1] * (MAIN_ENGINE_Y / SCALE + 2 * d[0]) - side[1] * d[1]
    sim = _impulse(sim, -jnp.array([ox, oy]) * MAIN_ENGINE_POWER * m_power, origin + jnp.array([ox, oy]))

    s_power = ((action == 1) | (action == 3)).astype(jnp.float32)
    direction = (action - 2).astype(jnp.float32)
    ox = tip[0] * d[0] + side[0] * (3 * d[1] + direction * SIDE_ENGINE_AWAY / SCALE)
    oy = -tip[1] * d[0] - side[1] * (3 * d[1] + direction * SIDE_ENGINE_AWAY / SCALE)
    pos = origin + jnp.array([ox - tip[0] * 17 / SCALE, oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE])
    sim = _impulse(sim, -jnp.array([ox, oy]) * SIDE_ENGINE_POWER * s_power, pos)

    p = sim.polygon
    sim = sim.replace(polygon=p.replace(velocity=p.velocity.at[LANDER, 1].add(GRAVITY * DT)))
    sim = _leg_impulses(sim, ground_y)
    sim, (rr, _, _) = ENGINE.step(sim, PARAMS, NO_MOTORS)
    hit = (rr.active & PAIR_LANDER).any()  # the lander's body touches the ground
    return sim, m_power, s_power, hit


def _observe(sim: SimState, ground_y):
    p = sim.polygon
    angle = p.rotation[LANDER]
    pos = p.position[LANDER] - _rot(angle) @ COM_OFFSET
    vel = p.velocity[LANDER]  # Box2D's linearVelocity is the centre of mass velocity, like Jax2D's
    _, pen, _, _ = _legs(p.position[LANDER], angle, vel, p.angular_velocity[LANDER], ground_y)
    leg_contact = (pen > -CONTACT_SLOP).astype(jnp.float32)
    return jnp.array([
        (pos[0] - W / 2) / (W / 2),
        (pos[1] - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H / 2),
        vel[0] * (W / 2) / FPS,
        vel[1] * (H / 2) / FPS,
        angle,
        20.0 * p.angular_velocity[LANDER] / FPS,
        leg_contact[0],
        leg_contact[1],
    ])


def _shaping(s):
    return (-100 * jnp.sqrt(s[0] ** 2 + s[1] ** 2) - 100 * jnp.sqrt(s[2] ** 2 + s[3] ** 2) - 100 * jnp.abs(s[4])
            + 10 * s[6] + 10 * s[7])


def reset(key):
    k_h, k_f, k_step = jax.random.split(key, 3)
    height = jax.random.uniform(k_h, (CHUNKS + 1,), minval=0.0, maxval=H / 2)
    height = height.at[CHUNKS // 2 - 2: CHUNKS // 2 + 3].set(HELIPAD_Y)
    ground_y = 0.33 * (jnp.roll(height, 1) + height + jnp.roll(height, -1))[:CHUNKS]  # height[-1] wraps, as in gym
    xs = jnp.asarray(CHUNK_X)
    bottom = jnp.full(N_TERRAIN, -1.0)
    quads = jnp.stack([jnp.stack([xs[:-1], ground_y[:-1]], -1), jnp.stack([xs[1:], ground_y[1:]], -1),
                       jnp.stack([xs[1:], bottom], -1), jnp.stack([xs[:-1], bottom], -1)], axis=1)  # clockwise
    centre = quads.mean(axis=1)
    p = TEMPLATE.polygon
    force = jax.random.uniform(k_f, (2,), minval=-INITIAL_RANDOM, maxval=INITIAL_RANDOM)
    sim = TEMPLATE.replace(polygon=p.replace(
        position=p.position.at[:N_TERRAIN].set(centre),
        vertices=p.vertices.at[:N_TERRAIN].set(quads - centre[:, None]),
        velocity=p.velocity.at[LANDER].set(force / GYM_MASS * DT)))  # ApplyForceToCenter for one step
    sim, _, _, _ = _physics(k_step, sim, ground_y, jnp.array(0))  # gymnasium's reset() calls step(0)
    obs = _observe(sim, ground_y)
    return obs, EnvState(sim=sim, ground_y=ground_y, prev_shaping=_shaping(obs), t=jnp.array(0, jnp.int32),
                         still=jnp.array(0, jnp.int32))


def step(key, state: EnvState, action):
    """Returns (obs, state, reward, terminated, truncated, final_obs), resetting by itself when an episode ends."""
    k_phys, k_reset = jax.random.split(key)
    sim, m_power, s_power, hit = _physics(k_phys, state.sim, state.ground_y, action)
    obs = _observe(sim, state.ground_y)
    shaping = _shaping(obs)
    reward = shaping - state.prev_shaping - m_power * 0.30 - s_power * 0.03

    p = sim.polygon
    resting = (jnp.linalg.norm(p.velocity[LANDER]) < SLEEP_V) & (jnp.abs(p.angular_velocity[LANDER]) < SLEEP_W)
    still = jnp.where(resting, state.still + 1, 0)
    crashed = hit | (jnp.abs(obs[0]) >= 1.0)
    asleep = still >= SLEEP_STEPS
    reward = jnp.where(crashed, -100.0, reward)
    reward = jnp.where(asleep, 100.0, reward)
    terminated = crashed | asleep
    t = state.t + 1
    truncated = (t >= MAX_STEPS) & ~terminated

    obs_reset, state_reset = reset(k_reset)
    done = terminated | truncated
    state_next = jax.tree.map(lambda a, b: jnp.where(done, a, b), state_reset,
                              EnvState(sim=sim, ground_y=state.ground_y, prev_shaping=shaping, t=t, still=still))
    return jnp.where(done, obs_reset, obs), state_next, reward.astype(jnp.float32), terminated, truncated, obs


# The playground's ObsEncoder ("scaled") for LunarLander-v3 (rl/envs.py ENV_INFO)
OBS_LOW = jnp.array([-1.0, -0.2, -2.0, -2.0, -2.0, -4.0, 0.0, 0.0])
OBS_HIGH = jnp.array([1.0, 1.6, 2.0, 2.0, 2.0, 4.0, 1.0, 1.0])


def encode(obs):
    return 2.0 * (obs - OBS_LOW) / (OBS_HIGH - OBS_LOW) - 1.0


def heuristic(s):
    """gymnasium's hand-written controller (lunar_lander.heuristic, discrete), in JAX."""
    angle_targ = jnp.clip(s[0] * 0.5 + s[2] * 1.0, -0.4, 0.4)
    hover_targ = 0.55 * jnp.abs(s[0])
    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5
    legs = (s[6] > 0) | (s[7] > 0)
    angle_todo = jnp.where(legs, 0.0, angle_todo)
    hover_todo = jnp.where(legs, -s[3] * 0.5, hover_todo)
    return jnp.where((hover_todo > jnp.abs(angle_todo)) & (hover_todo > 0.05), 2,
                     jnp.where(angle_todo < -0.05, 3, jnp.where(angle_todo > 0.05, 1, 0)))
