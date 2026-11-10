"""Double cart-pole: a cart with two sticks joined by a free hinge (no motor at the joint).

        ● top stick
       /
      ●  joint            φ  = angle between the sticks (top relative to lower)
      |  lower stick      θ1 = angle of the lower stick from vertical
   ┌──┴──┐
   │ cart │ ← push        x  = cart position
───┴──────┴─────────

The agent only pushes the cart; the sticks follow from the physics. The only way both
sticks can stay balanced is both straight up: a bent shape cannot be held, because nothing
at the joint holds the top stick.

Observation (6 numbers):  x, ẋ, θ1, θ̇1, φ, φ̇      (angles in rad, positive = leaning to +x)
    the absolute angle of the top stick is θ2 = θ1 + φ
Actions:   `force_levels` constant pushes, default ±10 N (two actions, like CartPole)
Reward:    "alive":   +1 per step (like CartPole)
           "upright": 1 − ½(θ1/limit)² − ½(θ2/limit)²   (1 = both straight, 0 at the limit)
Ends:      terminated if the cart leaves the track (|x| > 2.4) or either stick leans more
           than `angle_limit` from vertical; truncated after 500 steps (TimeLimit).
Start:     all 6 numbers uniform in ±0.05 (almost straight, like CartPole).

Physics: Lagrange's equations for a cart (mass M) with two uniform rods (mass m, length L,
centre of mass at L/2, inertia mL²/12), no friction, integrated with RK4 (4 sub-steps per
0.02 s action). With θ positive toward +x and q = (x, θ1, θ2):

    D(q) q̈ = [F + d2 sinθ1 θ̇1² + d3 sinθ2 θ̇2²,
              f1 sinθ1 − d5 sin(θ1−θ2) θ̇2²,
              f2 sinθ2 + d5 sin(θ1−θ2) θ̇1²]

    D = [[d1,         d2 cosθ1,        d3 cosθ2      ],
         [d2 cosθ1,   d4,              d5 cos(θ1−θ2) ],
         [d3 cosθ2,   d5 cos(θ1−θ2),   d6            ]]

    d1 = M + m1 + m2,  d2 = m1 l1 + m2 L1,  d3 = m2 l2,  d4 = m1 l1² + m2 L1² + I1,
    d5 = m2 L1 l2,     d6 = m2 l2² + I2,    f1 = (m1 l1 + m2 L1) g,  f2 = m2 l2 g
    (l = L/2: distance from a rod's lower end to its centre of mass)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Params:
    cart_mass: float = 1.0
    m1: float = 0.1  # lower stick
    L1: float = 0.5
    m2: float = 0.1  # top stick
    L2: float = 0.5
    g: float = 9.8
    dt: float = 0.02  # one action every 0.02 s (50 per second), as CartPole
    substeps: int = 4

    @property
    def d(self):
        l1, l2 = self.L1 / 2, self.L2 / 2
        I1, I2 = self.m1 * self.L1**2 / 12, self.m2 * self.L2**2 / 12
        return dict(
            d1=self.cart_mass + self.m1 + self.m2,
            d2=self.m1 * l1 + self.m2 * self.L1,
            d3=self.m2 * l2,
            d4=self.m1 * l1**2 + self.m2 * self.L1**2 + I1,
            d5=self.m2 * self.L1 * l2,
            d6=self.m2 * l2**2 + I2,
            f1=(self.m1 * l1 + self.m2 * self.L1) * self.g,
            f2=self.m2 * l2 * self.g,
        )


def accelerations(q: np.ndarray, qd: np.ndarray, force: float, c: dict) -> np.ndarray:
    """q̈ = D(q)⁻¹ · rhs, for q = (x, θ1, θ2)."""
    _, t1, t2 = q
    _, w1, w2 = qd
    s1, c1, s2, c2 = math.sin(t1), math.cos(t1), math.sin(t2), math.cos(t2)
    s12, c12 = math.sin(t1 - t2), math.cos(t1 - t2)
    D = np.array([[c["d1"], c["d2"] * c1, c["d3"] * c2],
                  [c["d2"] * c1, c["d4"], c["d5"] * c12],
                  [c["d3"] * c2, c["d5"] * c12, c["d6"]]])
    rhs = np.array([force + c["d2"] * s1 * w1**2 + c["d3"] * s2 * w2**2,
                    c["f1"] * s1 - c["d5"] * s12 * w2**2,
                    c["f2"] * s2 + c["d5"] * s12 * w1**2])
    return np.linalg.solve(D, rhs)


def energy(q: np.ndarray, qd: np.ndarray, p: Params) -> float:
    """Kinetic + potential energy (constant when no force is applied: a physics check)."""
    c = p.d
    _, t1, t2 = q
    xd, w1, w2 = qd
    T = (0.5 * c["d1"] * xd**2 + 0.5 * c["d4"] * w1**2 + 0.5 * c["d6"] * w2**2
         + c["d2"] * math.cos(t1) * xd * w1 + c["d3"] * math.cos(t2) * xd * w2
         + c["d5"] * math.cos(t1 - t2) * w1 * w2)
    V = c["f1"] * math.cos(t1) + c["f2"] * math.cos(t2)
    return T + V


def rk4_step(q: np.ndarray, qd: np.ndarray, force: float, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Advance one action period (dt) with `substeps` RK4 steps, the force held constant."""
    c, h = p.d, p.dt / p.substeps
    for _ in range(p.substeps):
        k1q, k1v = qd, accelerations(q, qd, force, c)
        k2q, k2v = qd + 0.5 * h * k1v, accelerations(q + 0.5 * h * k1q, qd + 0.5 * h * k1v, force, c)
        k3q, k3v = qd + 0.5 * h * k2v, accelerations(q + 0.5 * h * k2q, qd + 0.5 * h * k2v, force, c)
        k4q, k4v = qd + h * k3v, accelerations(q + h * k3q, qd + h * k3v, force, c)
        q = q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        qd = qd + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return q, qd


def to_obs(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Internal (x, θ1, θ2) and velocities -> observation (x, ẋ, θ1, θ̇1, φ, φ̇)."""
    return np.array([q[0], qd[0], q[1], qd[1], q[2] - q[1], qd[2] - qd[1]], dtype=np.float64)


def from_obs(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, xd, t1, w1, phi, wphi = (float(v) for v in obs)
    return np.array([x, t1, t1 + phi]), np.array([xd, w1, w1 + wphi])


class DoubleCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    X_LIMIT = 2.4

    def __init__(self, render_mode: str | None = None, force_levels: tuple[float, ...] = (-10.0, 10.0),
                 angle_limit_deg: float = 15.0, reward: str = "alive", start_noise: float = 0.05,
                 params: Params = Params()):
        self.p = params
        self.force_levels = tuple(float(f) for f in force_levels)
        self.angle_limit = math.radians(angle_limit_deg)
        self.reward_mode = reward
        self.start_noise = start_noise
        self.action_space = spaces.Discrete(len(self.force_levels))
        hi = np.array([self.X_LIMIT * 2, np.inf, self.angle_limit * 2, np.inf, self.angle_limit * 4, np.inf])
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float64)
        self.render_mode = render_mode
        self._screen = self._clock = None
        self.q = np.zeros(3)
        self.qd = np.zeros(3)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self.np_random.uniform(-self.start_noise, self.start_noise, size=6)
        self.q, self.qd = from_obs(obs)
        if self.render_mode == "human":
            self.render()
        return to_obs(self.q, self.qd), {}

    def step(self, action):
        force = self.force_levels[int(action)]
        self.q, self.qd = rk4_step(self.q, self.qd, force, self.p)
        x, t1, t2 = self.q
        terminated = bool(abs(x) > self.X_LIMIT or abs(t1) > self.angle_limit or abs(t2) > self.angle_limit)
        if self.reward_mode == "upright":
            r = 1.0 - 0.5 * (t1 / self.angle_limit) ** 2 - 0.5 * (t2 / self.angle_limit) ** 2
            reward = max(0.0, r)
        else:
            reward = 1.0
        if self.render_mode == "human":
            self.render()
        return to_obs(self.q, self.qd), reward, terminated, False, {"force": force}

    # ------------------------------------------------------------------ drawing
    def render(self):
        import pygame

        W, H = 600, 400
        if self._screen is None:  # display only: pygame.init() would also start audio
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Double CartPole")
            else:
                self._screen = pygame.Surface((W, H))
            self._clock = pygame.time.Clock()
        scale = W / (2 * self.X_LIMIT)
        surf = pygame.Surface((W, H))
        surf.fill((255, 255, 255))
        ground = 300
        x, t1, t2 = self.q
        cx = W / 2 + x * scale
        pygame.draw.line(surf, (0, 0, 0), (0, ground), (W, ground), 1)
        for edge in (-self.X_LIMIT, self.X_LIMIT):
            ex = W / 2 + edge * scale
            pygame.draw.line(surf, (200, 60, 60), (ex, ground - 8), (ex, ground + 8), 2)
        pygame.draw.rect(surf, (20, 20, 20), pygame.Rect(cx - 25, ground - 30, 50, 30))
        base = (cx, ground - 30)
        joint = (base[0] + self.p.L1 * scale * math.sin(t1), base[1] - self.p.L1 * scale * math.cos(t1))
        tip = (joint[0] + self.p.L2 * scale * math.sin(t2), joint[1] - self.p.L2 * scale * math.cos(t2))
        pygame.draw.line(surf, (202, 152, 101), base, joint, 8)
        pygame.draw.line(surf, (98, 80, 214), joint, tip, 8)
        for pt in (base, joint):
            pygame.draw.circle(surf, (120, 120, 120), (int(pt[0]), int(pt[1])), 5)
        if self.render_mode == "human":
            self._screen.blit(surf, (0, 0))
            pygame.event.pump()
            self._clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)).copy()

    def close(self):
        if self._screen is not None:
            import pygame

            if self.render_mode == "human":
                pygame.display.quit()
            self._screen = None


# ----------------------------------------------------------------------------
# The hand-written controller (LQR on the straight-up linearisation)
# ----------------------------------------------------------------------------
def linear_model(p: Params) -> tuple[np.ndarray, np.ndarray]:
    """ṡ = A s + B F near straight up, s = (x, θ1, θ2, ẋ, θ̇1, θ̇2) (sin θ ≈ θ, cos θ ≈ 1)."""
    c = p.d
    D0 = np.array([[c["d1"], c["d2"], c["d3"]], [c["d2"], c["d4"], c["d5"]], [c["d3"], c["d5"], c["d6"]]])
    Dinv = np.linalg.inv(D0)
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = Dinv @ np.diag([0.0, c["f1"], c["f2"]])
    B = np.zeros((6, 1))
    B[3:, 0] = Dinv @ np.array([1.0, 0.0, 0.0])
    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float, terms: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation by the matrix-exponential series."""
    n = A.shape[0]
    Ad, S, term = np.eye(n), np.eye(n) * dt, np.eye(n)
    for k in range(1, terms):
        term = term @ A * dt / k  # (A dt)^k / k!
        Ad = Ad + term
        S = S + term * dt / (k + 1)  # ∫ e^{Aτ} dτ
    return Ad, S @ B


def lqr_gain(p: Params, Q: np.ndarray, R: float, iters: int = 5000) -> np.ndarray:
    """Discrete-time LQR: F = −K s minimises Σ sᵀQs + R F². Riccati equation solved by iteration."""
    Ad, Bd = discretize(*linear_model(p), p.dt)
    P = Q.copy()
    for _ in range(iters):
        K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        P_new = Q + Ad.T @ P @ (Ad - Bd @ K)
        if np.max(np.abs(P_new - P)) < 1e-10 * max(1.0, np.max(np.abs(P))):
            P = P_new
            break
        P = P_new
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad).ravel()


class LQRController:
    """The best push by the textbook linear controller, mapped to the env's force levels."""

    def __init__(self, p: Params, force_levels: tuple[float, ...], Q=None, R: float = 1.0):
        Q = np.diag([1.0, 100.0, 100.0, 0.1, 1.0, 1.0]) if Q is None else Q
        self.K = lqr_gain(p, Q, R)
        self.levels = np.array(force_levels)

    def force(self, obs: np.ndarray) -> float:
        q, qd = from_obs(obs)
        return float(-self.K @ np.concatenate([q, qd]))

    def __call__(self, obs: np.ndarray) -> int:
        """Action index: the force level closest to the LQR force (2 levels: its sign)."""
        u = np.clip(self.force(obs), self.levels.min(), self.levels.max())
        return int(np.argmin(np.abs(self.levels - u)))


# gym.make("DoubleCartPole-v0") -> this env with a 500-step TimeLimit (truncated at 500)
if "DoubleCartPole-v0" not in gym.registry:
    gym.register(id="DoubleCartPole-v0", entry_point="rl.double_cartpole:DoubleCartPoleEnv", max_episode_steps=500)


def reference_controller() -> LQRController:
    """The 'good policy' for the training metrics: LQR mapped to the two pushes (checked by
    scripts/check_double_cartpole.py: 100/100 starts survive 500 steps, 20/20 survive 10,000)."""
    return LQRController(Params(), (-10.0, 10.0))
