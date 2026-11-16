"""Entorno de almacén (Práctica 3 · Sesión 2 · RL IMAT).

Convenciones del enunciado:

* Recinto cuadrado de 10 x 10 m, con las tres estanterías de la sesión 1
  (0,2 x 5 m, esquina inferior izquierda en (1,9, 1), (4,9, 1) y (7,9, 1)) y
  una zona de entrega de 5 x 1 m con la esquina inferior izquierda en (2,5, 9).
* El agente es un círculo de radio 0,2 m que se desplaza 0,25 m por acción.
* Acciones: 0 = arriba, 1 = abajo, 2 = izquierda, 3 = derecha, 4 = coger y,
  solo si ``drop=True``, 5 = soltar.
* Tres objetos, uno por estantería, ordenados de izquierda a derecha. Con
  ``random_objects=False`` están siempre en (2, 3), (5, 2) y (8, 4); con
  ``random_objects=True`` se sortean en cada episodio sobre su estantería.
* Coger: tiene éxito si el agente no lleva nada y hay un objeto a
  ``PICKUP_DISTANCE`` (0,6 m) o menos del borde del agente, es decir, a 0,8 m o
  menos de su centro. Desde ese momento el objeto viaja con el agente.
* Soltar: si el agente lleva un objeto, el episodio termina; con éxito si su
  centro está dentro de la zona de entrega y con fracaso si no. Soltar sin
  llevar nada no tiene efecto.
* Los tres entornos del enunciado son:

  ===========  ==================  ========  ==================================
  Entorno      ``random_objects``  ``drop``  El episodio termina con éxito...
  ===========  ==================  ========  ==================================
  1            False               False     al coger un objeto
  2            False               True      al soltar un objeto en la zona
  3            True                True      al soltar un objeto en la zona
  ===========  ==================  ========  ==================================

  En los tres termina con fracaso si el agente choca con el perímetro o con una
  estantería, y se trunca (``truncated``) al llegar a ``max_steps`` pasos.
* Posición inicial aleatoria y uniforme entre las posiciones libres.
  ``reset(options={"posicion": (x, y)})`` la fija.
* Observación de 11 componentes: posición del agente (0-1), de los objetos
  1, 2 y 3 (2-7; la de un objeto cogido es la del agente), ``agent_has_object``
  (8), ``collision`` (9) y ``delivery`` (10).
* Recompensa: la calcula la función que se pasa en ``recompensa``; por defecto,
  ``recompensa.recompensa``, que diseñáis vosotros.

Este fichero NO debe modificarse durante la práctica.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

UP, DOWN, LEFT, RIGHT, PICK, DROP = 0, 1, 2, 3, 4, 5
ACTION_NAMES = ("arriba", "abajo", "izquierda", "derecha", "coger", "soltar")

WIDTH = 10.0
HEIGHT = 10.0
# (x, y, ancho, alto) de cada estantería, de izquierda a derecha
SHELVES = (
    (1.9, 1.0, 0.2, 5.0),
    (4.9, 1.0, 0.2, 5.0),
    (7.9, 1.0, 0.2, 5.0),
)
DELIVERY_AREA = (2.5, 9.0, 5.0, 1.0)
FIXED_OBJECTS = ((2.0, 3.0), (5.0, 2.0), (8.0, 4.0))
AGENT_RADIUS = 0.2
AGENT_STEP = 0.25
PICKUP_DISTANCE = 0.6  # desde el borde del agente
MAX_STEPS = 20000

# Configuración de cada uno de los tres entornos del enunciado
ENTORNOS = {
    1: {"random_objects": False, "drop": False},
    2: {"random_objects": False, "drop": True},
    3: {"random_objects": True, "drop": True},
}

# Lo que ha pasado en un paso; se le pasa a la función de recompensa
EVENTOS = (
    "movimiento",  # se ha movido sin chocar
    "choque",  # ha chocado: el episodio termina con fracaso
    "recogida",  # ha cogido un objeto (en el entorno 1, termina con éxito)
    "entrega",  # ha soltado el objeto en la zona: termina con éxito
    "entrega_fallida",  # ha soltado el objeto fuera de la zona: termina
    "sin_efecto",  # coger o soltar sin que pase nada
)

# Desplazamiento (dx, dy) de cada acción de movimiento
MOVES = {
    UP: (0.0, AGENT_STEP),
    DOWN: (0.0, -AGENT_STEP),
    LEFT: (-AGENT_STEP, 0.0),
    RIGHT: (AGENT_STEP, 0.0),
}


def is_collision(x: float, y: float) -> bool:
    """``True`` si el agente centrado en ``(x, y)`` toca una pared o estantería."""
    r = AGENT_RADIUS
    if x <= r or x >= WIDTH - r or y <= r or y >= HEIGHT - r:
        return True
    return any(
        sx - r <= x <= sx + sw + r and sy - r <= y <= sy + sh + r
        for sx, sy, sw, sh in SHELVES
    )


def in_delivery(x: float, y: float) -> bool:
    """``True`` si el centro del agente está dentro de la zona de entrega."""
    dx, dy, dw, dh = DELIVERY_AREA
    return dx <= x <= dx + dw and dy <= y <= dy + dh


def can_pick(agent: tuple[float, float], obj: tuple[float, float]) -> bool:
    """``True`` si el objeto está a ``PICKUP_DISTANCE`` o menos del borde del agente."""
    distancia = float(np.hypot(agent[0] - obj[0], agent[1] - obj[1]))
    return distancia <= PICKUP_DISTANCE + AGENT_RADIUS


def draw_environment(ax) -> None:
    """Dibuja recinto, estanterías y zona de entrega sobre unos ejes de matplotlib."""
    from matplotlib.patches import Rectangle

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")
    for sx, sy, sw, sh in SHELVES:
        ax.add_patch(
            Rectangle((sx, sy), sw, sh, facecolor="indianred", edgecolor="darkred")
        )
    dx, dy, dw, dh = DELIVERY_AREA
    ax.add_patch(
        Rectangle(
            (dx, dy), dw, dh, facecolor="lightgreen", edgecolor="green", alpha=0.6
        )
    )


class WarehouseEnv(gym.Env):
    """Almacén con recogida y entrega de objetos.

    Args:
        random_objects (bool): si ``True``, la posición de los objetos se sortea
            en cada episodio (entorno 3).
        drop (bool): si ``True``, existe la acción soltar y la tarea es entregar
            un objeto en la zona (entornos 2 y 3); si ``False``, la tarea acaba
            al coger un objeto (entorno 1).
        recompensa (callable | None): función
            ``recompensa(obs, accion, nueva_obs, evento) -> float``.
            Si es ``None`` se usa ``recompensa.recompensa``.
        max_steps (int): pasos a partir de los cuales se trunca el episodio.
        render_mode (str | None): ``"human"``, ``"rgb_array"`` o ``None``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        random_objects=False,
        drop=False,
        recompensa=None,
        max_steps=MAX_STEPS,
        render_mode=None,
    ):
        super().__init__()
        if recompensa is None:
            from recompensa import recompensa
        self.random_objects = bool(random_objects)
        self.drop = bool(drop)
        self.recompensa = recompensa
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6 if self.drop else 5)
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(11,), dtype=np.float32
        )

        self.agent_pos = (0.0, 0.0)
        self.objects: list[tuple[float, float]] = list(FIXED_OBJECTS)
        self.held: int | None = None  # índice del objeto que lleva el agente
        self.collision = False
        self.delivery = False
        self.steps = 0
        self._fig = None
        self._ax = None

    @property
    def agent_has_object(self) -> bool:
        return self.held is not None

    # ------------------------------------------------------------------ api
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_objects:
            self.objects = [self._sample_on_shelf(s) for s in SHELVES]
        else:
            self.objects = list(FIXED_OBJECTS)
        if options and options.get("posicion") is not None:
            x, y = (float(v) for v in options["posicion"])
            if is_collision(x, y):
                raise ValueError(f"Posición inicial no válida: ({x}, {y})")
            self.agent_pos = (x, y)
        else:
            self.agent_pos = self._sample_free_position()
        self.held = None
        self.collision = False
        self.delivery = False
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Acción no válida en este entorno: {action}")
        self.steps += 1
        obs = self._get_obs()
        terminated = False

        if action in MOVES:
            dx, dy = MOVES[action]
            x, y = self.agent_pos
            new_pos = (x + dx, y + dy)
            if is_collision(*new_pos):
                # El agente se queda donde estaba: el episodio acaba en fracaso.
                self.collision = True
                terminated = True
                evento = "choque"
            else:
                self.agent_pos = new_pos
                evento = "movimiento"
        elif action == PICK:
            evento = self._pick()
            terminated = evento == "recogida" and not self.drop
        else:
            evento = self._drop()
            terminated = evento != "sin_efecto"

        truncated = (not terminated) and self.steps >= self.max_steps
        nueva_obs = self._get_obs()
        reward = float(self.recompensa(obs, action, nueva_obs, evento))
        return nueva_obs, reward, terminated, truncated, {"evento": evento}

    # ------------------------------------------------------------- internos
    def _pick(self) -> str:
        if self.held is not None:
            return "sin_efecto"
        for i, obj in enumerate(self.objects):
            if can_pick(self.agent_pos, obj):
                self.held = i
                return "recogida"
        return "sin_efecto"

    def _drop(self) -> str:
        if self.held is None:
            return "sin_efecto"
        # El objeto se queda donde lo suelta el agente.
        self.objects[self.held] = self.agent_pos
        self.held = None
        if in_delivery(*self.agent_pos):
            self.delivery = True
            return "entrega"
        return "entrega_fallida"

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)
        obs[0:2] = self.agent_pos
        for i, obj in enumerate(self.objects):
            obs[2 + 2 * i : 4 + 2 * i] = self.agent_pos if i == self.held else obj
        obs[8] = float(self.agent_has_object)
        obs[9] = float(self.collision)
        obs[10] = float(self.delivery)
        return obs

    def _sample_free_position(self):
        rng = self.np_random
        while True:
            x = float(rng.uniform(AGENT_RADIUS, WIDTH - AGENT_RADIUS))
            y = float(rng.uniform(AGENT_RADIUS, HEIGHT - AGENT_RADIUS))
            if not is_collision(x, y):
                return x, y

    def _sample_on_shelf(self, shelf):
        """Un objeto sobre la estantería: a un cuarto o a tres cuartos de su
        anchura, y a 0,5 m o más de sus extremos."""
        rng = self.np_random
        sx, sy, sw, sh = shelf
        x = sx + (0.25 if rng.uniform() < 0.5 else 0.75) * sw
        y = float(rng.uniform(sy + 0.5, sy + sh - 0.5))
        return float(x), y

    # --------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return None
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        if self._fig is None:
            if self.render_mode == "human":
                plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        ax = self._ax
        ax.clear()
        draw_environment(ax)
        for i, obj in enumerate(self.objects):
            if i != self.held:
                ax.add_patch(Circle(obj, 0.2, facecolor="royalblue"))
        color = "red" if self.agent_has_object else "orange"
        ax.add_patch(Circle(self.agent_pos, AGENT_RADIUS, facecolor=color))
        ax.set_title(f"Paso {self.steps}")

        if self.render_mode == "human":
            plt.pause(1.0 / self.metadata["render_fps"])
            return None
        self._fig.canvas.draw()
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        return rgba[..., :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None


if __name__ == "__main__":
    # Comprobación del entorno: unos pasos aleatorios en cada uno de los tres
    # entornos, con recompensa nula. Con --render se ven en una ventana.
    import sys

    modo = "human" if "--render" in sys.argv else None
    for n, config in ENTORNOS.items():
        env = WarehouseEnv(**config, recompensa=lambda *_: 0.0, render_mode=modo)
        obs, _ = env.reset(seed=n)
        for _ in range(20):
            obs, _, terminated, truncated, info = env.step(env.action_space.sample())
            env.render()
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        print(f"Entorno {n}: {env.action_space.n} acciones, observación {obs.shape}")
    print("El entorno funciona.")
