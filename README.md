# Práctica final · Aprendizaje por Refuerzo (IMAT)

Práctica 3 · Sesión 2: un agente que se mueve por un almacén y aprende a coger
objetos de las estanterías y a entregarlos en una zona de entrega.

- Enunciado: [docs/P3S2_RL_IMAT_2627.pdf](docs/P3S2_RL_IMAT_2627.pdf)
- Pautas del proyecto (qué se evalúa, presentación y cuaderno):
  [docs/P3_pautas_proyecto_2627.pdf](docs/P3_pautas_proyecto_2627.pdf)

## Entorno

Un recinto de 10 × 10 m con tres estanterías y una zona de entrega. El agente
es un círculo que se mueve 0,25 m por acción.

| Entorno | Objetos  | Acción soltar | Termina con éxito al...          |
|---------|----------|---------------|----------------------------------|
| 1       | fijos    | no            | coger un objeto                  |
| 2       | fijos    | sí            | soltar un objeto en la zona      |
| 3       | aleatorios | sí          | soltar un objeto en la zona      |

En los tres el episodio termina con fracaso si el agente choca contra una pared
o una estantería.

## Estructura

```
.
├── env.py            # WarehouseEnv (Gymnasium). No se modifica.
├── recompensa.py     # Función de recompensa (a implementar)
├── docs/             # Enunciado y pautas del proyecto
└── requirements.txt
```

El resto de ficheros `.py` (`agente.py`, `representacion.py`, `sarsa.py`,
`q_learning.py`, `entrenamiento.py`, `evaluacion.py`, `main.py`...) vienen de la
sesión 1 (tile coding + SARSA/Q-learning lineal) y sirven de punto de partida.
Todavía usan el entorno de navegación de la sesión 1 (`NavigationEnv`), así que
hay que adaptarlos a `WarehouseEnv`.

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Comprobación rápida del entorno (añade --render para verlo)
python env.py
```

Uso básico:

```python
from env import WarehouseEnv, ENTORNOS
from recompensa import recompensa

env = WarehouseEnv(**ENTORNOS[2], recompensa=recompensa)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Estado

Los entornos se resuelven en orden (1 → 2 → 3), con un único agente por entorno
(no se permite ensamblar agentes).

- [ ] Adaptar el código de la sesión 1 a `WarehouseEnv`
- [ ] Ingeniería de variables: representación del estado (no basta el tile coding)
- [ ] Diseñar la recompensa de cada entorno (`recompensa.py`)
- [ ] Entrenar y evaluar: entorno 1 → 2 → 3
- [ ] Cuaderno de laboratorio y presentación (8 min)
