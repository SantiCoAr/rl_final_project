# Práctica final · Aprendizaje por Refuerzo (IMAT)

Práctica 3 · Sesión 2: un agente que se mueve por un almacén y aprende a coger
objetos de las estanterías y a entregarlos en una zona de entrega.

El enunciado completo está en [docs/P3S2_RL_IMAT_2627.pdf](docs/P3S2_RL_IMAT_2627.pdf).

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
├── docs/             # Enunciado de la práctica
└── requirements.txt
```

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

- [ ] Diseñar la recompensa de cada entorno (`recompensa.py`)
- [ ] Entrenar un agente para cada entorno
- [ ] Evaluar los resultados y redactar la memoria
