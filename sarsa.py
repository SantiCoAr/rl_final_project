"""Semi-gradient SARSA con aproximación lineal (Práctica 3 · Sesión 1).

SARSA es on-policy: el target usa la acción ``A'`` que el agente va a
ejecutar de verdad en ``S'``, elegida con la misma política epsilon-greedy que
está siguiendo (Sutton & Barto, §10.1):

    w <- w + alpha * [R + gamma * q̂(S', A', w) - q̂(S, A, w)] * grad q̂(S, A, w)

Hay que completar dos funciones:

* ``epsilon_sarsa``: vuestro programa de exploración, el epsilon de cada
  episodio;
* ``episodio_sarsa``: un episodio completo de semi-gradient SARSA.
"""

from __future__ import annotations

from agente import epsilon_greedy, valores_q  # noqa: F401  (en los huecos)
from entrenamiento import entrena, es_exito  # noqa: F401  (es_exito, en el hueco)


def epsilon_sarsa(k, num_episodios):
    """Epsilon con el que SARSA explora en el episodio ``k``.

    Args:
        k (int): episodio actual, de 0 a ``num_episodios - 1``.
        num_episodios (int): episodios totales del entrenamiento.

    Returns:
        float: epsilon del episodio, entre 0 y 1.
    """
    # ============================================================
    # START CODE HERE

    # Decae linealmente hasta 0: al final SARSA evalúa la política greedy,
    # que es la determinista con la que se le evalúa.
    return 0.2 * (1 - k / num_episodios)

    # END CODE HERE
    # ============================================================


def episodio_sarsa(env, agente, alpha, gamma, epsilon, rng):
    """Ejecuta un episodio de semi-gradient SARSA actualizando ``agente.w``.

    Args:
        env: entorno de navegación.
        agente (AgenteLineal): agente; sus pesos ``agente.w`` se actualizan.
        alpha (float): tamaño de paso.
        gamma (float): factor de descuento.
        epsilon (float): probabilidad de explorar en este episodio.
        rng (np.random.Generator): generador para ``epsilon_greedy``.

    Returns:
        tuple: ``(retorno, pasos, exito)``: suma de recompensas sin descontar,
        número de pasos y si el episodio acabó en la zona objetivo (usa
        ``es_exito``).
    """
    # ============================================================
    # START CODE HERE
    # TODO: estructura de un episodio:
    #   1. obs, _ = env.reset(); tiles activos con agente.activos(obs) y
    #      primera acción con epsilon_greedy.
    #   2. Repite: env.step(accion), elige A' en S', calcula el target,
    #      actualiza agente.w y avanza S <- S', A <- A'.
    #   3. Acumula retorno y pasos, y devuelve los tres valores al terminar.
    # HINT: env.step devuelve terminated y truncated. Solo terminated significa
    #   que S' es terminal (q̂(S', ·) = 0); un episodio truncado acaba, pero
    #   S' no es terminal.

    obs, _ = env.reset()
    activos = agente.activos(obs)
    accion = epsilon_greedy(valores_q(agente.w, activos), epsilon, rng)
    retorno, pasos = 0.0, 0
    while True:
        obs, recompensa, terminated, truncated, _ = env.step(accion)
        retorno += recompensa
        pasos += 1
        target = recompensa
        if not terminated:
            activos_sig = agente.activos(obs)
            accion_sig = epsilon_greedy(valores_q(agente.w, activos_sig), epsilon, rng)
            target += gamma * valores_q(agente.w, activos_sig)[accion_sig]
        delta = target - valores_q(agente.w, activos)[accion]
        # grad q̂(S, A, w) = x(S) en la fila A: solo cambian los tiles activos.
        agente.w[accion, activos] += alpha * delta
        if terminated or truncated:
            return retorno, pasos, es_exito(obs, terminated)
        activos, accion = activos_sig, accion_sig

    # END CODE HERE
    # ============================================================


def sarsa(env, agente, num_episodios, alpha, gamma, seed=None, evalua_cada=0):
    """Entrena a ``agente`` con semi-gradient SARSA.

    Args y valor devuelto: ver :func:`entrenamiento.entrena`.
    """
    return entrena(
        env,
        agente,
        episodio_sarsa,
        epsilon_sarsa,
        num_episodios,
        alpha,
        gamma,
        seed=seed,
        evalua_cada=evalua_cada,
    )
