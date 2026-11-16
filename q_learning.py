"""Semi-gradient Q-learning con aproximación lineal (Práctica 3 · Sesión 1).

Q-learning es off-policy: el agente se mueve con una política epsilon-greedy,
pero el target usa la acción greedy en ``S'`` (Sutton & Barto, §6.5 y §10.1):

    w <- w + alpha * [R + gamma * max_a q̂(S', a, w) - q̂(S, A, w)] * grad q̂(S, A, w)

Hay que completar dos funciones:

* ``epsilon_q_learning``: vuestro programa de exploración para Q-learning;
* ``episodio_q_learning``: un episodio completo de semi-gradient Q-learning.
"""

from __future__ import annotations

from agente import epsilon_greedy, valores_q  # noqa: F401  (en los huecos)
from entrenamiento import entrena, es_exito  # noqa: F401  (es_exito, en el hueco)


def epsilon_q_learning(k, num_episodios):
    """Epsilon con el que Q-learning explora en el episodio ``k``.

    Args:
        k (int): episodio actual, de 0 a ``num_episodios - 1``.
        num_episodios (int): episodios totales del entrenamiento.

    Returns:
        float: epsilon del episodio, entre 0 y 1.
    """
    # ============================================================
    # START CODE HERE

    # Decae linealmente de 0.2 a 0 en el 80% de los episodios: el 20% final
    # se juega con la política greedy, para que sus bucles se vean y se corrijan.
    return max(0.0, 0.2 * (1 - k / (0.8 * num_episodios)))

    # END CODE HERE
    # ============================================================


def episodio_q_learning(env, agente, alpha, gamma, epsilon, rng):
    """Ejecuta un episodio de semi-gradient Q-learning actualizando ``agente.w``.

    Args y valor devuelto: como en :func:`sarsa.episodio_sarsa`.
    """
    # ============================================================
    # START CODE HERE
    # TODO: como en SARSA, pero el target no depende de la acción que se
    #   ejecutará en S'. Elige la acción con epsilon_greedy al principio de
    #   cada paso y trata terminated y truncated igual que en SARSA.

    obs, _ = env.reset()
    retorno, pasos = 0.0, 0
    while True:
        activos = agente.activos(obs)
        accion = epsilon_greedy(valores_q(agente.w, activos), epsilon, rng)
        obs, recompensa, terminated, truncated, _ = env.step(accion)
        retorno += recompensa
        pasos += 1
        target = recompensa
        if not terminated:
            target += gamma * valores_q(agente.w, agente.activos(obs)).max()
        delta = target - valores_q(agente.w, activos)[accion]
        agente.w[accion, activos] += alpha * delta
        if terminated or truncated:
            return retorno, pasos, es_exito(obs, terminated)

    # END CODE HERE
    # ============================================================


def q_learning(env, agente, num_episodios, alpha, gamma, seed=None, evalua_cada=0):
    """Entrena a ``agente`` con semi-gradient Q-learning.

    Args y valor devuelto: ver :func:`entrenamiento.entrena`.
    """
    return entrena(
        env,
        agente,
        episodio_q_learning,
        epsilon_q_learning,
        num_episodios,
        alpha,
        gamma,
        seed=seed,
        evalua_cada=evalua_cada,
    )
