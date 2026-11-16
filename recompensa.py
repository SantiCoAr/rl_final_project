"""Diseño de la recompensa (Práctica 3 · Sesión 2 · RL IMAT).

El entorno llama a ``recompensa`` en cada paso, después de aplicar la acción, y
devuelve lo que esta función calcule como ``reward`` de ``env.step``. Lo que
decidáis aquí define la tarea que el agente aprende de verdad: comprobad que,
en cada entorno, maximizar el retorno coincide con lo que queréis (coger un
objeto, o entregarlo en la zona, sin chocar y en el menor número de pasos).

Seguramente necesitéis una recompensa distinta en cada entorno. Podéis
programar varias funciones con esta misma firma y pasar la que toque al crear
el entorno: ``WarehouseEnv(**ENTORNOS[2], recompensa=mi_recompensa_2)``.

Si necesitáis la geometría del almacén, importadla de ``env`` (``SHELVES``,
``DELIVERY_AREA``, ``FIXED_OBJECTS``, ``WIDTH``, ``HEIGHT``...).
"""

from __future__ import annotations


def recompensa(obs, accion, nueva_obs, evento):
    """Recompensa de una transición.

    Args:
        obs (np.ndarray): observación de 11 componentes antes de la acción.
        accion (int): acción elegida (0-3 moverse, 4 coger, 5 soltar).
        nueva_obs (np.ndarray): observación después de la acción.
        evento (str): lo que ha pasado, uno de ``env.EVENTOS``:
            ``"movimiento"``, ``"choque"``, ``"recogida"``, ``"entrega"``,
            ``"entrega_fallida"`` o ``"sin_efecto"``.

    Returns:
        float: recompensa de la transición.
    """
    # ============================================================
    # START CODE HERE

    raise NotImplementedError("Implementa recompensa y borra esta línea.")

    # END CODE HERE
    # ============================================================
