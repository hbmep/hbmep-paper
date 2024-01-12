import jax.numpy as jnp


def Existing(mu, c_1, c_2):
    return c_1 + jnp.true_divide(c_2, mu)


def SD(mu, c_1, c_2):
    return (
        jnp.true_divide(
            mu,
            jnp.power(
                c_1 + jnp.multiply(c_2, mu),
                2
            )
        )
    )


def PowerSD(mu, c_1, c_2, c_3):
    return (
        jnp.true_divide(
            mu,
            jnp.power(
                c_1 + jnp.multiply(c_2, jnp.power(mu, c_3)),
                2
            )
        )
    )


def PowerSDMinusL(mu, L, c_1, c_2, c_3):
    return (
        jnp.true_divide(
            mu,
            jnp.power(
                c_1 + jnp.multiply(c_2, jnp.power(mu - L, c_3)),
                2
            )
        )
    )


def PieceWiseSD(mu, c_1, c_2, c_3, x, a,):
    return (
        jnp.where(
            jnp.less(x, a),
            jnp.true_divide(
                mu,
                jnp.power(c_3, 2)
            ),
            jnp.true_divide(
                mu,
                jnp.power(
                    c_1 + jnp.multiply(c_2, mu),
                    2
                )
            )
        )
    )


def PieceWiseVariance(mu, c_1, c_2, c_3, x, a,):
    return (
        jnp.where(
            jnp.less(x, a),
            jnp.true_divide(mu, c_3),
            jnp.true_divide(mu, c_1 + jnp.multiply(c_2, mu))
        )
    )
