import jax.numpy as jnp

EPSILON = 1e-15


def sd_minus_L(mu, L, c_1, c_2):
    return (
        jnp.true_divide(
            mu,
            c_1
            + jnp.multiply(
                c_2,
                mu - L + EPSILON
            )
        )
    )


def sd_power(mu, c_1, c_2, c_3):
    return (
        jnp.true_divide(
            mu,
            c_1
            + jnp.multiply(
                c_2,
                jnp.power(mu + EPSILON, c_3)
            )
        )
    )


def sd_power_minus_L(mu, L, c_1, c_2, c_3):
    return (
        jnp.true_divide(
            mu,
            c_1
            + jnp.multiply(
                c_2,
                jnp.power(mu - L + EPSILON, c_3)
            )
        )
    )


def sd_power_minus_L2(mu, L, c_1, c_2, c_3):
    return (
        jnp.true_divide(
            mu,
            c_1
            + jnp.multiply(
                c_2,
                jnp.power(mu - L + EPSILON, c_3 + EPSILON)
            )
        )
    )
