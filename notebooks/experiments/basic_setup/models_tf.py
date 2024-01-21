import tensorflow as tf

def rectified_logistic_tf(x, a, b, v, L, ell, H):
    return (
        L
        + tf.where(
            tf.less(x, a),
            0.,
            -ell + (
                (H + ell)
                / tf.pow(
                    1
                    + (
                        -1
                        + tf.pow(
                            (H + ell) / ell,
                            v
                        )
                    ) * tf.exp(-b * (x - a)),
                    1 / v
                )
            )
        )
    )


def main():
    import matplotlib.pyplot as plt
    # Parameters for the function
    a = 0.5
    b = 1.0
    v = 2.0
    L = 0.0
    ell = 1.0
    H = 2.0

    # Generate input values
    x_values = tf.linspace(0.0, 100.0, 300)

    # Compute the output values
    y_values = rectified_logistic_tf(x_values, a, b, v, L, ell, H)

    # Convert to numpy for plotting (if running in a non-eager context)
    x_values_np = x_values.numpy()
    y_values_np = y_values.numpy()

    # Plotting
    plt.plot(x_values_np, y_values_np, label='Rectified Logistic Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rectified Logistic Function Plot')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
