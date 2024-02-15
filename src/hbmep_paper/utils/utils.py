import os
import logging

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(dir, fname, level=logging.INFO):
    fname = f"{fname.split('.')[0]}.log"
    dest = os.path.join(
        dir, fname
    )
    logging.basicConfig(
        format=FORMAT,
        level=level,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")
    return


def run_svi(
    df, model, n_steps=2000
)
    optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
    self._guide = numpyro.infer.autoguide.AutoNormal(model._model)
    svi = SVI(
        model._model,
        model._guide,
        optimizer,
        loss=Trace_ELBO()
    )
    svi_result = svi.run(
        model.rng_key,
        n_steps,
        *model._get_from_dataframe(df, model.regressor),
        *self._collect_response(df=df),
        progress_bar=PROGRESS_BAR
    )
    end = time.time()
    time_taken = end - start
    time_taken = np.array(time_taken)
    predictive = Predictive(
        self._guide,
        params=svi_result.params,
        num_samples=4000
    )
    posterior_samples = predictive(self.rng_key, *self._collect_regressor(df=df))
    posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
    return svi_result, posterior_samples, time_taken
