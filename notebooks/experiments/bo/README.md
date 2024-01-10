### Order of executing these scripts:
    * learn_posterior.py -> Learns posterior and saves it to disk
    * simulate_data.py
        - uses learnt posterior
        - simulates existing and new participants and saves it to disk
    * plot_simulated_data.py
        - uses simulated posterior
        - plots specified number of draws (as if they were different muscles, horizontally) and saves it to disk

* All models generally go in models.py (for eg. a new model used for fitting)
* All OS dependent paths are defined as global variables (ALL_CAPITALS) right below import statements
