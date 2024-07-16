hbMEP paper
=====

This repository has code to reproduce the results in the paper `Hierarchical Bayesian estimation of motor-evoked potential recruitment curves yields accurate and robust estimates <https://arxiv.org/abs/2407.08709>`_.

It uses the `hbmep v0.5.0 <https://github.com/hbmep/hbmep>`_. See `pyproject.toml <https://github.com/hbmep/hbmep-paper/blob/main/pyproject.toml>`_ for dependencies.

Installation
---------------

Begin by creating a virtual environment.

.. code-block:: bash

    python3.11 -m venv .venv

Note that the above command uses Python 3.11. If you have a different version of Python, you can use `conda <https://conda.io>`_ to create a new environment with the required version of Python.

.. code-block:: bash

    conda create -n python-311 python=3.11.9 -y
    conda activate python-311
    python -m venv .venv
    conda deactivate

We can then install in editable mode.

.. code-block:: bash

	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e .

Now, the Python interpreter should be located at ``.venv/bin/python``. You can use this to run the scripts in the `notebooks  <https://github.com/hbmep/hbmep-paper/tree/main/notebooks>`_ directory.

Citation
-----------

Please cite `Tyagi et al., 2024 <https://arxiv.org/abs/2407.08709>`_ if you find this code useful in your research. The BibTeX entry for the paper is::

    @article{tyagi_hierarchical_2024,
        title = {Hierarchical {Bayesian} estimation of motor-evoked potential recruitment curves yields accurate and robust estimates},
        author = {Tyagi, Vishweshwar and Murray, Lynda M. and Asan, Ahmet S. and Mandigo, Christopher and Virk, Michael S. and Harel, Noam Y. and Carmel, Jason B. and McIntosh, James R.},
        journal={arXiv preprint arXiv:2407.08709},
        year = {2024},
        doi = {http://doi.org/10.48550/arXiv.2407.08709}
    }
