
## Running PS4RL

Generate the the filter:

```bash
./generate_dfta.sh
```

Run the experiment for CartPole:

```bash
./run_exp.sh cartpole @examples/rl/envs/cart_pole.txt
```

## Installation

### From Source

If you are installing from source, you will need Python 3.8 or later.

#### Install DeepSynth2

DeepSynth2 can be installed from source with `pip`, `conda` or `poetry`.

```bash
pip install .
```

When using `poetry` in an CUDA environment, then you need to follow every `poetry install` or `poetry update` with:

```bash
pip install torch
```

See this [open issue of poetry](https://github.com/python-poetry/poetry/issues/6409) for more information.

## Documentation

[Online Documentation](https://theomat.github.io/DeepSynth2/)


You might want to generate html pages of the documentation locally, where usage, contribution guidelines and more can be found.
In which case, you will need to use [Sphinx](https://www.sphinx-doc.org/en/master/). 

```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

If Sphinx installation was successful, then use the following command line to generate html pages that you can view by opening the file `docs/build/html/index.html` in your favorite web browser.

```bash
sphinx-build -b html docs/source docs/build/html
```

## Troubleshooting

There are some known issues:

- **seed = 0** is the **same as no seeding**.
- if you get an error after installation try to update/upgrade ``numpy``, it is often due to a discrepancy between the version with which ``vose`` is compiled and the version the environment is running.
- **if you have issues with ``vose``**, you can just uninstall ``vose``, generation speed will be slower but everything will work.
- some dependencies may be missing depending on the DSL you want to use, running any example script with -h will list you the list of available DSL with your current installation.

## The Team

DeepSynth2 is a project initiated by [Nathanaël Fijalkow](https://nathanael-fijalkow.github.io/) and by [Théo Matricon](https://theomat.github.io/).
It is based on the [DeepSynth](https://github.com/nathanael-fijalkow/DeepSynth) project of [Nathanaël Fijalkow](https://nathanael-fijalkow.github.io/), [Guillaume Lagarde](https://guillaume-lagarde.github.io/), [Théo Matricon](https://theomat.github.io/), [Kevin Ellis](https://www.cs.cornell.edu/~ellisk/), [Pierre Ohlmann](https://www.irif.fr/~ohlmann/), Akarsh Potta

Former:

- (2023) [Félix Yvonnet](https://github.com/Felix-Yvonnet) did a 2 months internship to work on restarts, a future feature of DeepSynth2.
- (2023) [Priscilla Tissot](https://fr.linkedin.com/in/priscilla-tissot-9493851b8) did a 7 weeks long internship working on the Carel neural network and trying to improve the performance of our prediction models.
- (2022) [Gaëtan Margueritte](https://github.com/gaetanmargueritte) did a four-month internship. He created the regexp and transduction DSLs, the first tutorial and first drafts of code related to the use of user defined constants.
- (2022) Utkarsh Rajan did a two-month internship. He contributed to the implementation of bucket search and worked on the tower DSL.

## License

DeepSynth2 has a MIT license, as found in the [LICENSE](LICENSE.md) file.
