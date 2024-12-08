# Probabilistic Disease Spread on Directed Graph

![header](./header.png)

This code is for my current project, which investigates the probabilistic spread of epidemics on directed graphs and derives the steady-state distribution for each individual in the network. Our goal is to demystify the spread process, approximate computationally intensive components using numerical methods, and accelerate the entire process through relaxation techniques. For simplicity, the epidemic spread is modeled using the SIR framework, with Monte Carlo simulations serving as the baseline for comparison.

## Evironment Setting

First create and activate the virtual evironment:

```bash
conda create -n epidemics python==3.10

source activate epidemics
```


Run the following to install dependencies:

```bash
pip install -r requirements.txt
```

Specificlly, for installation of `torch` and `PyG`:

```bash
pip install https://download.pytorch.org/whl/cu118/torch-2.4.1%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=740bae6eb10c6b41cb86c4f9e84da0b4533b5595aed4f06694d95d5e32b4076c
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install torch-geometric
```

## Execution

Simply run 

```bash
python main.py
```

Options for arguments can be accessed in `main.py`.



## Results

After execution of the code, there will generate a folder containing a trajectory plot

![sir_trajectories](./output/sir_trajectories.png)

and a `.csv` table looks like

```
                             Metric         Value
0              Monte Carlo Time (s)  5.702568e+01
1             Ground Truth Time (s)  2.435303e-02
2            Approximation Time (s)  2.206039e-02
3      MC Average Convergence Steps  4.581400e+01
4              GT Convergence Steps  7.500000e+01
5   Approximation Convergence Steps  7.600000e+01
6             GT Final S Percentile  3.040000e+01
7         Approx Final S Percentile  3.040000e+01
8                    GT Kendall-Tau  8.372881e-01
9                        GT p-value  7.614124e-17
10               Approx Kendall-Tau  8.423729e-01
11                   Approx p-value  4.956615e-17
```

which reveal our analysis of result.



## To-do List

- [x] Change Monte Carlo simulation to multi-thread processing.
- [ ] Implement on large directed networks.
- [ ] Explore on the effect of initial infected set.