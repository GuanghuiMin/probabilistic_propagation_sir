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
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
```

## Parameters Setting 
We conduct a thorough survey of parameters for various diseases, including the Nipah virus, Andes hantavirus, MERS, Ebola, Mpox, common cold, and pertussis. The parameters are summarized in the table below.

| **Disease**              | **Infectious Period**                                                                     | **Transmission**                                   | **$R_0$**                                                                                                                                            | **$\gamma$ [1/day]** | **$\beta$ [1/day]** |
|-------------------------|-------------------------------------------------------------------------------------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|---------------------|
| Nipah Virus             | 4--14 days [[WHO]](https://www.who.int/news-room/fact-sheets/)                            | Body fluids                                       | 0.5 [[Luby et al., 2013]](https://doi.org/10.1016/j.antiviral.2013.07.011)                                                                           | 0.1111               | 0.0556              |
| Andes Hantavirus        | 7--39 days (median 18 days) [[Vial et al., 2006]](https://doi.org/10.3201/eid1208.051127) | Respiratory droplets and body fluids             | 1.2 (0.8--1.6) [[Martínez et al., 2020]](https://www.nejm.org/doi/full/10.1056/NEJMoa2009040)                                                        | 0.0556               | 0.0667              |
| MERS                    | 5 days (2--14 days) [[CDC]](https://www.cdc.gov/)                                         | Respiratory droplets                              | 0.5 (0.3--0.8) [[Kucharski et al., 2015]](https://www.eurosurveillance.org/content/10.2807/1560-7917.ES2015.20.25.21167)                             | 0.2                  | 0.1                 |
| Ebola                   | Average 12.7 days [[Eichner et al., 2011]](https://doi.org/10.1016/j.phrp.2011.04.001)    | Body fluids                                       | 1.8 (1.4--1.8) [[Wong et al., 2017]](https://doi.org/10.1017/S0950268817000164)                                                                      | 0.0787               | 0.1417              |
| Mpox                    | 4--14 days [[WHO]](https://www.who.int/news-room/fact-sheets/)                            | Physical contact, body fluids, respiratory droplets, sexual (MSM) | 2.1 (1.1--2.7) [[Grant et al., 2020]](https://doi.org/10.2471/BLT.19.242347), [[Al-Raeei et al. 2023]](https://doi.org/10.1097/MS9.0000000000000229) | 0.1111               | 0.2333              |
| Common Cold             | 7--10 days [[Mayo Clinic]](https://www.mayoclinic.org/)                                   | Respiratory droplets                              | 2--3 [[Mayo Clinic]](https://www.mayoclinic.org/)                                                                                                    | 0.1176               | 0.2941              |
| Pertussis               | At least 2 weeks [[CDC]](https://www.cdc.gov/)                                            | Respiratory droplets                              | 5.5 [[Kretzschmar et al., 2010]](https://doi.org/10.1371/journal.pmed.1000291)                                                                       | 0.0714               | 0.3929              |
## Execution

Simply run 

```bash
python main.py
```

Options for arguments can be accessed in `main.py`.

## Synthetic Data Generation
We try to generate synthetic different types of networks, including Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz (small world). The code is in `networks/__init__.py`.

## Real Data Analysis
We also conduct experiments on large-scale real-world networks.


## Results

After execution of the code, there will be a folder `./output` containing a trajectory plot

![sir_trajectories](./output/sir_trajectories.png)

and a `.csv` table looks like

```
                           Metric         Value
0            Monte Carlo Time (s)  8.587472e+01
1           Ground Truth Time (s)  3.333521e-02
2          Approximation Time (s)  2.951336e-02
3                    SOR Time (s)  2.105403e-02
4      Local Push Approx Time (s)  1.649356e-02
5    MC Average Convergence Steps  3.094000e+01
6            GT Convergence Steps  8.300000e+01
7        Approx Convergence Steps  8.300000e+01
8           SOR Convergence Steps  6.500000e+01
9    Local Push Convergence Steps  3.000000e+00
10          GT Final S Percentile  2.060000e+01
11      Approx Final S Percentile  2.060000e+01
12         SOR Final S Percentile  3.380000e+01
13  Local Push Final S Percentile  9.200000e+01
14                 GT Kendall-Tau  8.361359e-01
15                     GT p-value  1.740959e-84
16             Approx Kendall-Tau  8.361359e-01
17                 Approx p-value  1.740959e-84
18                SOR Kendall-Tau  8.361359e-01
19                    SOR p-value  1.740959e-84
20         Local Push Kendall-Tau  7.985120e-01
21             Local Push p-value  9.481057e-75
22            MC Final S Estimate  4.870860e+02
23      MC Final S Lower CI (95%)  4.820000e+02
24      MC Final S Upper CI (95%)  4.900000e+02
25                     GT Final S  4.852123e+02
26                 Approx Final S  4.852713e+02
27                    SOR Final S  4.861262e+02
28             Local Push Final S  4.896007e+02
```

which reveal our analysis of result.



## To-do List

- [x] Change Monte Carlo simulation to multi-thread processing.
- [x] Implement on large directed networks.
- [ ] Explore on the effect of initial infected set.