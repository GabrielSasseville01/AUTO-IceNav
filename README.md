# Autonomous Ship Navigation in Ice-Covered Waters with Diffusion Model Predictive Control

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 	![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

**Authors**: [Diego Calanzone](https://halixness.github.io/) & [Akash Karthikeyan](#) & [Gabriel Sasseville](#)<br>
**Affiliation**: MILA Quebec AI Institute, Université de Montréal

This project focuses on two main problems: (1) modeling ocean currents with ice covers and the interaction with icebreakers and commercial ships; (2) training ship controllers that leverage such dynamics model to either control autonomously or assist with the navigation of ships in glacial waters. The scope of this project originates from an increasingly relevant problem in modern transportation: the opening of novel trading routes induced by climate change [[1](https://www.thearcticinstitute.org/future-northern-sea-route-golden-waterway-niche/)] [[2](https://www.nature.com/articles/s41467-025-64437-4)] [[3](https://ici.radio-canada.ca/rci/en/news/2196943/melting-arctic-could-create-shipping-superhighway-and-a-surge-in-emissions-study)].

## Order of contents
- Installation & Environment Setup
- Data access
- Preprocessing of data
- Modeling ocean dynamics
- Running the simulator

### Installation & Environment Setup

1. Clone the repo and navigate to the directory
```
  https://github.com/ddidacus/diffmpc
```
2. Create an environment with `uv`
```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv
```
3. Install packages and the `ship_ice_planner` module
```
  uv sync
  uv pip install -e .
```

### Data access
You will need to download: the pre-trained dynamics residual model; the simulation map (pre-generated polygons) or the raw satellite map for preprocessing; the experimental configuration.

First, download the experiment configuration file (pickle):
```
  gdown 1DuuVJfHHxXJqVZ1KG60q_X2uK5ZoS2cS --output data/
```

Secondly, download the pre-generated polygon map, which represents the rendered satellite image that will be loaded by the simulator:
```
  gdown 1aJJdClZHxwOFd7wFO7PW8CDcQ-hsF4f5 --output data/
```

For one of our baselines (the neural model), you can downloaded our training dataset of samples trajectories:
```
  gdown 1Ssd7J_VgaTUYymt7bdxg6_a-d0lpkzYv --output data/
```

### Preprocessing data
Alternatively, you can process raw satellite images of ice-covered waters to generate your own simulation map. We used imagery collected by the MEDEA satellite system [Multi-satellite floe size distribution of Arctic sea ice 2000-2020, Hwang et al. 2022](https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=7dc6e19d-79fa-41f1-99a7-da408592382f). 

You can download the satellite image for our simulation location of reference:
```
  gdown 1Bk2QQaUft7j8kSz088lGYwHEAMRWR_cf --output data/
  gdown 1SUFHewABXArSkEn3cs8bqU-5OuViSlQY --output data/
```

Then generate the poly map:
```
  python ship_ice_planner/image_process/generate_poly_map.py \
    --input <path_to_segmented_png> \
    --output <path_to_pkl_icefloes_map> \
    --crop_size <edge_in_pixels_of_squared_map> \
    --visualize
```

Remember to update your configuration file to point to the generated ice flow map, for instance in `configs/debug_mode.yaml`:
```
  ...
  map_file: "data/MEDEA_fram_20100629.pkl"
  ...
```

With the same satellite image, it is also possible to generate a ridge density map using the method from [Ice ridge density signatures in high-resolution SAR images, Lensu et al. 2022](https://tc.copernicus.org/articles/16/4363/2022/). This is an important phenomenon to account for during optimization. The process is similar:

Download the high-resolution satellite image as a .tif file:
```
  gdown 1jV6xLCtFoe4qCWu_pOy-OnXI583lJ3Q3 --output data/
```

Then generate the ridge density map:
```
  python ship_ice_planner/image_process/generate_ridge_costmap.py \
  --input data/MEDEA_fram_20100629.png \
  --satellite data/MEDEA_fram_20100629_satellite.tif \
  --output data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl \
  --crop_size 600 \
  --costmap_scale 0.5 \
  --method combined \
  --window_size 25 \
  --visualize
```

Ensure your simulation config file has the following:
```
  ...
  costmap:
  ridge_costmap_file: "data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl"
  ridge_cost_weight: 2.0
  polygon_file: "data/ice_floes_MEDEA_fram_20100629.pkl"
  ...
```

### Modeling ocean dynamics
Our dynamics model for simulation builds upon the following works:
- [AUTO-IceNav: A Local Navigation Strategy for Autonomous Surface Ships in Broken Ice Fields, de Schaetzen et al. 2024](https://arxiv.org/abs/2411.17155)
- [NASA ECCO4, Physical Oceanography Distributed Archive Center](https://podaac.jpl.nasa.gov/announcements/2021-04-27-ECCO-Version-4-Datasets-Release)
- [SubZero: A Sea Ice Model With an Explicit Representation of the Floe Life Cycle, Manucharyan et al. 2022]([https://ams.confex.com/ams/2025Summit/meetingapp.cgi/Paper/459731](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003247))
- [On generalized residual network for deep learning of unknown dynamical systems, Chen et al. 2021](https://www.sciencedirect.com/science/article/abs/pii/S0021999121002576)

We extend the `AUTO-IceNav` simulator with: 
- Physics for the ice-ridging phenomenon, leading to more accurate predictions of ice floe thickness and drift.
- Implemented navier-stokes equations for 2D shallow waters, augmented with a deep residual model ([Chen et al. 2021](https://www.sciencedirect.com/science/article/abs/pii/S0021999121002576)) for supervised training on ECCO4 oceanographic data.
- Map rendering based on satellite imagery of ice-covered waters, allowing us to ground our simulation to realistic scenarios for which we can embed real data on exogenous disturbances e.g. ocean currents.

You can select and download data from ECCO4 through our script:
```
  python scripts/dynamics/seanet_preprocess.py
```
Where you will need to set the WEBDAV credentials obtained through signing up to ECCO4:
```
  WEBDAV_USERNAME = ""
  WEBDAV_PASSWORD = ""
```

The script will generate the equivalent file of `data/ECCO4.pkl`, which can be downloaded from here instead:
```
  gdown 1LC-PYrSpmVAkj4x2Cp8VIWQYA--Mf3oo --output data/
```

To train our dynamics model with ECCO4, run:
```
  python scripts/dynamics/seanet_train.py
```

Alternatively, you can download our pretrained simulator here:
```
  gdown 15r4YbE-Gka0AmnPMgghKlxLgnVxim3Og --output data/
```

You will need to download the tensor with the initialization of the state of ocean currents (based on the first sampled timeframe in ECCO4):
```
  gdown 1N6iXWUxjvToKQSipU6mSU-F52iH98Cjg --output data/
```

You will finally need to set the config file accordingly, e.g. in `config/debug_mode.yaml`:
```
  ...
  seamap_path: "ecco4_x0.pth"
  seanet_path: "seanet.pth"
  ...
```

**Note**: this pre-generated dataset is set to our simulation location (indicated in the following section).

You can alternative generate a random ice floes map:
```
  python -m ship_ice_planner.experiments.generate_rand_exp
```

## Running the Simulator
Our simulation is based on real ice-floe maps sampled from the Arctic Ocean and Beaufort Sea. Here follow the reference simulation details:
- Codename: [MEDEA_fram_20100629](https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=synth:7dc6e19d-79fa-41f1-99a7-da408592382f:L0ZyYWN0dXJpbmcvZnNkL01FREVBX2ZyYW1fMjAxMDA2MjkucG5n)
- Location: Arctic Ocean, `85°00'00"N 0°03'00"E`
- Timeframe: 2000-2020
- We visualized available simulation locations [at this link](https://earth.google.com/earth/d/1y1QtmHqyp0btaxTsEV_iT2W4x2XOZvOh?usp=sharing)

From this location, we used collected MEDEA satellite imagery and ocean currents data from [NASA ECCO4, Physical Oceanography Distributed Archive Center](https://podaac.jpl.nasa.gov/announcements/2021-04-27-ECCO-Version-4-Datasets-Release).


### Running experiments
You can run the main simulation script with different controllers:
```
  # Diffusion + MPC
  python scripts/run_simulation.py data/experiment_configs.pkl configs/controllers/diffusion_controller.yaml
  
  # A* "lattice" algorithm + MPC
  python scripts/run_simulation.py data/experiment_configs.pkl configs/controllers/astar_controller.yaml

  # Fully connected NN + MPC
  python scripts/run_simulation.py data/experiment_configs.pkl configs/controllers/nmpc_controller.yaml
```

You can additionally run the simulation in debug mod for development purposes:
```
  python scripts/run_simulation.py data/experiment_configs.pkl configs/debug_mode.yaml
```
Debug mode simplifies the simulation steps to minimize latency, the A* lattice planner is used by default. 

> The folder `trajectories/` stores, for each method, simulation information that will be used for evaluation. At the beginning of the simulation, the polygon map and the cost map are stored in pickle files, along with the initially predicted path lines. At the end of the simulation, an additional path line will be stored along with a `.json` file reporting collision data and consumption. You can set the target path to store all this information through the field `analysis_save_path` in the yaml config file.

## Directory Structure

```
├── configs/                    # Configuration files
│   ├── controllers/            # Controller-specific configs
│   └── debug_mode.yaml         # Debug configuration
├── data/                       # Datasets and experiment configs
├── docs/                       # Documentation and images
├── scripts/                    # Utility scripts
│   ├── dynamics/               # Dynamics model training
│   └── run_simulation.py       # Main simulation runner
├── ship_ice_planner/           # Main package
│   ├── controller/             # Ship controllers and dynamics
│   ├── planners/               # Path planners (diffusion, lattice, skeleton)
│   ├── image_process/          # Satellite image processing
│   ├── evaluation/             # Evaluation and metrics
│   ├── experiments/            # Experiment generation
│   ├── geometry/               # Geometric utilities
│   ├── utils/                  # General utilities
│   ├── sim2d.py                # 2D simulator main loop
│   └── a_star_search.py        # A* search implementation
├── test/                       # Test suite
├── main.py                     # Main entry point
├── pyproject.toml              # Project configuration
└── requirements.txt            # Python dependencies
```

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@misc{calanzone2025diffmpc,
  title        = {Autonomous Ship Navigation in Ice-Covered Waters with Diffusion Model Predictive Control},
  author       = {Calanzone, Diego and Karthikeyan, Akash and Sasseville, Gabriel},
  year         = {2025},
  howpublished = {\url{https://github.com/ddidacus/diffmpc}},
  note         = {MILA Quebec AI Institute, Université de Montréal}
}
```

## Contact

Feel free to open a git issue if you come across any issues or have any questions.
