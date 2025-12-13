# `Refgen` : Discovering Sustainable Refrigerant Compounds with Physics-Aligned Molecule Sequence Models

[![arXiv](https://img.shields.io/badge/arXiv-2509.19588-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2509.19588)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) 
[![Model on HF](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)](#)


**Authors**: Adrien Goldszal, Diego Calanzone, Vincent Taboga, Pierre-Luc Bacon. <br>
**Affiliation**: Mila Quebec AI Institute, Universite de Montreal.

Most refrigerants currently used in air-conditioning systems, such as hydrofluorocarbons, are potent greenhouse gases and are being phased down. Large-scale molecular screening has been applied to the search for alternatives, but in practice only about 300 refrigerants are known, and only a few additional candidates have been suggested without experimental validation. This scarcity of reliable data limits the effectiveness of purely data-driven methods. We present `Refgen`, a generative pipeline that integrates machine learning with physics-grounded inductive biases. Alongside fine-tuning for valid molecular generation, `Refgen` incorporates predictive models for critical properties, equations of state, thermochemical polynomials, and full vapor compression cycle simulations. These models enable reinforcement learning fine-tuning under thermodynamic constraints, enforcing consistency and guiding discovery toward molecules that balance efficiency, safety, and environmental impact. By embedding physics into the learning process, `Refgen` leverages scarce data effectively and enables de novo refrigerant discovery beyond the known set of compounds.
## Environment setup

1. Making sure `UV` is installed in your system, [check this page](https://docs.astral.sh/uv/getting-started/installation/).
2. Creating the virtual environment
```
    uv venv
```
3. Installing the required libraries
```
    uv pip install -e .
```

## Training the property predictors
Property prediction models are based on [SMI-TED, Soares et al. 2024](https://openreview.net/forum?id=Yq8At31hLi). You will neet to download the pretrained weights with the following command:
```
    mkdir -p src/smi_ted/smi_ted_light/
    wget https://huggingface.co/ibm-research/materials.smi-ted/resolve/main/smi-ted-Light_40.pt -O src/smi_ted/smi_ted_light/smi-ted-Light_40.pt
```
You can then run the script to fine-tune SMI-TED on our provided property datasets:

First, create the dataset splits
```
    bash scripts/predictors/create_dataset_splits.sh <property>
```
Where `<property>` must be must be one of: `lfl, nasa_poly, omega, Pc, re, Tc`.

Then run the finetuning process
```
    bash scripts/predictors/finetune_predictor.sh <property>
```
Alternatively, you can download our predictor checkpoints (fine-tuned SMI-TED):
```
    gdown 15ex_K33AftBrUYyJP1MFcA21rOtwZtTV
    tar -xzvf smi_ted_checkpoints.tar.gz
```

## Training the molecule generator policy
We train a language model as a molecule sequence model in two stages: (1) supervised fine-tuning on a large collection of compounds; (2) RL fine-tuning on property predictors to refine the sampling of optimal molecules. 

### Supervised fine-tuning
You can find our collection of molecules here:
```
    gdown 1V1sx0R1XjhqVoI2iVbmDJEafl0KMJf1G
    tar -xzvf molgen40M.tar.gz
```
You can train your language model with the following script:
```
    bash scripts/sft/pretrain.sh 
```
Alternatively, you can download our pre-trained version of `Llama 3.2 1B`:
```
    gdown 1q7yE2oHTPxSXSJJF4lPf8pzjmOj_HLuB
    tar -xzvf base_model.gz
```
Before pre-training, adjust the paths and parameters in the config file `configs/training/base.yaml` accordingly.

### RL fine-tuning
> **Note**: after training the property predictors, adjust the paths in the dictionary `PROPERTY_PREDICTORS_CKPTS` in `modules/constants.py`.

You can then fine-tune the base model on our defined property predictors. The configuration can be adjusted in `configs/training/grpo.yaml`.
```
    bash scripts/rl/finetune_policy.sh 
```

### Directory structure

```tree
├── data/                               # Datasets
│   ├── aopwin_koh.csv                  # kOH rate constants dataset
│   ├── chedl_thermo_properties.csv     # Tc, Pc, w dataset
│   ├── chen_jaquemin_lfl.csv           # Lower flammability limit dataset
│   ├── gwp_with_smiles.csv             # GWP100 list of 230 molecules from IPCC AR6 report (EPA website)
│   ├── mclinden_refrigerants_with_metrics.csv  # Refrigerant molecules with properties
│   ├── muthiah_rf.csv                  # Radiative efficiency dataset (Muthiah et al 2023)
│   ├── nasa_polynomials.csv            # NASA coefficients dataset (Farina et al 2021)
│   ├── rate_constants.csv              # Rate constants of reactions with OH, NO3, O3 (McGillen et al 2020)
│   └── smi_ted_finetune_data/          # SMI-TED fine-tuning datasets
│       ├── coolprop_thermo_data.csv
│       └── nasa_polynomials/
│           └── 19_to_7_coeff_conversion.py
├── src/
│   ├── predict/                        # Main prediction code
│   │   ├── cop/                        # SMILES to COP prediction
│   │   │   ├── smile_to_cycle_smi_ted.py     # Full SMILES -> Refrigeration Cycle + COP code
│   │   │   └── simulation/                    # Simulate cycle from properties
│   │   │       ├── calculate_saturation_dome.py  # SMILES -> Saturation dome using departure functions
│   │   │       ├── departure.py               # Departure functions from Peng Robinson EOS
│   │   │       └── simulate_vcc_nasa.py       # SMILES to VCC simulation
│   │   └── gwp/                        # GWP prediction
│   │       ├── gwp_pred_smi_ted.py     # Main GWP prediction pipeline
│   │       └── lifetime/               # Lifetime estimation of a molecule
│   │           └── atkinson_gc.py      # Atkinson Group Contribution implementation (Kwok et al 1995)
│   ├── rl/                             # Reinforcement learning components
│   │   └── grpo.py                     # GRPO implementation
│   ├── smi_ted/                        # SMI-TED model components
│   │   ├── args.py                     # Argument parsing for SMI-TED
│   │   ├── dataset_utils/              # Dataset utilities
│   │   ├── finetune_regression.py      # Fine-tuning pipeline with test set
│   │   ├── finetune_regression_no_test.py  # Fine-tuning pipeline without test set
│   │   ├── post_finetune_inference/    # Post fine-tuning inference utilities
│   │   ├── smi_ted_light/              # SMI-TED Light model files
│   │   ├── trainers.py                 # Training utilities with test set
│   │   ├── trainers_no_test.py         # Training utilities without test set
│   │   └── utils.py                    # SMI-TED utilities
│   └── test/                           # Testing code
│       ├── cop/
│       │   ├── analyse_smiles_to_cycle_results.py  # Analyze test results with statistics
│       │   └── test_smiles_to_cycle_smi_ted.py     # Test EOS and predictions on full CoolProp database
│       └── gwp/
│           ├── analyse_predicted_data.py           # Analyze test results with statistics
│           ├── test_atkinson_mcgillen.py           # Test Atkinson kOH predictions on McGillen true dataset
│           └── test_atkinson_smi_ted.py            # Test Atkinson + SMI-TED on filtered molecules
├── models/                             # Model storage
│   ├── base_model.tar.gz               # Compressed base model archive
│   └── models/
│       └── base/
│           └── base-llama-1b/          # Pre-trained Llama 3.2 1B base model
├── scripts/                            # Training and execution scripts
│   ├── compare_comprehensive.py        # Comprehensive comparison scripts
│   ├── compute_molecule_scores.py      # Molecule scoring utilities
│   ├── run_test_atkinson_smi_ted.sh    # Test script for Atkinson method
│   ├── predictors/
│   │   └── finetune_predictor.sh       # Script to fine-tune SMI-TED on property datasets
│   └── rl/
│       └── finetune_policy.sh          # Script for RL fine-tuning of policy
├── configs/                            # Configuration files
│   ├── accelerate.yaml                 # Accelerate configuration
│   └── training/
│       └── grpo.yaml                   # GRPO training configuration
├── modules/                            # Utility modules
│   ├── constants.py                    # Project constants
│   ├── rdkit.py                        # RDKit utilities
│   ├── utils.py                        # General utilities
│   ├── rewards/                        # Reward function implementations
│   │   ├── calculate_inference_metrics.py  # Inference metrics calculation
│   │   ├── cop_reward.py               # COP-based reward function
│   │   ├── diversity_reward.py         # Diversity reward function
│   │   ├── gwp_reward.py               # GWP-based reward function
│   │   ├── inference_loose_tc_reward.py    # Loose critical temperature reward
│   │   ├── lfl_reward.py               # Lower flammability limit reward
│   │   ├── molecular_length_reward.py  # Molecular length reward
│   │   ├── saturation_pressure_reward.py   # Saturation pressure reward
│   │   ├── tc_reward.py                # Critical temperature reward
│   │   └── validity_reward.py          # Molecular validity reward
│   └── rlhf/                           # RLHF utilities
│       ├── entropy.py                  # Entropy calculations
│       ├── preprocessing.py            # Data preprocessing
│       ├── reward_strategies.py        # Reward strategy implementations
│       └── utils.py                    # RLHF utilities
├── main.py                             # Main entry point
├── eval.sh                             # Evaluation script
├── pyproject.toml                      # Project configuration
└── setup.py                            # Setup script
```

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@article{goldszal2025discovery,
  title        = {Discovery of Sustainable Refrigerants through Physics-Informed RL Fine-Tuning of Sequence Models},
  author       = {Adrien Goldszal, Diego Calanzone, Vincent Taboga, Pierre-Luc Bacon},
  year         = {2025},
  eprint       = {2509.19588},
  archivePrefix= {arXiv},
  primaryClass = {physics.chem-ph},
  note         = {Accepted at the 1st EurIPS 2025 Workshop on SIMBIOCHEM},
  url          = {https://arxiv.org/abs/2509.19588}
}
```
