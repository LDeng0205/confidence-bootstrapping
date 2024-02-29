# DEEP CONFIDENT STEPS TO NEW POCKETS: STRATEGIES FOR DOCKING GENERALIZATION


Here we provide our implementation of the Confidence Bootstrapping method, pretrained diffusion and confidence models, and processed receptors from the DockGen validation and test sets. 

## Dataset

The DockGen benchmark can be downloaded from Zenodo: https://zenodo.org/records/10656052. All 189 complexes in the DockGen test set can be found in `data/BindingMOAD_2020_processed/test_names.npy`, and the 85 complexes from DockGen-clusters can be found in `data/BindingMOAD_2020_processed/test_names_bootstrapping.npy`. The list of complexes in the DockGen benchmark can also be found at `data/BindingMOAD_2020_processed/new_cluster_to_ligands.pkl`, which is a dictionary with cluster names as keys and lists of ligand names as values. Complexes from Binding MOAD should be downloaded also to `data/BindingMOAD_2020_processed`. Here, we also provide the processed receptors at `data/MOAD_new_test_processed` and `data/MOAD_new_val_processed`.

## Setup

We will set up the environment with anaconda [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), and have provided an `environment.yml` file. While in the project directory, run
    
    conda env create

Activate the environment

    conda activate confidence-bootstrapping

## ESM Embeddings

In order to run the diffusion model, we need to generate ESM2 embeddings for complexes in Binding MOAD. First we prepare sequences:

    python datasets/moad_lm_embedding_preparation.py --data_dir data/MOAD_new_test_processed

Then, we install esm and generate embeddings for the test proteins:
    
    git clone https://github.com/facebookresearch/esm
    cd esm
    pip install -e .
    cd ..
    HOME=esm/model_weights python esm/scripts/extract.py esm2_t33_650M_UR50D data/BindingMOAD_2020_processed/moad_sequences_new.fasta data/esm2_output --repr_layers 33 --include per_tok
    
Then we convert the embeddings to a single `.pt` file:

    python datasets/esm_embeddings_to_pt.py --esm_embeddings_path data/esm2_output    

## Running finetuning:

After downloading the complexes from Binding MOAD, we can run the Confidence Bootstrapping finetuning on a cluster like `Homo-oligomeric flavin-containing Cys decarboxylases, HFCD`:

    python -m finetune_train --sampling_alpha 1 --sampling_beta 1 --cudnn_benchmark --cb_inference_freq 5 --num_inference_complexes 100 --use_ema --n_epochs 10 --inference_samples 8 --moad_esm_embeddings_sequences_path TODO --moad_esm_embeddings_path TODO --moad_dir TODO --confidence_cutoff -4 --pretrain_ckpt best_ema_inference_epoch_model --pretrain_dir workdir/pretrained_score --filtering_ckpt best_model.pt --filtering_model_dir workdir/pretrained_confidence --max_complexes_per_couple 20 --cb_cluster "Molybdenum cofactor biosynthesis proteins" --fixed_length 100 --initial_iterations 5 --minimum_t 0 --cache_path cache --inference_batch_size 4 --save_model_freq 25 --split test --inference_iterations 4 --buffer_sampling_alpha 2 --buffer_sampling_beta 1


Note that the command above is not the same as the one used in experiments in the paper, which also samples random complexes from PDBBind at every bootstrapping step. To reproduce paper results, we need to download the PDBBind dataset:

    1. download it from [zenodo](https://zenodo.org/record/6034088) 
    2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`

Then, we can run the finetuning command with `--keep_original_train` and `--totoal_trainset_size 100` to reproduce paper numbers.

