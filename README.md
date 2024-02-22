# The Discovery of Binding Modes Rrequires Rethinking Docking Generalization

Here we provide our implementation of the Confidence Bootstrapping method, pretrained diffusion and confidence models, and processed receptors from the DockGen-clusters validation and test sets. 

## Dataset

The Binding MOAD database can be downloaded from [http://www.bindingmoad.org/]. All 189 complexes in the DockGen test set can be found in `data/BindingMOAD_2020_ab_processed_biounit/test_names.npy`, and the 85 complexes from DockGen-clusters can be found in `data/BindingMOAD_2020_ab_processed_biounit/test_names_bootstrapping.npy`. The list of complexes in the DockGen benchmark can also be found at `data/BindingMOAD_2020_ab_processed_biounit/new_cluster_to_ligands.pkl`, which is a dictionary with cluster names as keys and lists of ligand names as values. Complexes from Binding MOAD should be downloaded also to `data/BindingMOAD_2020_ab_processed_biounit`. Here, we also provide the processed receptors at `data/MOAD_new_test_processed` and `data/MOAD_new_val_processed`.

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
    HOME=esm/model_weights python esm/scripts/extract.py esm2_t33_650M_UR50D data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.fasta data/esm2_output --repr_layers 33 --include per_tok
    
Then we convert the embeddings to a single `.pt` file:

    python datasets/esm_embeddings_to_pt.py --esm_embeddings_path data/esm2_output    

## Running finetuning:

After downloading the complexes from Binding MOAD, we can run the Confidence Bootstrapping finetuning on a cluster like `Homo-oligomeric flavin-containing Cys decarboxylases, HFCD`:

    python -m finetune_train --test_sigma_intervals --log_dir workdir --lr 1e-4 --batch_size 5 --ns 32 --nv 6 --scale_by_sigma --dropout 0.1 --sampling_alpha 2 --sampling_beta 1 --remove_hs --c_alpha_max_neighbors 24 --cudnn_benchmark --rot_alpha 1 --rot_beta 1 --tor_alpha 1 --tor_beta 1 --n_epochs 300 --sh_lmax 1 --num_prot_emb_layers 3 --reduce_pseudoscalars --moad_esm_embeddings_sequences_path data/BindingMOAD_2020_ab_processed_biounit/sequences_to_id.fasta --moad_esm_embeddings_path data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.pt --enforce_timesplit --moad_dir data/BindingMOAD_2020_ab_processed_biounit --inference_out_dir results/triple_crop20 --confidence_cutoff -4 --restart_ckpt best_ema_inference_epoch_model --restart_dir workdir/pretrained_score_model --filtering_model_dir workdir/pretrained_confidence_model --max_complexes_per_couple 20 --bootstrapping_train_cluster "Homo-oligomeric flavin-containing Cys decarboxylases, HFCD" --fixed_length 100 --initial_iterations 10 --minimum_t 0.3 --cache_path cache --inference_batch_size 4 --save_model_freq 5 --inference_iterations 4 --val_inference_freq 5 --inference_samples 8 --split test 

Note that the command above is not the same as the one used in experiments in the paper, which also samples random complexes from PDBBind at every bootstrapping step. To reproduce paper results, we need to download the PDBBind dataset:

    1. download it from [zenodo](https://zenodo.org/record/6034088) 
    2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`

Then, we can run the finetuning command with `--keep_original_train` and `--totoal_trainset_size 100` to reproduce paper numbers.

