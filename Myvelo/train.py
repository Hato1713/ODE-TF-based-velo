from copy import deepcopy
from typing import Callable, Mapping
import numpy as np

import torch
from anndata import AnnData
import scvelo as scv
from scvelo import logging as logg
from scipy.stats import pearsonr

from Myvelo.parseconfig import ConfigParser
from Myvelo.model import Trainer
import Myvelo.data as module_data
import Myvelo.module as module_arch

import os
import scipy

# a hack to make constants, see https://stackoverflow.com/questions/3203286
class MetaConstants(type):
    @property
    def default_configs(cls):
        return deepcopy(cls._default_configs)


class Constants(object, metaclass=MetaConstants):
    _default_configs = {
        "name": "InterVelo_project",
        "n_gpu": 1,  # whether to use GPU
        "arch": {
            "type": "InterVELO",
            "args": {
            "n_ode_hidden": 25,
            "n_hidden": 128,
            "n_latent": 20,    
            # "TF_mask" : None,
            # "n_TF": 6,
            "log_variational": False,
            "pred_unspliced": False,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "ode_method": 'euler',
            "step_size": None,
            "alpha_recon_lec": 0.5,
            "alpha_recon_lode": 0.5,
            "loss1_scale": 1.,
            "loss2_scale": 1.,
            "l1reg_loss_scale": 1e-3,
            "alpha_latent_loss_scale": 0.5,
            'gamma_loss_scale': 0.01,
            "dropout_rate": 0.1, 
            "scale1":1,
            "scale2":1,       
            'do_extend_TF': False,
            },
        },
        "data_loader": {
            "type": "VeloDataLoader",
            "args": {
                "batch_size": 1024,
                "shuffle": True,
                "validation_split": 0.1,
                "num_workers": 10,
                "velocity_genes": False,
                "use_scaled_u": False,
            },
        },
        "optimizer": {
            "type": "Adam",
            "args": {"lr": 0.01, "weight_decay": 0.01, "amsgrad": True, "eps":0.01},
        },
        "loss_pearson": {
            "coeff_u": 1.0,
            "coeff_s": 1.0,
        },
        "mask_zeros": False,
        "lr_scheduler": {"type": "StepLR", "args": {"step_size": 1, "gamma": 0.97}},
        "trainer": {
            "check_direction":True,
            "epochs": 100,
            "loss1_epochs": 0,
            "save_dir": "saved/",
            "save_period": 1000,
            "verbosity": 1,
            "early_stop": 10,
            "tensorboard": True,
            "grad_clip": True,
        },
    }

def get_TFs(data, databases, TFdata_path='./', copy=False):
    """
    From TFvelo (Li et al., 2024, https://doi.org/10.1038/s41467-024-45661-w)
    
    Add TFs to each gene(target) in adata.layers['TFs']
    ----------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    copy: `bool` (default: `False`)
        Return a copy of `adata` instead of updating it.

    Returns
    -------
    Returns or updates `adata` depending on `copy`.
    """
    print('Get TFs according to', databases)
    adata = data.copy() if copy else data
    n_gene = adata.shape[1]
    adata.varm['TFs'] = np.full([n_gene, n_gene], 'blank')
    adata.varm['TFs_id'] = np.full([n_gene, n_gene], -1)
    adata.varm['TFs_times'] = np.full([n_gene, n_gene], 0)
    adata.varm['TFs_correlation'] = np.full([n_gene, n_gene], 0.0)
    adata.varm['knockTF_Log2FC'] = np.full([n_gene, n_gene], 0.0)
    adata.var['n_TFs'] = np.zeros(n_gene, dtype=int)
    gene_names = list(adata.var_names)
    all_TFs = []

    if 'all' in databases:
        with open(TFdata_path + "TF_data/TF_names_v_1.01.txt", "r") as f:  
            for line in f.readlines():
                TF_name = line.strip('\n') 
                if not TF_name in gene_names:
                    continue
                if not TF_name in all_TFs:
                    all_TFs.append(TF_name)
                TF_expression = np.ravel(adata[:, TF_name].layers['M_total'])
                for target in gene_names:
                    target_idx = gene_names.index(target)
                    if (target==TF_name):
                        continue
                    if (TF_name in adata.varm['TFs'][target_idx]):
                        ii = list(adata.varm['TFs'][target_idx]).index(TF_name)
                        adata.varm['TFs_times'][target_idx, ii] += 1  
                        continue
                    target_expression = np.ravel(adata[:, target].layers['M_total'])
                    flag = (TF_expression>0.1) & (target_expression>0.1)
                    if flag.sum() < 2:
                        correlation = 0
                    else:
                        correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])
                    tmp_n_TF = adata.var['n_TFs'][target_idx]
                    adata.varm['TFs'][target_idx][tmp_n_TF] = TF_name
                    adata.varm['TFs_id'][target_idx][tmp_n_TF] = gene_names.index(TF_name)
                    adata.varm['TFs_times'][target_idx, tmp_n_TF] = 1 
                    adata.varm['TFs_correlation'][target_idx, tmp_n_TF] = correlation
                    adata.var['n_TFs'][target_idx] += 1 
        f.close()

    else:
        # if 'knockTF' in databases:
        #     processd_knockTF_path='knockTF/processed/'
        #     TF_files = os.listdir(processd_knockTF_path)
        #     for TF_file in TF_files:
        #         TF_name = TF_file.replace('.txt', '')
        #         if not TF_name in gene_names:
        #             continue
        #         if not TF_name in all_TFs:
        #             all_TFs.append(TF_name)
        #         TF_expression = np.ravel(adata[:, TF_name].layers['M_total'])
        #         with open(processd_knockTF_path+TF_file, "r") as f:  
        #             for line in f.readlines():
        #                 line = line.strip('\n')  
        #                 target, knockTF_Log2FC = line.split('\t')
        #                 target = target.upper()
        #                 knockTF_Log2FC = float(knockTF_Log2FC)
        #                 if target in gene_names:
        #                     target_idx = gene_names.index(target)
        #                     if (target==TF_name):
        #                         continue
        #                     if (TF_name in adata.varm['TFs'][target_idx]):
        #                         ii = list(adata.varm['TFs'][target_idx]).index(TF_name)
        #                         if (adata.varm['knockTF_Log2FC'][target_idx, ii]) * knockTF_Log2FC < 0:
        #                             adata.varm['knockTF_Log2FC'][target_idx, ii] = 0
        #                         adata.varm['TFs_times'][target_idx, ii] += 1  
        #                         continue
        #                     target_expression = np.ravel(adata[:, target].layers['M_total'])
        #                     flag = (TF_expression>0.1) & (target_expression>0.1)
        #                     if flag.sum() < 10:
        #                         correlation = 0
        #                         continue
        #                     else:
        #                         correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])
        #                     tmp_n_TF = adata.var['n_TFs'][target_idx]
        #                     adata.varm['TFs'][target_idx][tmp_n_TF] = TF_name
        #                     adata.varm['TFs_id'][target_idx][tmp_n_TF] = gene_names.index(TF_name)
        #                     adata.varm['TFs_times'][target_idx, tmp_n_TF] = 1 
        #                     adata.varm['knockTF_Log2FC'][target_idx, tmp_n_TF] = knockTF_Log2FC
        #                     adata.varm['TFs_correlation'][target_idx, tmp_n_TF] = correlation
        #                     adata.var['n_TFs'][target_idx] += 1 
        #             f.close()


        if 'ENCODE' in databases:
            processd_ENCODE_path= TFdata_path + 'TF_data/ENCODE/processed/'
            TF_files = os.listdir(processd_ENCODE_path)
            for TF_file in TF_files:
                TF_name = TF_file.replace('.txt', '')
                if not TF_name in gene_names:
                    continue
                if not TF_name in all_TFs:
                    all_TFs.append(TF_name)
                TF_expression = np.ravel(adata[:, TF_name].layers['M_total'])
                with open(processd_ENCODE_path+TF_file, "r") as f:  
                    for line in f.readlines():
                        line = line.strip('\n')  
                        target = line.upper()
                        if target in gene_names:
                            target_idx = gene_names.index(target)
                            if (target==TF_name):
                                continue
                            if (TF_name in adata.varm['TFs'][target_idx]):
                                ii = list(adata.varm['TFs'][target_idx]).index(TF_name)
                                adata.varm['TFs_times'][target_idx, ii] += 1  
                                continue
                            target_expression = np.ravel(adata[:, target].layers['M_total'])
                            flag = (TF_expression>0.1) & (target_expression>0.1)
                            if flag.sum() < 2:
                                correlation = 0
                            else:
                                correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])
                            tmp_n_TF = adata.var['n_TFs'][target_idx]
                            adata.varm['TFs'][target_idx][tmp_n_TF] = TF_name
                            adata.varm['TFs_id'][target_idx][tmp_n_TF] = gene_names.index(TF_name)
                            adata.varm['TFs_times'][target_idx, tmp_n_TF] = 1 
                            adata.varm['TFs_correlation'][target_idx, tmp_n_TF] = correlation
                            adata.var['n_TFs'][target_idx] += 1 
                    f.close()

        if 'ChEA' in databases:
            print(os.getcwd())
            with open(TFdata_path + 'TF_data/ChEA/ChEA_2016.txt', "r") as f:  
                for line in f.readlines():
                    line = line.strip('\n')  
                    line = line.split('\t')
                    TF_info = line[0]
                    TF_name = TF_info.split(' ')[0]
                    if not TF_name in gene_names:
                        continue
                    if not TF_name in all_TFs:
                        all_TFs.append(TF_name)
                    TF_expression = np.ravel(adata[:, TF_name].layers['M_total'])
                    targets = line[2:]
                    for target in targets:
                        if target in gene_names:
                            target_idx = gene_names.index(target)
                            if (target==TF_name):
                                continue
                            if (TF_name in adata.varm['TFs'][target_idx]):
                                ii = list(adata.varm['TFs'][target_idx]).index(TF_name)
                                adata.varm['TFs_times'][target_idx, ii] += 1  
                                continue
                            target_expression = np.ravel(adata[:, target].layers['M_total'])
                            flag = (TF_expression>0.1) & (target_expression>0.1)
                            if flag.sum() < 2:
                                correlation = 0
                            else:
                                correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])
                            tmp_n_TF = adata.var['n_TFs'][target_idx]
                            adata.varm['TFs'][target_idx][tmp_n_TF] = TF_name
                            adata.varm['TFs_id'][target_idx][tmp_n_TF] = gene_names.index(TF_name)
                            adata.varm['TFs_times'][target_idx, tmp_n_TF] = 1 
                            adata.varm['TFs_correlation'][target_idx, tmp_n_TF] = correlation
                            adata.var['n_TFs'][target_idx] += 1 
                f.close()


    adata.uns['all_TFs'] = all_TFs
    max_n_TF = adata.var['n_TFs'].max()
    adata.varm['TFs'] = adata.varm['TFs'][:,:max_n_TF]
    adata.varm['TFs_id'] = adata.varm['TFs_id'][:,:max_n_TF]
    adata.varm['TFs_times'] = adata.varm['TFs_times'][:,:max_n_TF]
    adata.varm['TFs_correlation'] = adata.varm['TFs_correlation'][:,:max_n_TF]
    adata.varm['knockTF_Log2FC'] = adata.varm['knockTF_Log2FC'][:,:max_n_TF]
    print('max_n_TF:', max_n_TF)
    print('mean_n_TF:', np.mean(adata.var['n_TFs']))
    print('gene num of 0 TF:', (adata.var['n_TFs']==0).sum())
    print('total num of TFs:', len(all_TFs))

    count_0, count_total = 0, 0
    # if 'knockTF' in databases:
    #     for i in range(n_gene):
    #         tmp_Log2FC = adata.varm['knockTF_Log2FC'][i, 0:adata.var['n_TFs'][i]]
    #         count_0 += (tmp_Log2FC == 0).sum()
    #         count_total += len(tmp_Log2FC)
    #     print(count_0, count_total, count_0/count_total)
    
    return adata if copy else None
                
def train(
    adata: AnnData,
    inputdata: torch.Tensor,
    configs: Mapping,
    verbose: bool = False,
    return_kinetic_rates: bool = True,
    callback: Callable = None,
    TF_databases = 'all',
    TFdata_path='./',
    debug=False,
    **kwargs,
):
    # adata.var_names = [name.upper() for name in adata.var_names]
    n_cells, n_genes = adata.layers["Ms"].shape
    print('pwd: ' + os.getcwd())
    # get TF mask matrix
    adata.var_names_make_unique()
    get_TFs(adata, databases=TF_databases, TFdata_path=TFdata_path, copy=False)
    
    TF_names_all = sorted(list(set(adata.varm['TFs'].flatten())))
    TF_names = [TF_name for TF_name in TF_names_all if TF_name in adata.var_names]
    adata.uns['valid_TF_names'] = TF_names
    
    print(f'{len(TF_names_all) - len(TF_names)} genes in TF list not included in adata')
    TF_mask = np.zeros((n_genes, len(TF_names)))
    
    TF_name_idx_mapping = {name: i for i, name in enumerate(TF_names)}
    print(TF_name_idx_mapping.keys())
    for i_row, row_TFmatrix in enumerate(adata.varm['TFs']):
        for TF in row_TFmatrix:
            if TF != 'blank' and TF in TF_name_idx_mapping: 
                TF_mask[i_row, TF_name_idx_mapping[TF]] = 1
            else:
                # print(f"Skipping TF: {TF}, not found in mapping.")
                pass
    TF_mask_tensor = torch.from_numpy(TF_mask).float()
    
    ### Original
    if configs["data_loader"]["args"]["velocity_genes"]:
        n_genes = int(np.sum(adata.var["velocity_genes"]))
    configs["arch"]["args"]["n_genes"] = n_genes 
    # configs["arch"]["args"]["TF_mask"] = TF_mask_tensor
    config = ConfigParser(configs)
    logger = config.get_logger("train")

    # setup data_loader instances, use adata as the data_source to load inmemory data
    data_loader = config.init_obj("data_loader", module_data, data_source=adata, inputdata=inputdata)
    valid_data_loader = data_loader.split_validation()
    
    model = config.init_obj(
        "arch", 
        module_arch, 
        n_input=inputdata.shape[1], 
        TF_mask=TF_mask_tensor, 
        n_TF=len(TF_names),
    )
    logger.info(f"Beginning training of {configs['name']} ...")
    if verbose:
        logger.info(configs)
        logger.info(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    def callback_wrapper(epoch):
        # evaluate all and return the velocity matrix (cells, features)
        config_copy = configs["data_loader"]["args"].copy()
        config_copy.update(shuffle=False, validation_split=0, training=False, data_source=adata, inputdata=inputdata)
        eval_loader = getattr(module_data, configs["data_loader"]["type"])(
            **config_copy
        )
        velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, vector_field, *debug_params = trainer.eval(
            eval_loader, return_kinetic_rates=return_kinetic_rates
        )
        alpha_rates_latent = debug_params[0]
        alpha_rates_TF = debug_params[1]

        if callback is not None:
            callback(adata, velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, epoch)
        else:
            logg.warn(
                "Set verbose to True but no callback function provided. A possible "
                "callback function accepts at least two arguments: adata, velo_mat "
            )

    if verbose:
        trainer.train_with_epoch_callback(
            callback=callback_wrapper,
            freq=kwargs.get("freq", 30),
        )
    else:
        trainer.train()

    
    config_copy = configs["data_loader"]["args"].copy()
    config_copy.update(shuffle=False, validation_split=0, training=False, data_source=adata, inputdata=inputdata)
    eval_loader = getattr(module_data, configs["data_loader"]["type"])(
            **config_copy
        )
    velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, vector_field, *debug_params = trainer.eval(
        eval_loader, return_kinetic_rates=return_kinetic_rates
    )
    alpha_rates_latent = debug_params[0]
    alpha_rates_TF = debug_params[1]
    
    print("velo_mat shape:", velo_mat.shape)
    # add velocity
    if configs["data_loader"]["args"]["velocity_genes"]:
        # the predictions only contain the velocity genes
        velocity_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
        idx = adata.var["velocity_genes"].values
        velocity_[:, idx] = velo_mat
        if len(velo_mat_u) > 0:
            velocity_u = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
            velocity_u[:, idx] = velo_mat_u
    else:
        velocity_ = velo_mat
        velocity_u = velo_mat_u

    assert adata.layers["Ms"].shape == velocity_.shape
    adata.layers["velocity"] = velocity_  # (cells, genes)
    adata.obs["pseudotime"] = pseudotime
    adata.obsm['X_TNODE'] = mix_z
    adata.obsm['X_VF']= vector_field
    if len(velo_mat_u) > 0:
        adata.layers["velocity_unspliced"] = velocity_u
        logg.hint(f"added 'velocity_unspliced' (adata.layers)")

    logg.hint(f"added 'velocity' (adata.layers)")
    logg.hint(f"added 'pseudotime'(adata.obs)")
    logg.hint(f"added 'X_TNODE'(adata.obsm)")
    logg.hint(f"added 'X_VF'(adata.obsm)")

    print(f'return_kinetic_rates: {return_kinetic_rates}')
    if return_kinetic_rates:
        if configs["arch"]["args"]["pred_unspliced"]:
            if configs["data_loader"]["args"]["velocity_genes"]:
                alpha_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                alpha_[:, adata.var["velocity_genes"].values] = alpha_rates
            else:
                alpha_= alpha_rates
            adata.layers['pred_alpha'] = alpha_
            # adata.uns['alpha_latent'] = alpha_rates_latent
            logg.hint(f"added 'pred_alpha'(adata.layers)")
        for k, v in kinetic_rates.items():
            if v is not None:
                if configs["data_loader"]["args"]["velocity_genes"]:
                    v_ = np.zeros(adata.shape, dtype=v.dtype)
                    v_[adata.var["velocity_genes"].values] = v
                    v = v_
                adata.var["pred_" + k] = v
                logg.hint(f"added 'pred_{k}' (adata.var)")

    if configs["trainer"]["check_direction"]:
        scv.tl.velocity_graph(adata, n_jobs=10)
        scv.tl.velocity_pseudotime(adata)
        logg.hint("added 'velocity_pseudotime'(adata.obs)")

    
        A = adata.obs["pseudotime"]
        B = adata.obs["velocity_pseudotime"]
        correlation = np.corrcoef(A, B)[0, 1]

        if correlation < 0:
            logg.hint("Train again to correct direction of pseudotime.")
            
            trainer.model.scale1 = torch.nn.Parameter(-trainer.model.scale1)
            trainer.model.scale2 = torch.nn.Parameter(-trainer.model.scale2)
            if verbose:
                trainer.train_with_epoch_callback(
                    callback=callback_wrapper,
                    freq=kwargs.get("freq", 30),
                )
            else:
                trainer.train()

      
            config_copy = configs["data_loader"]["args"].copy()
            config_copy.update(shuffle=False, validation_split=0, training=False, data_source=adata, inputdata=inputdata)
            eval_loader = getattr(module_data, configs["data_loader"]["type"])(
                **config_copy
            )
            velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, mix_z, vector_field, *debug_params = trainer.eval(
                eval_loader, return_kinetic_rates=return_kinetic_rates
            )
            alpha_rates_latent = debug_params[0]
            alpha_rates_TF = debug_params[1]

            print("velo_mat shape:", velo_mat.shape)
            # add velocity
            if configs["data_loader"]["args"]["velocity_genes"]:
                # the predictions only contain the velocity genes
                velocity_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                idx = adata.var["velocity_genes"].values
                velocity_[:, idx] = velo_mat
                if len(velo_mat_u) > 0:
                    velocity_u = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                    velocity_u[:, idx] = velo_mat_u
            else:
                velocity_ = velo_mat
                velocity_u = velo_mat_u

            assert adata.layers["Ms"].shape == velocity_.shape
            adata.layers["velocity"] = velocity_  # (cells, genes)
            adata.obs["pseudotime"] = pseudotime
            adata.obsm['X_TNODE'] = mix_z
            adata.obsm['X_VF'] = vector_field
            if len(velo_mat_u) > 0:
                adata.layers["velocity_unspliced"] = velocity_u
                logg.hint(f"added 'velocity_unspliced' (adata.layers)")
                num_columns = adata.layers["velocity"].shape[1]
                correlations = []
                for i in range(num_columns):
                    corr, _ = pearsonr(adata.layers["velocity"][:, i], adata.layers["velocity_unspliced"][:, i])
                    correlations.append(corr)
                correlation2 = np.mean(correlations)
                if correlation2 < 0:
                    logg.hint(f"the correlation of 'velocity_unspliced' and 'velocity' is negative, consider to reverse 'velocity_unspliced'")

            logg.hint(f"added 'velocity' (adata.layers)")
            logg.hint(f"added 'pseudotime'(adata.obs)")
            logg.hint(f"added 'X_TNODE'(adata.obsm)")
            logg.hint(f"added 'X_VF'(adata.obsm)")

            if return_kinetic_rates:
                if configs["arch"]["args"]["pred_unspliced"]:
                    if configs["data_loader"]["args"]["velocity_genes"]:
                        alpha_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                        alpha_[:, adata.var["velocity_genes"].values] = alpha_rates
                    else:
                        alpha_= alpha_rates
                    adata.layers['pred_alpha'] = alpha_
                    # adata.uns['alpha_latent'] = alpha_rates_latent
                    logg.hint(f"added 'pred_alpha'(adata.layers)")
                for k, v in kinetic_rates.items():
                    if v is not None:
                        if configs["data_loader"]["args"]["velocity_genes"]:
                            v_ = np.zeros(adata.shape, dtype=v.dtype)
                            v_[adata.var["velocity_genes"].values] = v
                            v = v_
                        adata.var["pred_" + k] = v
                        logg.hint(f"added 'pred_{k}' (adata.var)")

            scv.tl.velocity_graph(adata, n_jobs=10)
            scv.tl.velocity_pseudotime(adata)
            logg.hint(f"added 'velocity_pseudotime'(adata.obs)")
    if debug: 
        adata.uns['alpha_latent'] = alpha_rates_latent  
        adata.uns['alpha_TF'] = alpha_rates_TF

    logg.hint(f"model scale1: {trainer.model.scale1}")
    logg.hint(f"model scale2: {trainer.model.scale2}")
    return trainer