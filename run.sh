# python pre_process_sysu.py # once only


# Source pre-training
# python train_mine.py --dataset sysu --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --run_name 'agw_sysu' --method 'agw'


# Examples of training on mini dataset without SDA:
# python train_mine.py --dataset sysu --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --target_ids 80 --run_name 'agw_sysu' --method 'agw'
# python train_mine.py --dataset regdb --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --target_ids 40 --margin_mmd 1.40 --run_name 'regdb40' --method 'base' --dist_disc 'margin_mmd'


# Examples of training with SDA:
# python train_adaptation.py --aug --dataset regdb --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --target_ids 40 --method 'base' --dist_disc 'margin_mmd' --margin_mmd 1.40 --run_name 'exp_da_regdb40' --source_model_path 'save_model/margin_mmd1.40_sysu_c_tri_pcb_off_w_tri_2.0_share_net3_base_gm10_k4_p4_lr_0.1_seed_0_best.t'
python train_adaptation.py --aug --dataset sysu --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --target_ids 80 --margin_mmd 1.40 --run_name 'exp_da_sysu80' --source_model_path 'save_model/agw_regdb_regdb_c_tri_pcb_off_w_tri_2.0_share_net3_agw_k4_p4_lr_0.1_seed_0_trial_1_best.t' --method 'agw'


# Examples of testing scipts:
# echo "For RegDB IDs: 20"
# python test.py --dataset regdb --num_class 415 --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --run_name 'margin_mmd1.40' --method 'agw' --model_path 'save_model/exp_da_regdb_sweep_ids_regdb_c_tri_pcb_off_w_tri_2.0_share_net3_agw_k4_p4_lr_0.1_seed_0_trial_1_best.t'
# python test.py --dataset regdb --num_class 415 --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --run_name 'margin_mmd1.40' --method 'agw' --tvsearch 'save_model/exp_da_regdb_sweep_ids_regdb_c_tri_pcb_off_w_tri_2.0_share_net3_agw_k4_p4_lr_0.1_seed_0_trial_1_best.t'

# echo "For SYSU IDs: 160"
# python test.py --dataset sysu --num_class 366 --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --run_name 'margin_mmd1.40' --method 'agw' --model_path 'save_model/exp_da_sysu160_sweep_hdmmd_sysu_c_tri_pcb_off_w_tri_2.0_share_net3_agw_k4_p4_lr_0.1_seed_0_best.t'
# python test.py --dataset sysu --num_class 366 --gpu 0 --pcb off --share_net 3 --batch-size 4 --num_pos 4 --run_name 'margin_mmd1.40' --method 'agw' --model_path 'save_model/exp_da_sysu160_sweep_hdmmd_sysu_c_tri_pcb_off_w_tri_2.0_share_net3_agw_k4_p4_lr_0.1_seed_0_best.t' --mode 'indoor'
