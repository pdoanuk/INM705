# INM705
INM705 - Image analysis

Instruction:
- Setup environment.
- Download data & model weights at: https://drive.google.com/drive/folders/1EvFuqJdnuOYwRGZ-ymwBktcV8zRSu7D9?usp=drive_link
- Following instruction to run program;
- Train command: 
   `CUDA_VISIBILE_DEVICES=0 python run_pipeline_full_options.py --run_mode full --model ViTAD_Fusion_v2`
- Evaluate mode:
  `CUDA_VISIBILE_DEVICES=0 python run_pipeline_full_options.py --run_mode evaluate --model ViTAD_Fusion_v2 --final_checkpoint_path FULL_PATH_TO_CHECK_POINT.pt`
