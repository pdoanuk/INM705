# INM705
INM705 - Image analysis

Instruction:
- Setup environment. (Pytho=3.9, Torch 2.1.2)
- To install all dependencies:
  ```python
pip install -r ./requirements_inm705_pkgs.txt
    ```
- Download data & model weights at: https://drive.google.com/drive/folders/1EvFuqJdnuOYwRGZ-ymwBktcV8zRSu7D9?usp=drive_link
- Following instruction to run program;
- Train command:
```python
CUDA_VISIBILE_DEVICES=0 python run_pipeline_full_options.py --run_mode full --model ViTAD_Fusion_v2
```
- Evaluate mode:
```python
CUDA_VISIBILE_DEVICES=0 python run_pipeline_full_options.py --run_mode evaluate --model ViTAD_Fusion_v2 --final_checkpoint_path FULL_PATH_TO_CHECK_POINT.pt
```

