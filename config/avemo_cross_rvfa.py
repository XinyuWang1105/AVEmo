from copy import deepcopy


VIDEO_BACKBONE_NAME = "marlin_vitb"

VIDEO_BACKBONES = {
    "marlin_vitb": {
        "name": "marlin_vitb",
        "model_name": "marlin_vit_base_ytf",
        "checkpoint": "weights/marlin_vit_base_ytf.full.pt",
        "freeze": False,
        "allow_online": False,
        "tubelet_size": 2,
    },
    "clip14": {
        "name": "clip14",
        "model_name": "openai/clip-vit-large-patch14",
        "model_path": "<CLIP_MODEL_DIR>",
        "local_files_only": True,
        "freeze": True,
        "use_ln_tuning": False,
        "use_lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "token_mode": "patch",
        "normalize": True,
        "image_size": 224,
    },
}


CONFIG = {
    "name": "avemo_cross_rvfa",
    "output": "runs",
    "model": {
        "name": "avemo",
        "params": {
            "audio_backbone": {
                "name": "audiomae_vitb",
                "checkpoint": "weights/audiomae_pretrained.pth",
                "input_size": (1024, 128),
                "patch_size": (16, 16),
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "freeze": False,
                "require_pretrained": True,
            },
            "video_backbone": deepcopy(VIDEO_BACKBONES[VIDEO_BACKBONE_NAME]),
            "hidden_dim": 768,
            "num_segments": 8,
            "num_heads": 4,
            "num_layers": 3,
            "dropout": 0.1,
            "ssm_kernel_size": 3,
            "ssm_backend": "mamba",
            "ssm_variant": "mamba",
            "ssm_d_state": 16,
            "ssm_d_conv": 4,
            "ssm_expand": 2,
            "xlstm_backend": "official",
            "xlstm_blocks": 2,
            "xlstm_slstm_at": (1,),
            "xlstm_slstm_backend": "vanilla",
            "xlstm_conv1d_kernel_size": 4,
            "xlstm_qkv_proj_blocksize": 4,
            "binary_only": False,
            "pretrained_path": None,
            "require_pretrained": False,
            "pretrained_prefixes": "consistency",

            "backbone_tuning": {
                "audio": {"strategy": "full"},
            },
        },
    },

    "data": {
        "train": {
            "name": "FakeAVCeleb-Cross-RVFA",
            "csv": "<DATA_ROOT>/processed/fakeavceleb/cross_manipulation/folds/RVFA/train.csv",
            "lrs2_real_balance": {
                "enabled": False,
                "csv": "<DATA_ROOT>/processed/lrs2/processed_training_set.csv",
                "max_samples": "auto",
                "fixed": True,
                "seed": 42,
            },
        },
        "eval": {
            "CrossManipulation-RVFA": "<DATA_ROOT>/processed/fakeavceleb/cross_manipulation/folds/RVFA/test.csv",
        },
        "num_frames": 16,
        "image_size": 224,
        "target_length": 1024,
        "mel_bins": 128,
        "audio_mean": -6.9960,
        "audio_std": 3.1205,
        "audio_cache_dir": None,
        "freq_mask_param": 20,
        "time_mask_param": 100,
        "video_augment": {
            "hflip_prob": 0.5,
            "rotate_prob": 0.5,
            "rotate_limit": (-10, 10),
            "color_prob": 0.5,
            "brightness": 0.1,
            "contrast": 0.1,
            "blur_prob": 0.5,
            "blur_limit": (3, 7),
            "jpeg_prob": 1.0,
            "jpeg_quality": (40, 100),
        }
    },

    "train": {
        "epochs": 50,
        "batch_size": 4,
        "eval_batch_size": 8,
        "grad_accum_steps": 8,
        "num_workers": 2,
        "seed": 42,
        "amp": True,
        "augment": True,
        "weighted_sampling": True,
        "sampler": "binary_subset",
        "real_fake_ratio": (1, 4),
        "drop_last": False,
        "pin_memory": True,
        "grad_clip": 1.0,
        "eval_every": 1,
    },

    "optimizer": {
        "name": "adamw",
        "backbone_lr": 1e-5,
        "adapter_lr": 1e-4,
        "lr": 1e-4,
        "betas": (0.95, 0.999),
        "weight_decay": 3e-2,
    },

    "scheduler": {
        "name": "cosine_warm_restarts",
        "step": "epoch",
        "t0": 50,
        "t_mult": 1,
        "eta_min": 1e-9,
    },

    "loss": {
        "fusion": 1.0,
        "audio": 0.1,
        "video": 0.1,
        "content_consistency": 0.02,
        "temporal_sync": 0.02,
    },

    "save": {
        "datasets": ["CrossManipulation-RVFA"],
        "head": "fusion",
        "metric": "auc",
        "mode": "max",
        "last": True,
    },
}
