"""
@author: JiXuan Xu, Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""

import torch

from FaceX_Zoo.face_sdk.core.model_loader.BaseModelLoader import (
    BaseModelLoader,
)

# Rebuild RetinaFace from weights instead of moving the legacy pickled module to CUDA.
from models.network_def.retinaface_def import RetinaFace


class FaceDetModelLoader(BaseModelLoader):

    def __init__(
        self,
        model_path,
        model_category,
        model_name,
        meta_file="model_meta.json",
    ):
        super().__init__(
            model_path,
            model_category,
            model_name,
            meta_file,
        )

        self.cfg["min_sizes"] = self.meta_conf[
            "min_sizes"
        ]

        self.cfg["steps"] = self.meta_conf[
            "steps"
        ]

        self.cfg["variance"] = self.meta_conf[
            "variance"
        ]

        self.cfg["in_channel"] = self.meta_conf[
            "in_channel"
        ]

        self.cfg["out_channel"] = self.meta_conf[
            "out_channel"
        ]

        self.cfg["confidence_threshold"] = (
            self.meta_conf[
                "confidence_threshold"
            ]
        )


    def load_model(self):
        """
        The original FaceX-Zoo checkpoint stores the whole PyTorch
        module object.

        That legacy module can be loaded on CPU, but moving the
        deserialized object directly to CUDA may crash with modern
        PyTorch/torchvision.

        Therefore:

        1. Load legacy checkpoint on CPU.
        2. Extract only state_dict.
        3. Rebuild an equivalent RetinaFace using current source code.
        4. Load exactly the same weights.
        5. Return the fresh model.

        Detection configuration, weights and architecture remain
        unchanged.
        """

        # ----------------------------------------------------
        # 1. Load legacy full-model checkpoint on CPU
        # ----------------------------------------------------

        legacy_model = torch.load(
            self.cfg["model_file_path"],
            map_location="cpu",
        )

        # ----------------------------------------------------
        # 2. Extract weights
        # ----------------------------------------------------

        state_dict = legacy_model.state_dict()

        # ----------------------------------------------------
        # 3. Recover the ACTUAL architecture parameters
        #
        # The serialized model may not match model_meta.json exactly.
        # ----------------------------------------------------

        return_layers = dict(
            legacy_model.body.return_layers
        )

        # For this RetinaFace implementation:
        #
        # in_channels_list = [
        #     in_channel * 2,
        #     in_channel * 4,
        #     in_channel * 8,
        # ]
        #
        # Therefore first FPN input channel / 2 gives in_channel.

        first_fpn_conv = (
            legacy_model.fpn.output1[0]
        )

        in_channel = (
            first_fpn_conv.in_channels // 2
        )

        out_channel = (
            first_fpn_conv.out_channels
        )

        rebuild_cfg = {
            "return_layers": return_layers,
            "in_channel": in_channel,
            "out_channel": out_channel,
        }

        phase = getattr(
            legacy_model,
            "phase",
            "test",
        )

        # Preserve original train/eval state exactly.
        was_training = legacy_model.training

        # ----------------------------------------------------
        # 4. Build fresh RetinaFace with current PyTorch classes
        # ----------------------------------------------------

        model = RetinaFace(
            cfg=rebuild_cfg,
            phase=phase,
        )

        # ----------------------------------------------------
        # 5. Load exactly the same weights
        # ----------------------------------------------------

        result = model.load_state_dict(
            state_dict,
            strict=True,
        )

        # Strict=True should already guarantee this.
        if (
            result.missing_keys
            or result.unexpected_keys
        ):
            raise RuntimeError(
                "RetinaFace state_dict mismatch: "
                f"missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )

        # Keep exactly the same mode as legacy checkpoint.
        model.train(
            was_training
        )

        # Legacy object is no longer needed.
        del legacy_model

        print(
            "[INFO] RetinaFace rebuilt from legacy weights: "
            f"in_channel={in_channel}, "
            f"out_channel={out_channel}, "
            f"return_layers={return_layers}"
        )

        return model, self.cfg
