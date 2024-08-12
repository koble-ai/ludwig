# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import os

from ludwig.api_annotations import PublicAPI
from ludwig.callbacks import Callback
from ludwig.utils.package_utils import LazyLoader

wandb = LazyLoader("wandb", globals(), "wandb")

logger = logging.getLogger(__name__)


@PublicAPI
class WandbCallback(Callback):
    """Class that defines the methods necessary to hook into process."""
    run = None
    default = True

    def __init__(self, run):
        if run:
            self.run = run
            self.default = False

    def on_train_init(
        self,
        base_config,
        experiment_directory,
        experiment_name,
        model_name,
        output_directory,
        resume_directory,
    ):
        logger.info("wandb.on_train_init() called...")
        if not self.run:
            self.run = wandb.init(
                project=os.getenv("WANDB_PROJECT", experiment_name),
                name=model_name,
                sync_tensorboard=True,
                dir=output_directory,
            )
        self.run.save(os.path.join(experiment_directory, "*"))


    def on_train_start(self, model, config, *args, **kwargs):
        logger.info("wandb.on_train_start() called...")
        config = config.copy()
        self.run.config.update(config)

    def on_eval_end(self, trainer, progress_tracker, save_path):
        """Called from ludwig/models/model.py."""
        for key, value in progress_tracker.log_metrics().items():
            self.run.log({key: value})

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        """Called from ludwig/models/model.py."""
        for key, value in progress_tracker.log_metrics().items():
            self.run.log({key: value})

    def on_visualize_figure(self, fig):
        logger.info("wandb.on_visualize_figure() called...")
        if self.run:
            self.run.log({"figure": fig})

    def on_train_end(self, output_directory):
        if not self.default:
            wandb.finish()
