from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.tensorboard import TensorboardLoggerHook


@HOOKS.register_module()
class TensorboardLoggerHookEpoch(TensorboardLoggerHook):
    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            elif "val" in tag:
                self.writer.add_scalar(tag, val, self.get_epoch(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
