from mmengine.registry import VISBACKENDS
from mmengine.visualization.vis_backend import TensorboardVisBackend


@VISBACKENDS.register_module()
class TensorboardLoggerHookEpoch(TensorboardVisBackend):
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            elif "val" in tag:
                self.writer.add_scalar(tag, val, self.get_epoch(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
