import time
import json

import torch
import wandb
import gin


@gin.configurable
class Tracker:
    def __init__(self, project: str, entity: str = "hofvarpnir", args=None):
        self.project = project
        self.entity = entity

    def set_conf(self):
        pass

    def add_histogram(self, tag, data, i):
        pass

    def add_dictionary(self, dict):
        pass

    def add_plot(self, tag, fig):
        pass

    def add_image(self, tag, value, i):
        pass

    def set_summary(self, key, value):
        pass

    def add_scalar(self, tag, value, i):
        pass

    def log_iteration_time(self, batch_size, i):
        pass

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), str(model_path))


@gin.configurable
class WandBTracker(Tracker):
    def __init__(self, project: str, entity: str = "hofvarpnir", args=None):
        super().__init__(entity, project, args)
        wandb.init(entity=entity, project=project, config=args)

    def set_conf(self):
        wandb.config = gin.operative_config_str()

    def add_dictionary(self, dict):
        wandb.log(dict)

    def add_plot(self, tag, fig):
        wandb.log({tag: fig})

    def add_histogram(self, tag, data, i):
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach()
        wandb.log({tag: wandb.Histogram(data)}, step=i)

    def add_scalar(self, tag, value, i):
        wandb.log({tag: value}, step=i)

    def add_image(self, tag, value, i):
        wandb.log({tag: [wandb.Image(value, caption="Label")]}, step=i)

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  # noqa
            self.last_time = time.time()
            if i % 5 == 0:
                self.add_scalar("timings/iterations-per-sec", 1 / dt, i)
                self.add_scalar("timings/samples-per-sec", batch_size / dt, i)
        except AttributeError:
            self.last_time = time.time()

    def set_summary(self, key, value):
        wandb.run.summary[key] = value

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), str(model_path))
        wandb.save(str(model_path))


@gin.configurable
class ConsoleTracker(Tracker):
    def __init__(self, project: str, entity="hofvarpnir", args=None):
        super().__init__(entity, project, args)
        pass

    def add_scalar(self, tag, value, i):
        print(f"{i}  {tag}: {value}")

    def add_dictionary(self, dict):
        print(json.dumps(dict))

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  # noqa
            self.last_time = time.time()
            if i % 10 == 0:
                print(f"{i}  iterations-per-sec: {1/dt}")
                print(f"{i}  samples-per-sec: {batch_size/dt}")
        except AttributeError:
            self.last_time = time.time()

    def set_summary(self, key, value):
        print(f"{key}: {value}")
