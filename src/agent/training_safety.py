"""Fail closed on invalid gradients or skipped AMP updates, not just finite loss."""
import math


class UpdateHealth:
    def __init__(self):
        self.successful_updates = 0
        self.skipped_updates = 0
        self.gradient_norms = []

    def observe_update(self, skipped):
        if skipped:
            self.skipped_updates += 1
            raise FloatingPointError('AMP skipped an optimizer update (gradient overflow); run is not healthy')
        self.successful_updates += 1

    def observe_logs(self, logs):
        if 'grad_norm' in logs:
            norm = float(logs['grad_norm'])
            self.gradient_norms.append(norm)
            if not math.isfinite(norm):
                raise FloatingPointError('Nonfinite grad_norm; finite train/eval loss does not prove a valid update')

    def summary(self):
        return {'successful_optimizer_updates': self.successful_updates,
                'skipped_optimizer_updates': self.skipped_updates,
                'logged_gradient_norms': self.gradient_norms}

    def smoke_is_healthy(self, parameter_delta):
        return (self.successful_updates >= 2 and self.skipped_updates == 0
                and len(self.gradient_norms) >= 2
                and all(math.isfinite(x) and x > 0 for x in self.gradient_norms)
                and math.isfinite(parameter_delta) and parameter_delta > 0)


def make_update_callback(base_class, health):
    class UpdateCallback(base_class):
        def on_step_end(self, args, state, control, optimizer=None, **kwargs):
            if optimizer is None or not hasattr(optimizer, 'step_was_skipped'):
                raise RuntimeError('Expected Accelerate optimizer with explicit step_was_skipped status')
            health.observe_update(bool(optimizer.step_was_skipped))

        def on_log(self, args, state, control, logs=None, **kwargs):
            health.observe_logs(logs or {})
    return UpdateCallback()
