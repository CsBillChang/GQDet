"""Optional sample-scheduling controller kept as a lightweight public stub."""


class AFSSController:
    def __init__(self, *args, **kwargs):
        self.states = {}
        self.enabled = False

    def initialize_states(self, total_samples):
        self.states = {idx: {} for idx in range(int(total_samples))}

    def state_dict(self):
        return {"states": self.states}

    def load_state_dict(self, state):
        self.states = dict(state.get("states", {}))

    def level_counts(self):
        return {"easy": 0, "moderate": 0, "hard": len(self.states)}

    def select_indices_for_epoch(self, epoch):
        return list(self.states.keys())

    def mark_selected_indices(self, selected_indices, epoch):
        return None

    def should_refresh_scores(self, epoch):
        return False

    def compute_image_precision_recall(self, *args, **kwargs):
        return 0.0, 0.0

    def compute_image_mask_precision_recall(self, *args, **kwargs):
        return 0.0, 0.0

    def update_image_metrics(self, *args, **kwargs):
        return None
