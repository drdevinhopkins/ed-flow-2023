# neuralprophet_safe_load.py
import torch
import neuralprophet.configure as npc

torch.serialization.add_safe_globals([
    npc.ConfigSeasonality,
    npc.ConfigLaggedRegressors,
    npc.ConfigEvents,
    npc.ConfigFutureRegressors,
])
