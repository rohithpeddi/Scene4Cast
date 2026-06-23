"""
WorldWise Loss — AMWAE triple-objective + (WorldWise-exclusive) tail-aware
logit adjustment.
==========================================================================

WorldWise extends the AMWAE loss with a *tail-aware* objective that the
baselines do not get. This is the loss-side half of WorldWise's exclusive
contribution (the other half being the ego-motion + MWAE modules).

Tail-aware term: **logit adjustment** (Menon et al., 2021). During training we
add a per-class log-prior offset to the relation-head logits, then compute the
usual loss. At inference the model logits are left untouched (the loss never
runs), so rare predicate classes are favoured at test time. This directly
lifts mean-recall (mR@K), which averages recall over predicate classes and is
otherwise dominated by the head classes.

  - attention (single-label, 3 classes): logit += tau * log(pi_c)
  - spatial   (multi-label, 6 classes):  logit += tau * log(pi_c / (1 - pi_c))
  - contacting(multi-label, 17 classes): logit += tau * log(pi_c / (1 - pi_c))

Class priors pi_c are read from a JSON file produced by
``tools/compute_predicate_priors.py``. If the file is missing or logit
adjustment is disabled, this falls back to the plain AMWAE loss with a warning.
"""

import json
import logging
from pathlib import Path

import torch

from lib.supervised.worldsgg.amwae.loss import AMWAELoss

logger = logging.getLogger(__name__)


class WorldWiseLoss(AMWAELoss):
    """AMWAE loss + optional tail-aware logit adjustment (WorldWise-exclusive)."""

    def __init__(
        self,
        *,
        use_logit_adjustment: bool = False,
        logit_adjustment_tau: float = 1.0,
        predicate_priors_path: str = None,
        data_path: str = None,
        **amwae_kwargs,
    ):
        super().__init__(**amwae_kwargs)
        self.use_logit_adjustment = bool(use_logit_adjustment)
        self.logit_adjustment_tau = float(logit_adjustment_tau)
        self._has_priors = False

        if self.use_logit_adjustment:
            self._load_priors(predicate_priors_path, data_path)

    def _load_priors(self, priors_path, data_path):
        """Load predicate priors and precompute the additive logit offsets."""
        if not priors_path:
            logger.warning(
                "[WorldWiseLoss] use_logit_adjustment=True but no "
                "predicate_priors_path given — disabling logit adjustment."
            )
            self.use_logit_adjustment = False
            return

        path = Path(priors_path)
        if data_path and not path.is_absolute():
            path = Path(data_path) / priors_path

        if not path.exists():
            logger.warning(
                f"[WorldWiseLoss] predicate priors not found at {path} — "
                f"disabling logit adjustment. Run tools/compute_predicate_priors.py first."
            )
            self.use_logit_adjustment = False
            return

        with open(path, "r") as f:
            priors = json.load(f)

        eps = 1e-6
        att = torch.tensor(priors["attention"], dtype=torch.float32).clamp(eps, 1 - eps)
        spa = torch.tensor(priors["spatial"], dtype=torch.float32).clamp(eps, 1 - eps)
        con = torch.tensor(priors["contacting"], dtype=torch.float32).clamp(eps, 1 - eps)

        # attention: single-label softmax head → additive log-prior
        # spatial/contacting: multi-label sigmoid heads → additive log-odds
        self.register_buffer("_off_att", torch.log(att))
        self.register_buffer("_off_spa", torch.log(spa) - torch.log1p(-spa))
        self.register_buffer("_off_con", torch.log(con) - torch.log1p(-con))
        self._has_priors = True

        logger.info(
            f"[WorldWiseLoss] tail-aware logit adjustment ON "
            f"(tau={self.logit_adjustment_tau}, priors={path})"
        )

    def forward(self, predictions, *args, **kwargs):
        if self.use_logit_adjustment and self._has_priors:
            tau = self.logit_adjustment_tau
            # Shallow-copy so we don't mutate the model's output dict
            # (eval / distributions must stay on the un-adjusted logits).
            predictions = dict(predictions)
            att = predictions["attention_logits"]
            predictions["attention_logits"] = att + tau * self._off_att.to(att.device, att.dtype)
            spa = predictions["spatial_logits"]
            predictions["spatial_logits"] = spa + tau * self._off_spa.to(spa.device, spa.dtype)
            con = predictions["contacting_logits"]
            predictions["contacting_logits"] = con + tau * self._off_con.to(con.device, con.dtype)

        return super().forward(predictions, *args, **kwargs)
