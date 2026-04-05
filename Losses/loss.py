import torch.nn.functional as F


def qa_nll_loss(p1, p2, y1, y2):
    """Standard QA span loss.
    Expects p1/p2 to be raw logits and applies log_softmax before nll."""
    p1 = F.log_softmax(p1, dim=1)
    p2 = F.log_softmax(p2, dim=1)
    return F.nll_loss(p1, y1) + F.nll_loss(p2, y2)


def qa_ce_loss(p1, p2, y1, y2):
    """QA span loss using cross-entropy.
    Expects p1/p2 to be raw logits (no softmax applied)."""
    return F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)


losses = {
    "qa_nll": qa_nll_loss,
    "qa_ce":  qa_ce_loss,
}
