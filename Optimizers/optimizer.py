from Optimizers.adam import Adam
from Optimizers.sgd import SGD
from Optimizers.sgd_momentum import SGDMomentum


# ── Optimizer factories ──────────────────────────────────────────────────────
#
# NOTE: All optimizers read args.learning_rate directly. Schedulers then
# transform that base LR over time.

def adam(params, args):
    return Adam(
        params=params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=getattr(args, "eps", 1e-7),
        weight_decay=args.weight_decay,
    )


def sgd(params, args):
    return SGD(
        params=params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def sgd_momentum(params, args):
    return SGDMomentum(
        params=params,
        lr=args.learning_rate,
        momentum=getattr(args, "momentum", 0.9),
        weight_decay=args.weight_decay,
    )


# ── Registry ─────────────────────────────────────────────────────────────────

optimizers = {
    "adam":          adam,
    "sgd":           sgd,
    "sgd_momentum":  sgd_momentum,
}
