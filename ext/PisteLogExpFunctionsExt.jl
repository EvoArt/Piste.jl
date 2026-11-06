# One-node reverse-mode rules for LogExpFunctions.
#
# These functions are written for Float64 with branches and thresholds, so on a
# `TapedReal` they decompose into several nodes AND re-derive their thresholds
# through tracked arithmetic on every call. `log1pexp` is the worst case and the
# commonest: `BernoulliLogit`'s logpdf is `-log1pexp(-x)`, and on a TapedReal
# its generic `_log1pexp_thresholds` evaluates `exponent`, `nextfloat` and a
# `log` per observation before taping EXP and LOG1P. Here each becomes one node
# whose value and derivative are computed once, in Float64, by the package's
# own implementation.
#
# Every rule below is a unary node with its derivative saved at record time,
# which is exactly what OP_MULC is (`adj[a] += ai * c`), so no new op is needed.
# `logistic` reuses OP_LOGISTIC, whose derivative s(1-s) is read off its value.
module PisteLogExpFunctionsExt

using Piste: TapedReal, tape, record!, OP_MULC, OP_LOGISTIC
import LogExpFunctions

@inline _unary(x::TapedReal, v::Float64, dv::Float64) =
    record!(tape(x), OP_MULC, x.idx, Int32(0), v, dv)

# d/dx log(1 + e^x) = logistic(x)
LogExpFunctions.log1pexp(x::TapedReal) =
    _unary(x, LogExpFunctions.log1pexp(x.v), LogExpFunctions.logistic(x.v))

LogExpFunctions.logistic(x::TapedReal) =
    record!(tape(x), OP_LOGISTIC, x.idx, Int32(0), LogExpFunctions.logistic(x.v), 0.0)

# d/dx log(x / (1 - x)) = 1 / (x (1 - x))
LogExpFunctions.logit(x::TapedReal) =
    _unary(x, LogExpFunctions.logit(x.v), 1.0 / (x.v * (1.0 - x.v)))

# d/dx log(1 - e^x) = -1 / expm1(-x), for x < 0
LogExpFunctions.log1mexp(x::TapedReal) =
    _unary(x, LogExpFunctions.log1mexp(x.v), -1.0 / expm1(-x.v))

# d/dx log(e^x - 1) = 1 / (1 - e^-x) = -1 / expm1(-x), for x > 0
LogExpFunctions.logexpm1(x::TapedReal) =
    _unary(x, LogExpFunctions.logexpm1(x.v), -1.0 / expm1(-x.v))

# d/dx log(1 + x^2) = 2x / (1 + x^2)
LogExpFunctions.log1psq(x::TapedReal) =
    _unary(x, LogExpFunctions.log1psq(x.v), 2.0 * x.v / (1.0 + x.v * x.v))

end
