import sympy
import collections

from sympy.core.function import AppliedUndef, Derivative
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.strategies.core import switch
from sympy.core.compatibility import reduce
from sympy import factor
from sympy import Symbol
from sympy import pi
from sympy import sin
from sympy import cos

from contextlib import contextmanager

from mpmath.libmp.backend import basestring
from sympy import latex, symbols
from sympy.parsing.latex import parse_latex


def Rule(name, props=""):
    # GOTCHA: namedtuple class name not considered!
    def __eq__(self, other):
        return self.__class__ == other.__class__ and tuple.__eq__(self, other)

    __neq__ = lambda self, other: not __eq__(self, other)
    cls = collections.namedtuple(name, props + " context symbol")
    cls.__eq__ = __eq__
    cls.__ne__ = __neq__
    return cls


def functionnames(numterms):
    if numterms == 2:
        return ["f", "g"]
    elif numterms == 3:
        return ["f", "g", "h"]
    else:
        return ["f_{}".format(i) for i in range(numterms)]


def replace_u_var(rule, old_u, new_u):
    d = rule._asdict()
    for field, val in d.items():
        if isinstance(val, sympy.Basic):
            d[field] = val.subs(old_u, new_u)
        elif isinstance(val, tuple):
            d[field] = replace_u_var(val, old_u, new_u)
        elif isinstance(val, list):
            result = []
            for item in val:
                if isinstance(item, tuple):
                    result.append(replace_u_var(item, old_u, new_u))
                else:
                    result.append(item)
            d[field] = result
    return rule.__class__(**d)


# def replace_all_u_vars(rule, replacements=None):
#     if replacements is None:
#         replacements = []

#     d = rule._asdict()
#     for field, val in d.items():
#         if isinstance(val, sympy.Basic):
#             for dummy in val.find(sympy.Dummy):
#                 replacements.append((dummy, ))
#         elif isinstance(val, tuple):
#             pass
#     return rule.__class__(**d)

class Printer(object):
    def __init__(self):
        self.lines = []
        self.level = 0

    def append(self, text):
        self.lines.append(self.level * "\t" + text)

    def finalize(self):
        return "\n".join(self.lines)

    def format_math(self, math):
        return str(math)

    def format_math_display(self, math):
        return self.format_math(math)

    @contextmanager
    def new_level(self):
        self.level += 1
        yield self.level
        self.level -= 1

    @contextmanager
    def new_step(self):
        yield self.level
        self.lines.append('\n')


class LaTeXPrinter(Printer):
    def format_math(self, math):
        return latex(math)


class HTMLPrinterP(LaTeXPrinter):
    def __init__(self):
        super(HTMLPrinterP, self).__init__()
        self.lines = ['<ol>']

    def format_math(self, math):
        return '<script type="math/tex; mode=inline">{}</script>'.format(
            latex(math))

    def format_math_display(self, math):
        if not isinstance(math, basestring):
            math = latex(math)
        return '<script type="math/tex; mode=display">{}</script>'.format(
            math)

    @contextmanager
    def new_level(self):
        self.level += 1
        self.lines.append(' ' * 4 * self.level + '<ol>')
        yield
        self.lines.append(' ' * 4 * self.level + '</ol><br/>')
        self.level -= 1

    @contextmanager
    def new_step(self):
        self.lines.append(' ' * 4 * self.level + '<li>')
        yield self.level
        self.lines.append(' ' * 4 * self.level + '</li><br>')

    @contextmanager
    def new_collapsible(self):
        self.lines.append(' ' * 4 * self.level + '<div class="collapsible">')
        yield self.level
        self.lines.append(' ' * 4 * self.level + '</div>')

    @contextmanager
    def new_u_vars(self):
        self.u, self.du = sympy.Symbol('u'), sympy.Symbol('du')
        yield self.u, self.du

    def append(self, text):
        self.lines.append(' ' * 4 * (self.level + 1) + '<p>{}</p>'.format(text))

    def append_header(self, text):
        self.lines.append(' ' * 4 * (self.level + 1) + '<h2>{}</h2>'.format(text))


########################################################3

def Rule(name, props=""):
    return collections.namedtuple(name, props + " context symbol")


ConstantRule = Rule("ConstantRule", "number")
ConstantTimesRule = Rule("ConstantTimesRule", "constant other substep")
PowerRule = Rule("PowerRule", "base exp")
AddRule = Rule("AddRule", "substeps")
MulRule = Rule("MulRule", "terms substeps")
DivRule = Rule("DivRule", "numerator denominator numerstep denomstep")
ChainRule = Rule("ChainRule", "substep inner u_var innerstep")
TrigRule = Rule("TrigRule", "f")
ExpRule = Rule("ExpRule", "f base")
LogRule = Rule("LogRule", "arg base")
FunctionRule = Rule("FunctionRule")
AlternativeRule = Rule("AlternativeRule", "alternatives")
DontKnowRule = Rule("DontKnowRule")
RewriteRule = Rule("RewriteRule", "rewritten substep")

DerivativeInfo = collections.namedtuple('DerivativeInfo', 'expr symbol')

evaluators = {}


def evaluates(rule):
    def _evaluates(func):
        func.rule = rule
        evaluators[rule] = func
        return func

    return _evaluates


def power_rule(derivative):
    expr, symbol = derivative.expr, derivative.symbol
    base, exp = expr.as_base_exp()

    if not base.has(symbol):
        if isinstance(exp, sympy.Symbol):
            return ExpRule(expr, base, expr, symbol)
        else:
            u = sympy.Dummy()
            f = base ** u
            return ChainRule(
                ExpRule(f, base, f, u),
                exp, u,
                diff_steps(exp, symbol),
                expr, symbol
            )
    elif not exp.has(symbol):
        if isinstance(base, sympy.Symbol):
            return PowerRule(base, exp, expr, symbol)
        else:
            u = sympy.Dummy()
            f = u ** exp
            return ChainRule(
                PowerRule(u, exp, f, u),
                base, u,
                diff_steps(base, symbol),
                expr, symbol
            )
    else:
        return DontKnowRule(expr, symbol)


def add_rule(derivative):
    expr, symbol = derivative.expr, derivative.symbol
    return AddRule([diff_steps(arg, symbol) for arg in expr.args],
                   expr, symbol)


def constant_rule(derivative):
    expr, symbol = derivative.expr, derivative.symbol
    return ConstantRule(expr, expr, symbol)


def mul_rule(derivative):
    expr, symbol = derivative
    terms = expr.args
    is_div = 1 / sympy.Wild("denominator")

    coeff, f = expr.as_independent(symbol)

    if coeff != 1:
        return ConstantTimesRule(coeff, f, diff_steps(f, symbol), expr, symbol)

    numerator, denominator = expr.as_numer_denom()
    if denominator != 1:
        return DivRule(numerator, denominator,
                       diff_steps(numerator, symbol),
                       diff_steps(denominator, symbol), expr, symbol)

    return MulRule(terms, [diff_steps(g, symbol) for g in terms], expr, symbol)


def trig_rule(derivative):
    expr, symbol = derivative
    arg = expr.args[0]

    default = TrigRule(expr, expr, symbol)
    if not isinstance(arg, sympy.Symbol):
        u = sympy.Dummy()
        default = ChainRule(
            TrigRule(expr.func(u), expr.func(u), u),
            arg, u, diff_steps(arg, symbol),
            expr, symbol)

    if isinstance(expr, (sympy.sin, sympy.cos)):
        return default
    elif isinstance(expr, sympy.tan):
        f_r = sympy.sin(arg) / sympy.cos(arg)

        return AlternativeRule([
            default,
            RewriteRule(f_r, diff_steps(f_r, symbol), expr, symbol)
        ], expr, symbol)
    elif isinstance(expr, sympy.csc):
        f_r = 1 / sympy.sin(arg)

        return AlternativeRule([
            default,
            RewriteRule(f_r, diff_steps(f_r, symbol), expr, symbol)
        ], expr, symbol)
    elif isinstance(expr, sympy.sec):
        f_r = 1 / sympy.cos(arg)

        return AlternativeRule([
            default,
            RewriteRule(f_r, diff_steps(f_r, symbol), expr, symbol)
        ], expr, symbol)
    elif isinstance(expr, sympy.cot):
        f_r_1 = 1 / sympy.tan(arg)
        f_r_2 = sympy.cos(arg) / sympy.sin(arg)
        return AlternativeRule([
            default,
            RewriteRule(f_r_1, diff_steps(f_r_1, symbol), expr, symbol),
            RewriteRule(f_r_2, diff_steps(f_r_2, symbol), expr, symbol)
        ], expr, symbol)
    else:
        return DontKnowRule(f, symbol)


def exp_rule(derivative):
    expr, symbol = derivative
    exp = expr.args[0]
    if isinstance(exp, sympy.Symbol):
        return ExpRule(expr, sympy.E, expr, symbol)
    else:
        u = sympy.Dummy()
        f = sympy.exp(u)
        return ChainRule(ExpRule(f, sympy.E, f, u),
                         exp, u, diff_steps(exp, symbol), expr, symbol)


def log_rule(derivative):
    expr, symbol = derivative
    arg = expr.args[0]
    if len(expr.args) == 2:
        base = expr.args[1]
    else:
        base = sympy.E
        if isinstance(arg, sympy.Symbol):
            return LogRule(arg, base, expr, symbol)
        else:
            u = sympy.Dummy()
            return ChainRule(LogRule(u, base, sympy.log(u, base), u),
                             arg, u, diff_steps(arg, symbol), expr, symbol)


def function_rule(derivative):
    return FunctionRule(derivative.expr, derivative.symbol)


@evaluates(ConstantRule)
def eval_constant(*args):
    return 0


@evaluates(ConstantTimesRule)
def eval_constanttimes(constant, other, substep, expr, symbol):
    return constant * diff(substep)


@evaluates(AddRule)
def eval_add(substeps, expr, symbol):
    results = [diff(step) for step in substeps]
    return sum(results)


@evaluates(DivRule)
def eval_div(numer, denom, numerstep, denomstep, expr, symbol):
    d_numer = diff(numerstep)
    d_denom = diff(denomstep)
    return (denom * d_numer - numer * d_denom) / (denom ** 2)


@evaluates(ChainRule)
def eval_chain(substep, inner, u_var, innerstep, expr, symbol):
    return diff(substep).subs(u_var, inner) * diff(innerstep)


@evaluates(PowerRule)
@evaluates(ExpRule)
@evaluates(LogRule)
@evaluates(DontKnowRule)
@evaluates(FunctionRule)
def eval_default(*args):
    func, symbol = args[-2], args[-1]

    if isinstance(func, sympy.Symbol):
        func = sympy.Pow(func, 1, evaluate=False)

    # Automatically derive and apply the rule (don't use diff() directly as
    # chain rule is a separate step)
    substitutions = []
    mapping = {}
    constant_symbol = sympy.Dummy()
    for arg in func.args:
        if symbol in arg.free_symbols:
            mapping[symbol] = arg
            substitutions.append(symbol)
        else:
            mapping[constant_symbol] = arg
            substitutions.append(constant_symbol)

    rule = func.func(*substitutions).diff(symbol)
    return rule.subs(mapping)


@evaluates(MulRule)
def eval_mul(terms, substeps, expr, symbol):
    diffs = list(map(diff, substeps))

    result = sympy.S.Zero
    for i in range(len(terms)):
        subresult = diffs[i]
        for index, term in enumerate(terms):
            if index != i:
                subresult *= term
        result += subresult
    return result


@evaluates(TrigRule)
def eval_default_trig(*args):
    return sympy.trigsimp(eval_default(*args))


@evaluates(RewriteRule)
def eval_rewrite(rewritten, substep, expr, symbol):
    return diff(substep)


@evaluates(AlternativeRule)
def eval_alternative(alternatives, expr, symbol):
    return diff(alternatives[1])


def diff_steps(expr, symbol):
    deriv = DerivativeInfo(expr, symbol)

    def key(deriv):
        expr = deriv.expr
        if isinstance(expr, TrigonometricFunction):
            return TrigonometricFunction
        elif isinstance(expr, AppliedUndef):
            return AppliedUndef
        elif not expr.has(symbol):
            return 'constant'
        else:
            return expr.func

    return switch(key, {
        sympy.Pow: power_rule,
        sympy.Symbol: power_rule,
        sympy.Dummy: power_rule,
        sympy.Add: add_rule,
        sympy.Mul: mul_rule,
        TrigonometricFunction: trig_rule,
        sympy.exp: exp_rule,
        sympy.log: log_rule,
        AppliedUndef: function_rule,
        'constant': constant_rule
    })(deriv)


def diff(rule):
    try:
        return evaluators[rule.__class__](*rule)
    except KeyError:
        raise ValueError("Cannot evaluate derivative")


class DiffPrinter(object):
    def __init__(self, rule):
        self.print_rule(rule)
        self.rule = rule

    def print_rule(self, rule):
        if isinstance(rule, PowerRule):
            self.print_Power(rule)
        elif isinstance(rule, ChainRule):
            self.print_Chain(rule)
        elif isinstance(rule, ConstantRule):
            self.print_Number(rule)
        elif isinstance(rule, ConstantTimesRule):
            self.print_ConstantTimes(rule)
        elif isinstance(rule, AddRule):
            self.print_Add(rule)
        elif isinstance(rule, MulRule):
            self.print_Mul(rule)
        elif isinstance(rule, DivRule):
            self.print_Div(rule)
        elif isinstance(rule, TrigRule):
            self.print_Trig(rule)
        elif isinstance(rule, ExpRule):
            self.print_Exp(rule)
        elif isinstance(rule, LogRule):
            self.print_Log(rule)
        elif isinstance(rule, DontKnowRule):
            self.print_DontKnow(rule)
        elif isinstance(rule, AlternativeRule):
            self.print_Alternative(rule)
        elif isinstance(rule, RewriteRule):
            self.print_Rewrite(rule)
        elif isinstance(rule, FunctionRule):
            self.print_Function(rule)
        else:
            self.append(repr(rule))

    def print_Power(self, rule):
        with self.new_step():
            self.append("Aplicando la regla de potencia a: {0} se obtiene {1}".format(
                self.format_math(rule.context),
                self.format_math(diff(rule))))

    def print_Number(self, rule):
        with self.new_step():
            self.append("La derivada de la constante {} es cero.".format(
                self.format_math(rule.number)))

    def print_ConstantTimes(self, rule):
        with self.new_step():
            self.append("La derivada de N veces una función "
                        "es N veces la derivada de la función")
            with self.new_level():
                self.print_rule(rule.substep)
            self.append("Así, el resultado es: {}".format(
                self.format_math(diff(rule))))

    def print_Add(self, rule):
        with self.new_step():
            self.append("Diferenciando {} término por término:".format(
                self.format_math(rule.context)))
            with self.new_level():
                for substep in rule.substeps:
                    self.print_rule(substep)
            self.append("El resultado es: {}".format(
                self.format_math(diff(rule))))

    def print_Mul(self, rule):
        with self.new_step():
            self.append("Aplicando la regla del producto:".format(
                self.format_math(rule.context)))

            fnames = list(map(lambda n: sympy.Function(n)(rule.symbol),
                              functionnames(len(rule.terms))))
            derivatives = list(map(lambda f: sympy.Derivative(f, rule.symbol), fnames))
            ruleform = []
            for index in range(len(rule.terms)):
                buf = []
                for i in range(len(rule.terms)):
                    if i == index:
                        buf.append(derivatives[i])
                    else:
                        buf.append(fnames[i])
                ruleform.append(reduce(lambda a, b: a * b, buf))
            self.append(self.format_math_display(
                sympy.Eq(sympy.Derivative(reduce(lambda a, b: a * b, fnames),
                                          rule.symbol),
                         sum(ruleform))))

            for fname, deriv, term, substep in zip(fnames, derivatives,
                                                   rule.terms, rule.substeps):
                self.append("{}; para hallar {}:".format(
                    self.format_math(sympy.Eq(fname, term)),
                    self.format_math(deriv)
                ))
                with self.new_level():
                    self.print_rule(substep)

            self.append("El resultado es: " + self.format_math(diff(rule)))

    def print_Div(self, rule):
        with self.new_step():
            f, g = rule.numerator, rule.denominator
            fp, gp = f.diff(rule.symbol), g.diff(rule.symbol)
            x = rule.symbol
            ff = sympy.Function("f")(x)
            gg = sympy.Function("g")(x)
            qrule_left = sympy.Derivative(ff / gg, rule.symbol)
            qrule_right = sympy.ratsimp(sympy.diff(sympy.Function("f")(x) /
                                                   sympy.Function("g")(x)))
            qrule = sympy.Eq(qrule_left, qrule_right)
            self.append("Aplicando la regla del cociente que es:")
            self.append(self.format_math_display(qrule))
            self.append("{} y {}.".format(self.format_math(sympy.Eq(ff, f)),
                                          self.format_math(sympy.Eq(gg, g))))
            self.append("Para hallar {}:".format(self.format_math(ff.diff(rule.symbol))))
            with self.new_level():
                self.print_rule(rule.numerstep)
            self.append("Para hallar {}:".format(self.format_math(gg.diff(rule.symbol))))
            with self.new_level():
                self.print_rule(rule.denomstep)
            self.append("Sutituyendo en la regla del cociente:")
            self.append(self.format_math(diff(rule)))

    def print_Chain(self, rule):
        with self.new_step(), self.new_u_vars() as (u, du):
            self.append("Sea {}.".format(self.format_math(sympy.Eq(u, rule.inner))))
            self.print_rule(replace_u_var(rule.substep, rule.u_var, u))
        with self.new_step():
            if isinstance(rule.innerstep, FunctionRule):
                self.append(
                    "Entonces, aplicando la regla de la cadena. Multipicamos por {}:".format(
                        self.format_math(
                            sympy.Derivative(rule.inner, rule.symbol))))
                self.append(self.format_math_display(diff(rule)))
            else:
                self.append(
                    "Entonces, aplicando la regla de la cadena. Multipicamos por {}:".format(
                        self.format_math(
                            sympy.Derivative(rule.inner, rule.symbol))))
                with self.new_level():
                    self.print_rule(rule.innerstep)
                self.append("El resultado de aplicar la regla de la cadena:")
                self.append(self.format_math_display(diff(rule)))

    def print_Trig(self, rule):
        with self.new_step():
            if isinstance(rule.f, sympy.sin):
                self.append("La derivada del seno es el coseno:")
            elif isinstance(rule.f, sympy.cos):
                self.append("La derivada del coseno es el negativo del seno:")
            elif isinstance(rule.f, sympy.sec):
                self.append("La derivada de la secante es secante por tangente:")
            elif isinstance(rule.f, sympy.csc):
                self.append("La derivada de la cosecante es el negativo de la cosecante por la cotangente:")
            self.append("{}".format(
                self.format_math_display(sympy.Eq(
                    sympy.Derivative(rule.f, rule.symbol),
                    diff(rule)))))

    def print_Exp(self, rule):
        with self.new_step():
            if rule.base == sympy.E:
                self.append("La derivada de {} es ella misma.".format(
                    self.format_math(sympy.exp(rule.symbol))))
            else:
                self.append(
                    self.format_math(sympy.Eq(sympy.Derivative(rule.f, rule.symbol),
                                              diff(rule))))

    def print_Log(self, rule):
        with self.new_step():
            if rule.base == sympy.E:
                self.append("La derivada de {} es {}.".format(
                    self.format_math(rule.context),
                    self.format_math(diff(rule))
                ))
            else:
                # This case shouldn't come up often, seeing as SymPy
                # automatically applies the change-of-base identity
                self.append("La derivada de {} es {}.".format(
                    self.format_math(sympy.log(rule.symbol, rule.base,
                                               evaluate=False)),
                    self.format_math(1 / (rule.arg * sympy.ln(rule.base)))))
                self.append("Por lo tanto {}".format(
                    self.format_math(sympy.Eq(
                        sympy.Derivative(rule.context, rule.symbol),
                        diff(rule)))))

    def print_Alternative(self, rule):
        with self.new_step():
            self.append("Hay muchas formas de efectuar la derivada.")
            self.append("Una forma:")
            with self.new_level():
                self.print_rule(rule.alternatives[0])

    def print_Rewrite(self, rule):
        with self.new_step():
            self.append("Reescribimos la función para ser derivada:")
            self.append(self.format_math_display(
                sympy.Eq(rule.context, rule.rewritten)))
            self.print_rule(rule.substep)

    def print_Function(self, rule):
        with self.new_step():
            self.append("Trivial:")
            self.append(self.format_math_display(
                sympy.Eq(sympy.Derivative(rule.context, rule.symbol),
                         diff(rule))))

    def print_DontKnow(self, rule):
        with self.new_step():
            self.append("Don't know the steps in finding this derivative.")
            self.append("But the derivative is")
            self.append(self.format_math_display(diff(rule)))


class HTMLPrinter(DiffPrinter, HTMLPrinterP):
    def __init__(self, rule):
        self.alternative_functions_printed = set()
        HTMLPrinterP.__init__(self)
        DiffPrinter.__init__(self, rule)

    def print_Alternative(self, rule):
        if rule.context.func in self.alternative_functions_printed:
            self.print_rule(rule.alternatives[0])
        elif len(rule.alternatives) == 2:
            self.alternative_functions_printed.add(rule.context.func)
            self.print_rule(rule.alternatives[1])
        else:
            self.alternative_functions_printed.add(rule.context.func)
            with self.new_step():
                self.append("Hay muchas formas de efectuar la derivada.")
                for index, r in enumerate(rule.alternatives[1:]):
                    with self.new_collapsible():
                        self.append_header("Método #{}".format(index + 1))
                        with self.new_level():
                            self.print_rule(r)

    def finalize(self):
        answer = diff(self.rule)
        if answer:
            simp = sympy.simplify(answer).rewrite(sin, cos)
            simp = sympy.trigsimp(simp).rewrite(sin, cos)
            simp = sympy.factor(simp).rewrite(sin, cos)
            if simp != answer:
                answer = simp
                with self.new_step():
                    self.append("Simplificando:")
                    self.append(self.format_math_display(simp))
            else:
                simp = sympy.expand(answer)
                simp = sympy.trigsimp(simp).rewrite(sin, cos)
                simp = sympy.factor(simp)
                if simp != answer:
                    answer = simp
                    with self.new_step():
                        self.append("Simplificando:")
                        self.append(self.format_math_display(simp))
                else:
                    simp = sympy.trigsimp(answer).rewrite(sin, cos)
                    simp = sympy.factor(simp)
                    if simp != answer:
                        answer = simp
                        with self.new_step():
                            self.append("Simplificando:")
                            self.append(self.format_math_display(simp))
        self.lines.append('</ol><br/>')
        self.lines.append('<hr/>')
        self.level = 0
        self.append('La respuesta es:')
        self.append(self.format_math_display(answer))
        return '\n'.join(self.lines)


def print_html_steps(function, symbol):
    a = HTMLPrinter(diff_steps(function, symbol))
    return a.finalize()


def acomodaNotacion(expresion):
    # parche para notación
    expresion = expresion.replace("\\frac{d}{d x} f{\\left(x \\right)} g{\\left(x \\right)}",
                                  "\\frac{d}{d x}( f{\\left(x \\right)} g{\\left(x \\right)})")
    expresion = expresion.replace("\\frac{d}{d x} \\frac{f{\\left(x \\right)}}{g{\left(x \\right)}}",
                                  "\\frac{d}{d x}( \\frac{f{\\left(x \\right)}}{g{\left(x \\right)}})")
    expresion = expresion.replace(
        "\\frac{- f{\\left(x \\right)} \\frac{d}{d x} g{\\left(x \\right)} + g{\\left(x \\right)} \\frac{d}{d x} f{\\left(x \\right)}}{g^{2}{\\left(x \\right)}}",
        "\\frac{- f{\\left(x \\right)} \\frac{d}{d x}( g{\\left(x \\right)}) + g{\\left(x \\right)} \\frac{d}{d x}( f{\\left(x \\right))}}{g^{2}{\\left(x \\right)}}")
    expresion = expresion.replace("\\frac{d}{d x} f{\\left(x \\right)}",
                                  "\\frac{d}{d x}( f{\\left(x \\right)})")
    expresion = expresion.replace("\\frac{d}{d x} g{\\left(x \\right)}",
                                  "\\frac{d}{d x}( g{\\left(x \\right)})")
    return expresion


##MAIN##

salida = open("/tmp/solucion_87ae5456-1344-4973-86e9-073c1fe60099.txt", "w")
x = symbols('x')
expr = parse_latex(r"3x^2-6x+2").subs({Symbol('pi'): pi})
salida.write("Obtener: $$%s$$<br><br>" % latex(Derivative(expr, x)))
solucion = print_html_steps(expr, x)
solucion = acomodaNotacion(solucion)
salida.write(solucion)
derivada = Derivative(expr)
x0 = 5
y_0 = expr.subs(x, x0)
yp_0 = derivada.subs(x, x0)
salida.write("\n $$x_{0}=0$$\n<br/>")
salida.write("$$f(x_{0})=%s$$ \n<br/>" % latex(y_0))
solucion="$$f'(x_{0})=%s=%s$$ \n<br/>" % (latex(yp_0), latex(yp_0.doit()))
solucion=solucion.replace("+-","-")
solucion = solucion.replace("--","+")
salida.write(solucion)
salida.write("Sustituyendo en $$y-f(x_{0})=f'(x_{0})(x-x_{0})$$ obtenemos:\n<br/>$$y-%s=%s(x-%s)$$ \n<br/>" % (
latex(y_0.doit()), latex(yp_0.doit()), x0))
solucion="Simplificando:\n<br/>$$y=%sx+%s$$"%(latex(yp_0.doit()),latex(y_0.doit()-yp_0.doit()*x0))
solucion=solucion.replace("+-","-")
solucion = solucion.replace("--","+")
salida.write(solucion)
salida.close()
