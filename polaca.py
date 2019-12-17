from sympy import *
from sympy.parsing.latex import parse_latex

def pre(expr):
    print(expr.func)
    print(expr)
    for arg in expr.args:
        pre(arg)

# Solución a derivada de y=u^n*v^m


salida = open("/tmp/salida.txt", "w")
init_printing()
x = var('x')
u = var('u')
n = var('n')
v = var('v')
m = var('m')
ux = parse_latex(r"x \tan(2\sqrt{x})")
#pre(ux)
print("PRE_TRANS")
for arg in postorder_traversal(ux):
#    print(arg.func)
#    print(arg)
    clase = arg.__class__.__name__
    if clase != "Integer" and clase != "Symbol" and clase != "Half":
        print("Clase")
        print(clase)
        print("Argumento")
        print(arg)
        print(arg.expr_free_symbols)
        print("Diff")
        d = Derivative(arg)
        print(d)

