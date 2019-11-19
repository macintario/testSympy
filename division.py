from sympy import *
from sympy.parsing.latex import parse_latex

salida = open("/tmp/salida.txt","w")
init_printing()


u = parse_latex(r"5")
#v = UnevaluatedExpr(sqrt(7*x-2))
#v = UnevaluatedExpr(parse_latex(r"\sqrt{5x-25}"))
v = parse_latex(r"\sqrt{5x-25}")
expresion = UnevaluatedExpr(u/v)
salida.write("$$\\displaystyle f(x)=\\frac{u}{v}=%s$$\n" % latex(expresion))
salida.write("$$\\displaystyle u=%s$$\n" % latex(u))
salida.write("$$\\displaystyle u'=%s$$\n" % latex(diff(u)))
salida.write("$$\\displaystyle v=%s$$\n" % latex(v))
salida.write("$$\\displaystyle v'=%s$$\n" % latex(diff(v)))
#resultado = sympify( v*diff(u)-u*diff(v), evaluate=False)
#salida.write("$$\displaystyle f'=%s$$\n" % latex(diff(expresion)))
resultado = sympify(diff(u/v), evaluate=False)
salida.write("$$\\displaystyle f'(x)=\\frac{u\'v -v\'u}{v^2} = \\frac{%s\\times%s - %s\\times%s}{%s^2}$$\n" %(latex(v),latex(diff(u)),latex(u),latex(diff(v)),latex(v)))
salida.write("$$\displaystyle f'(x)=%s$$\n" % latex(resultado.doit()))
#result = sp.diff('5/sqrt(5*x-6)','x')
#salida.write(sp.latex(result))
salida.close()