from sympy import *
from sympy.parsing.latex import parse_latex

# Solución a derivada de y=u^n*v^m


salida = open("/tmp/salida.txt", "w")
init_printing()
x = var('x')
u = var('u')
n = var('n')
v = var('v')
m = var('m')
ux = parse_latex(r"4x+3")
n = 4
vx = parse_latex(r"x+1")
m = -3
f1 = u ** n
f2 = v ** m
fx = ux ** n * vx ** m
salida.write("$$f(x)=%s$$<br>\n" % latex(fx))
salida.write("$$f'(x)=%s$$<br>\n" % latex(diff(fx)))
# u(x)
salida.write("$$u(x)=%s$$<br>\n" % latex(ux))
dfu = diff(u ** n)
salida.write("$$\\frac{d}{du}(%s)=%s$$<br>\n" % (latex(u ** n), latex(dfu)))
dux = diff(ux)
salida.write("$$\\frac{d}{dx}(%s)=%s$$<br>\n" % (latex(ux), latex(dux)))
df1 = dfu * dux
salida.write("$$\\frac{d}{du}\\frac{du}{dx}=%s$$<br>\n" % latex(df1))
df1 = df1.subs(u, ux)
salida.write("$$\\frac{d}{dx}(%s)=%s$$<br>\n" % (latex(f1.subs(u, ux)), latex(df1)))
# v(x)
salida.write("$$v(x)=%s$$<br>\n" % latex(vx))
dfv = diff(v ** m)
salida.write("$$\\frac{d}{dv}(%s)=%s$$<br>\n" % (latex(v ** m), latex(dfv)))
dvx = diff(vx)
salida.write("$$\\frac{d}{dx}(%s)=%s$$<br>\n" % (latex(vx), latex(dvx)))
df2 = dfv * dvx
salida.write("$$\\frac{d}{du}\\frac{du}{dx}=%s$$<br>\n" % latex(df2))
df2 = df2.subs(v, vx)
salida.write("$$\\frac{d}{dx}(%s)=%s$$<br>\n" % (latex(f2.subs(v, vx)), latex(df2)))

salida.write("$$\\frac{d}{dx}(u^{n}(x) v^{m}(x))=v^{m}(x) nu^{n-1}(x)\\frac{du}{dx} + u^{n}(x)mv^{m-1}(x)\\frac{dv}{dx}$$<br>\n")
salida.write("$$\\frac{d}{dx}(%s)=[(%s)(%s)]+[(%s)(%s)]=%s$$<br>\n" % (latex(fx),latex(df1),latex(vx**m), latex(df2), latex(ux**n), latex(df1*vx**m + df2*ux**n)))
