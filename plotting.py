#instalar matplotlib y seaborn
import sympy as syp
from sympy.plotting import plot
import seaborn as sns
sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})
x = syp.symbols('x')
p1 = syp.plot((x+7)*(x-5) , show=False)
p2 = syp.plot((x-5)**3, show=False)
#p2.xlim=-40,40
p1.append(p2[0])
p1.xlim=-40,40
p1.ylim=-40,40
p1.title='Hola'
#p1.autoscale=True
#p1.show()
p1.save('/home/yan/graph.svg')