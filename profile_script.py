import cProfile
import pstats
from aura_deep_face import main

# Exécuter le profilage
cProfile.run('main()', 'output.prof')

# Lire et afficher les résultats
p = pstats.Stats('output.prof')
p.sort_stats('cumulative').print_stats(20)