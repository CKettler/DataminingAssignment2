import pstats
p = pstats.Stats('main.cprof')
p.sort_stats('cumtime').print_stats(10)