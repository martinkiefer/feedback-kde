set terminal pdf

set style data boxplot
set style boxplot nooutlier
set style fill solid 0.25
set xtics (512, 1024, 2048, 4096, 8092, 16384, 32768, 65536)
set ylabel "Absolute estimation error in tuples"
set xlabel "Modelsize in tuples (log-scaled)"
set grid ytics
set key above box
set xrange [-0.2 : 8 ]
plot 'kde_heuristic.dat' using  (0.0):2:(0.1):1 title 'Heuristic' , \
     'kde_batch.dat' using  (0.0):2:(0.1):1 title 'Batch', \
     'kde_adaptive.dat' using  (0.0):2:(0.1):1 title 'Adaptive'
