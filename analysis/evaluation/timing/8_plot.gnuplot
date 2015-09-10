set terminal pdf
set key above vertical maxrows 2 width -3 samplen 2 box
set xlabel "Modelsize [Points in sample / Buckets in histogram]"
set ylabel "Average estimator runtime [ms]" offset 2.4
set logscale x
set logscale y
set grid ytics 
set xrange [900:2500000]
set xtics ("1K" 1024, "2K" 2048, "4K" 4096, "8K" 8192, "16K" 16384, "32K" 32768, "64K" 65536, "128K" 131072, "256K" 262144, "512K" 524288, "1M" 1048576, "2M" 2097152)
plot "8_kde_heuristic_cpu.dat" title "Heuristic (CPU)", "8_kde_adaptive_cpu.dat" title "Adaptive (CPU)", "8_kde_heuristic_gpu.dat" title "Heuristic (GPU)", "8_kde_adaptive_gpu.dat" title "Adaptive (GPU)", "8_stholes_cpu.dat" title "STHoles" 
