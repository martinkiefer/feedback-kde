set terminal pdf dashed
set output "out8.pdf"
set y2range [0:6500]
set ylabel "Averaged absolute selectivity estimation error"
set y2label "Tuples in database"
set y2tics
set xlabel "Queries"
set key above vertical maxrows 1 width -1 samplen 2
plot "8_heuristic_aggregated.log" using 1 with lines title "KDE heuristic" axis x1y1 lt 12, "8_adaptive_tkr_aggregated.log" using 1 with lines title "KDE adaptive" axis x1y1 lt 8, "8_stholes_aggregated.log" using 1 with lines title "STHoles" axis x1y1 lt 11, "8_adaptive_tkr_aggregated.log" using 2 with lines title "Tuples" axis x1y2 lt 16

