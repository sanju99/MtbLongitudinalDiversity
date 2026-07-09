set terminal png tiny size 800,800
set output "/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/MAv/mummer/alignment.png"
set size 1,1
set grid
unset key
set border 15
set tics scale 0
set xlabel "S0344-01_contig_1_pilon"
set ylabel "S0346-01_contig_1_pilon"
set format "%.0f"
set mouse format "%.0f"
set mouse mouseformat "[%.0f, %.0f]"
if(GPVAL_VERSION < 5) set mouse clipboardformat "[%.0f, %.0f]"
set xrange [1:5575292]
set yrange [1:5573596]
set style line 1  lt 2 lw 3 pt 6 ps 1
set style line 2  lt 2 lw 3 pt 6 ps 1
set style line 3  lt 1 lw 3 pt 6 ps 1
plot \
 "/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/MAv/mummer/alignment.fplot" title "FWD" w lp ls 1, \
 "/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/MAv/mummer/alignment.rplot" title "REV" w lp ls 2, \
 "/home/sak0914/MtbLongitudinalDiversity/hybrid_assemblies/MAv/mummer/alignment.hplot" title "HLT" w lp ls 3