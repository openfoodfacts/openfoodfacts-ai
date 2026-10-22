set datafile separator comma

set terminal pngcairo size 1500,900 enhanced
set output "graph.png"
# set terminal svg size 1500,900
# set output "graph.svg"

set style data histograms
set style histogram clustered gap 1
set style fill solid noborder
set boxwidth 0.85

set title "LLM benchmark: price tag extraction" font ",18"

set border 3
set ylabel "Accuracy (%)" font ",14"
set yrange [80:102]
set ytics 80,10,100 nomirror
set grid ytics lt 0 lc rgb "#dddddd"

set xtics nomirror rotate by 30 right font ",12"
set bmargin 9

set key top center horizontal opaque font ",13"

plot "results.csv" every ::1 using 2:xtic(1) title "Price"   lc rgb "#6f7ca8", \
     ""            every ::1 using 3  title "Barcode" lc rgb "#c4917a", \
     ""            every ::1 using ($0 - 1.0/6):2:(sprintf("%.1f",$2)) with labels offset 0,1 font ",10" tc rgb "#3b4463" notitle, \
     ""            every ::1 using ($0 + 1.0/6):3:(sprintf("%.1f",$3)) with labels offset 0,1 font ",10" tc rgb "#7a4a38" notitle
